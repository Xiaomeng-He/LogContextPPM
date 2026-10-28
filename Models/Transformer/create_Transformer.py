"""Transformer models with a log encoder and a trace encoder."""

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MultiheadAttention, Dropout, LayerNorm, Linear, functional as F

from Models.utils import lens_till_eoc, EOC_INDEX

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, padding_mask):
        # x: (batch_size, prefix_len, d_model); padding_mask: (batch_size, prefix_len), True = padding
        return self.encoder(src=x, src_key_padding_mask=padding_mask)

def embed_prefix(act_embedding, embeddings, cate_prefix, nume_prefix):
    """Concatenate the activity embedding, the categorical feature embeddings and the numerical features."""
    embedded = [act_embedding(cate_prefix[:, :, 0])]
    if cate_prefix.shape[-1] > 1:
        embedded += [emb(cate_prefix[:, :, i + 1]) for i, emb in enumerate(embeddings)]
    return torch.cat(embedded + [nume_prefix], dim=-1)  # (batch_size, prefix_len, enc_input_size)

class Transformer_integrate_rm(nn.Module):
    def __init__(self, max_seq_len, act_embed, input_dims, emb_dims, enc_input_size, num_act,
                 d_model, num_heads, d_ff, dropout, num_layers):
        """`max_seq_len` must be at least the longest log or trace prefix."""
        super().__init__()
        self.d_model = d_model

        self.act_embedding = nn.Embedding(num_act, act_embed, 0)
        self.embeddings = nn.ModuleList([nn.Embedding(inp, dim, 0) for inp, dim in zip(input_dims, emb_dims)])

        self.trace_reg_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.trace_reg_token, mean=0.0, std=0.02)
        self.log_reg_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.log_reg_token, mean=0.0, std=0.02)

        self.trace_enc_input_proj = nn.Linear(enc_input_size, d_model)
        self.log_enc_input_proj = nn.Linear(enc_input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_len + 1)
        self.enc_dropout = nn.Dropout(dropout)

        self.trace_encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers)
        self.log_encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers)

        self.fc_rm = nn.Linear(d_model * 2, 1)

    def forward(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix):
        trace_enc_input, trace_padding_mask = self.process_enc_input(trace_cate_prefix, trace_nume_prefix, "trace")
        trace_reg_output = self.trace_encoder(trace_enc_input, trace_padding_mask)[:, 0, :]

        log_enc_input, log_padding_mask = self.process_enc_input(log_cate_prefix, log_nume_prefix, "log")
        log_reg_output = self.log_encoder(log_enc_input, log_padding_mask)[:, 0, :]

        reg_output = torch.cat([trace_reg_output, log_reg_output], dim=1)
        return self.fc_rm(reg_output).squeeze(-1)  # (batch_size,)

    def process_enc_input(self, cate_prefix, nume_prefix, mode):
        """Project the prefix to d_model, prepend the regression token and build the padding mask."""
        if mode == "trace":
            proj, reg_token_param = self.trace_enc_input_proj, self.trace_reg_token
        elif mode == "log":
            proj, reg_token_param = self.log_enc_input_proj, self.log_reg_token
        else:
            raise ValueError(f"Unknown mode: {mode}")

        x = proj(embed_prefix(self.act_embedding, self.embeddings, cate_prefix, nume_prefix))
        batch_size = x.size(0)
        x = torch.cat([reg_token_param.expand(batch_size, -1, -1), x], dim=1)  # (batch_size, prefix_len + 1, d_model)
        x = self.enc_dropout(self.positional_encoding(x))

        padding_mask = cate_prefix[:, :, 0] == 0
        reg_padding_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)

        return x, torch.cat([reg_padding_mask, padding_mask], dim=1)

class DecoderIntegrate(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layers, output_size):
        super().__init__()
        self.layers = nn.ModuleList([
            DualAttentionDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
                                      dropout=dropout, batch_first=True)
            for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, dec_input, dec_padding_mask, dec_causal_mask,
                trace_enc_outputs, trace_enc_padding_mask, log_enc_outputs, log_enc_padding_mask):
        # boolean padding masks (True = padding) -> float masks (-inf = padding), like the causal mask
        dec_padding_mask = torch.where(dec_padding_mask, float("-inf"), 0.0)
        trace_enc_padding_mask = torch.where(trace_enc_padding_mask, float("-inf"), 0.0)
        log_enc_padding_mask = torch.where(log_enc_padding_mask, float("-inf"), 0.0)

        x = dec_input
        for layer in self.layers:
            x = layer(tgt=x, memory_trace=trace_enc_outputs, memory_log=log_enc_outputs,
                      tgt_mask=dec_causal_mask, tgt_key_padding_mask=dec_padding_mask,
                      memory_key_padding_mask_trace=trace_enc_padding_mask,
                      memory_key_padding_mask_log=log_enc_padding_mask, tgt_is_causal=True)

        return self.fc(x)  # (batch_size, suffix_len, num_act)

class Transformer_integrate_suffix(nn.Module):
    def __init__(self, max_seq_len, suffix_len, act_embed, input_dims, emb_dims, enc_input_size, dec_input_size,
                 num_act, d_model, num_heads, d_ff, dropout, num_layers, eoc_index=EOC_INDEX):
        """`max_seq_len` must be at least the longest log prefix, trace prefix and suffix."""
        super().__init__()
        self.suffix_len = suffix_len
        self.d_model = d_model
        self.eoc_index = eoc_index

        self.act_embedding = nn.Embedding(num_act, act_embed, 0)
        self.embeddings = nn.ModuleList([nn.Embedding(inp, dim, 0) for inp, dim in zip(input_dims, emb_dims)])

        self.trace_enc_input_proj = nn.Linear(enc_input_size, d_model)
        self.log_enc_input_proj = nn.Linear(enc_input_size, d_model)
        self.dec_input_proj = nn.Linear(dec_input_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.enc_dropout = nn.Dropout(dropout)
        self.dec_dropout = nn.Dropout(dropout)

        self.trace_encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers)
        self.log_encoder = Encoder(d_model, num_heads, d_ff, dropout, num_layers)
        self.decoder = DecoderIntegrate(d_model, num_heads, d_ff, dropout, num_layers, num_act)

    def encode(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix):
        trace_enc_input, trace_padding_mask = self.process_enc_input(trace_cate_prefix, trace_nume_prefix, "trace")
        trace_enc_outputs = self.trace_encoder(trace_enc_input, trace_padding_mask)

        log_enc_input, log_padding_mask = self.process_enc_input(log_cate_prefix, log_nume_prefix, "log")
        log_enc_outputs = self.log_encoder(log_enc_input, log_padding_mask)

        return trace_enc_outputs, trace_padding_mask, log_enc_outputs, log_padding_mask

    def forward(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix, dec_input_act):
        """Teacher-forced decoding; returns logits of shape (batch_size, suffix_len, num_act)."""
        memory = self.encode(log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix)
        dec_input, dec_padding_mask, dec_causal_mask = self.process_dec_input(dec_input_act)
        return self.decoder(dec_input, dec_padding_mask, dec_causal_mask, *memory)

    def argmax(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix, dec_input_act):
        """Greedy decoding starting from the first token of `dec_input_act`."""
        batch_size = trace_cate_prefix.shape[0]
        memory = self.encode(log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix)

        all_act_logits = []
        dec_act = dec_input_act[:, :1]
        for _ in range(self.suffix_len):
            dec_input, dec_padding_mask, dec_causal_mask = self.process_dec_input(dec_act)
            act_logits = self.decoder(dec_input, dec_padding_mask, dec_causal_mask, *memory)

            next_act_logits = act_logits[:, -1, :].clone()
            next_act_logits[:, [0, 2]] = float("-inf")  # never predict PAD or SOC
            all_act_logits.append(next_act_logits)
            dec_act = torch.cat([dec_act, next_act_logits.argmax(-1, keepdim=True)], dim=1)

        all_act_logits = torch.stack(all_act_logits, dim=1)  # (batch_size, suffix_len, num_act)
        all_act_predictions = all_act_logits.argmax(2)
        act_pred_len = lens_till_eoc(all_act_predictions, self.eoc_index)
        act_pred_list = [all_act_predictions[i, :int(act_pred_len[i])].tolist() for i in range(batch_size)]

        return all_act_logits, act_pred_list, act_pred_len

    def process_enc_input(self, cate_prefix, nume_prefix, mode):
        if mode == "trace":
            proj = self.trace_enc_input_proj
        elif mode == "log":
            proj = self.log_enc_input_proj
        else:
            raise ValueError(f"Unknown mode: {mode}")

        x = proj(embed_prefix(self.act_embedding, self.embeddings, cate_prefix, nume_prefix))
        x = self.enc_dropout(self.positional_encoding(x))

        return x, cate_prefix[:, :, 0] == 0

    def process_dec_input(self, dec_input_act):
        x = self.dec_input_proj(self.act_embedding(dec_input_act))  # (batch_size, T, d_model)
        x = self.dec_dropout(self.positional_encoding(x))

        dec_padding_mask = dec_input_act == 0
        dec_causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(dec_input_act.device)

        return x, dec_padding_mask, dec_causal_mask

class DualAttentionDecoderLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 1e-5,
                 batch_first: bool = True, norm_first: bool = False, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn_trace = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                       bias=bias, **factory_kwargs)
        self.multihead_attn_log = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                     bias=bias, **factory_kwargs)

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.ctx_q_proj = Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.ctx_k_proj = Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.ctx_v_proj = Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.ctx_out_proj = Linear(d_model, d_model, bias=bias, **factory_kwargs)

        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = F.relu if activation == "relu" else F.gelu
        else:
            self.activation = activation

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(self, tgt: Tensor, memory_trace: Tensor, memory_log: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask_trace: Optional[Tensor] = None, memory_mask_log: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask_trace: Optional[Tensor] = None,
                memory_key_padding_mask_log: Optional[Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal_trace: bool = False, memory_is_causal_log: bool = False) -> Tensor:
        x = tgt
        cross_args = (memory_trace, memory_log, memory_mask_trace, memory_mask_log,
                      memory_key_padding_mask_trace, memory_key_padding_mask_log,
                      memory_is_causal_trace, memory_is_causal_log)

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._dual_mha_block(self.norm2(x), *cross_args)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._dual_mha_block(x, *cross_args))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _dual_mha_block(self, x, memory_trace, memory_log, mask_trace, mask_log,
                        padding_mask_trace, padding_mask_log, is_causal_trace, is_causal_log) -> Tensor:
        attn_trace = self.multihead_attn_trace(x, memory_trace, memory_trace, attn_mask=mask_trace,
                                               key_padding_mask=padding_mask_trace, need_weights=False,
                                               is_causal=is_causal_trace)[0]  # (B, L, d)
        attn_log = self.multihead_attn_log(x, memory_log, memory_log, attn_mask=mask_log,
                                           key_padding_mask=padding_mask_log, need_weights=False,
                                           is_causal=is_causal_log)[0]  # (B, L, d)

        # multi-head attention over the two contexts, with the decoder state as query
        B, L, d = x.shape
        H, hd = self.nhead, self.head_dim
        contexts = torch.stack([attn_trace, attn_log], dim=2)  # (B, L, S=2, d)
        S = contexts.shape[2]

        q = self.ctx_q_proj(x).view(B, L, H, hd)
        k = self.ctx_k_proj(contexts).view(B, L, S, H, hd)
        v = self.ctx_v_proj(contexts).view(B, L, S, H, hd)

        scores = torch.einsum("blhd,blshd->blhs", q, k) / (hd ** 0.5)  # (B, L, H, S)
        weights = scores.softmax(dim=-1)
        out = torch.einsum("blhs,blshd->blhd", weights, v)  # (B, L, H, hd)
        fused = self.ctx_out_proj(out.reshape(B, L, d))

        return self.dropout2(fused)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
