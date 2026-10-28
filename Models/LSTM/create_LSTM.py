"""LSTM models with a log encoder and a trace encoder."""

import torch
import torch.nn as nn

from Models.utils import lens_till_eoc, EOC_INDEX

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        # enc_states: (batch_size, prefix_len, hidden_size)
        # hidden, cell: (num_layers, batch_size, hidden_size)
        enc_states, (hidden, cell) = self.lstm(x)
        return enc_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, dec_input, hidden, cell):
        outputs, (hidden, cell) = self.lstm(dec_input, (hidden, cell))
        return self.fc(outputs), hidden, cell

def embed_prefix(act_embedding, embeddings, cate_prefix, nume_prefix):
    embedded = [act_embedding(cate_prefix[:, :, 0])]
    if cate_prefix.shape[-1] > 1:
        embedded += [emb(cate_prefix[:, :, i + 1]) for i, emb in enumerate(embeddings)]
    return torch.cat(embedded + [nume_prefix], dim=-1)  # (batch_size, prefix_len, enc_input_size)

class LSTM_integrate_rm(nn.Module):
    def __init__(self, act_embed, input_dims, emb_dims, num_act,
                 enc_input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.act_embedding = nn.Embedding(num_act, act_embed, 0)
        self.embeddings = nn.ModuleList([nn.Embedding(inp, dim, 0) for inp, dim in zip(input_dims, emb_dims)])
        self.trace_encoder = Encoder(enc_input_size, hidden_size, num_layers, dropout)
        self.log_encoder = Encoder(enc_input_size, hidden_size, num_layers, dropout)
        self.rm_fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix):
        log_enc_states, _, _ = self.log_encoder(
            embed_prefix(self.act_embedding, self.embeddings, log_cate_prefix, log_nume_prefix))
        trace_enc_states, _, _ = self.trace_encoder(
            embed_prefix(self.act_embedding, self.embeddings, trace_cate_prefix, trace_nume_prefix))

        last_state = torch.cat((log_enc_states[:, -1, :], trace_enc_states[:, -1, :]), -1)
        return self.rm_fc(last_state).squeeze(-1)  # (batch_size,)

class LSTM_integrate_suffix(nn.Module):
    def __init__(self, act_embed, input_dims, emb_dims, num_act,
                 enc_input_size, hidden_size, num_layers, dropout,
                 dec_input_size, eoc_index=EOC_INDEX):
        super().__init__()
        self.num_act = num_act
        self.eoc_index = eoc_index
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.act_embedding = nn.Embedding(num_act, act_embed, 0)
        self.embeddings = nn.ModuleList([nn.Embedding(inp, dim, 0) for inp, dim in zip(input_dims, emb_dims)])
        self.trace_encoder = Encoder(enc_input_size, hidden_size, num_layers, dropout)
        self.log_encoder = Encoder(enc_input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(dec_input_size, hidden_size * 2, num_layers, dropout, num_act)

    def encode(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix):
        """Return the decoder's initial hidden and cell states: (num_layers, batch_size, 2 * hidden_size)."""
        _, log_hidden, log_cell = self.log_encoder(
            embed_prefix(self.act_embedding, self.embeddings, log_cate_prefix, log_nume_prefix))
        _, trace_hidden, trace_cell = self.trace_encoder(
            embed_prefix(self.act_embedding, self.embeddings, trace_cate_prefix, trace_nume_prefix))

        return torch.cat((log_hidden, trace_hidden), -1), torch.cat((log_cell, trace_cell), -1)

    def forward(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix, dec_input_act):
        """Teacher-forced decoding; returns logits of shape (batch_size, suffix_len, num_act)."""
        hidden, cell = self.encode(log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix)
        act_logits, _, _ = self.decoder(self.act_embedding(dec_input_act), hidden, cell)
        return act_logits

    def argmax(self, log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix, dec_input_act):
        """Greedy decoding starting from the first token of `dec_input_act`."""
        batch_size, suffix_len = dec_input_act.shape
        hidden, cell = self.encode(log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix)

        all_act_logits = []
        dec_act = dec_input_act[:, :1]
        for _ in range(suffix_len):
            act_logits, hidden, cell = self.decoder(self.act_embedding(dec_act), hidden, cell)
            act_logits = act_logits.squeeze(1)
            act_logits[:, [0, 2]] = float("-inf")  # never predict PAD or SOC
            all_act_logits.append(act_logits)
            dec_act = act_logits.argmax(1, keepdim=True)

        all_act_logits = torch.stack(all_act_logits, dim=1)  # (batch_size, suffix_len, num_act)
        all_act_predictions = all_act_logits.argmax(2)
        pred_len = lens_till_eoc(all_act_predictions, self.eoc_index)
        pred_list = [all_act_predictions[i, :int(pred_len[i])].tolist() for i in range(batch_size)]

        return all_act_logits, pred_list, pred_len

def init_weights_uni(module):
    if isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.08, 0.08)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    elif isinstance(module, nn.LSTM):
        for name, parameter in module.named_parameters(recurse=False):
            if "weight" in name:
                nn.init.uniform_(parameter, -0.08, 0.08)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    elif isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -0.08, 0.08)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
