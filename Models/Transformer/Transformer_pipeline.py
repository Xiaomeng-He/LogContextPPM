from pathlib import Path

from Models.config import load_model_config
from Models.pipeline import run_pipeline
from Models.utils import check_config
from .create_Transformer import Transformer_integrate_suffix, Transformer_integrate_rm

CONFIG_PATH = Path(__file__).resolve().parent / "Transformer_hparams.yml"
MODEL_KEYS = ("d_model", "num_heads", "d_ff", "dropout", "num_layers")

def embedding_sizes(num_act, input_dims):
    act_embed = round(1.6 * (num_act ** 0.56))
    emb_dims = [min(600, round(1.6 * (input_dim ** 0.56))) for input_dim in input_dims]
    return act_embed, emb_dims

def train_evaluate_Transformer(dataset_name, tasks=("suffix", "rm"), seed=42, train=True,
                               config_path=CONFIG_PATH):
    cfg, task_hparams, training = load_model_config(config_path, dataset_name, MODEL_KEYS)

    num_act = cfg["num_act"]
    input_dims = cfg["input_dims"]
    suffix_len = cfg["suffix_len"]
    max_seq_len = max(cfg["log_prefix_len"], cfg["trace_prefix_len"], suffix_len)
    act_embed, emb_dims = embedding_sizes(num_act, input_dims)
    enc_input_size = act_embed + sum(emb_dims) + cfg["num_nume_feature"]
    dec_input_size = act_embed

    def build_model(task, hp):
        model_hp = (hp["d_model"], hp["num_heads"], hp["d_ff"], hp["dropout"], hp["num_layers"])
        if task == "suffix":
            return Transformer_integrate_suffix(max_seq_len, suffix_len, act_embed, input_dims, emb_dims,
                                                enc_input_size, dec_input_size, num_act, *model_hp)
        return Transformer_integrate_rm(max_seq_len, act_embed, input_dims, emb_dims,
                                        enc_input_size, num_act, *model_hp)

    def check(log_info):
        check_config(cfg, log_info, "Transformer", dataset_name, extra_checks=[
            ("trace_prefix_len", cfg["trace_prefix_len"], log_info["trace_prefix_len"]),
            ("suffix_len", suffix_len, log_info["trace_suffix_len"] - 1),
        ])

    return run_pipeline("Transformer", dataset_name, cfg, task_hparams, training, build_model,
                        trace_padding="right", tasks=tasks, seed=seed, train=train, check=check)
