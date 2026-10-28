from pathlib import Path

from Models.config import load_model_config
from Models.pipeline import run_pipeline
from Models.utils import check_config
from .create_LSTM import LSTM_integrate_suffix, LSTM_integrate_rm, init_weights_uni

CONFIG_PATH = Path(__file__).resolve().parent / "LSTM_hparams.yml"
MODEL_KEYS = ("hidden_size", "dropout", "num_layers")

def embedding_sizes(num_act, input_dims):
    act_embed = round(1.6 * (num_act ** 0.56))
    emb_dims = [min(600, round(1.6 * (input_dim ** 0.56))) for input_dim in input_dims]
    return act_embed, emb_dims

def train_evaluate_LSTM(dataset_name, tasks=("suffix", "rm"), seed=42, train=True, config_path=CONFIG_PATH):
    cfg, task_hparams, training = load_model_config(config_path, dataset_name, MODEL_KEYS)

    num_act = cfg["num_act"]
    input_dims = cfg["input_dims"]
    act_embed, emb_dims = embedding_sizes(num_act, input_dims)
    enc_input_size = act_embed + sum(emb_dims) + cfg["num_nume_feature"]
    dec_input_size = act_embed

    def build_model(task, hp):
        if task == "suffix":
            return LSTM_integrate_suffix(act_embed, input_dims, emb_dims, num_act, enc_input_size,
                                         hp["hidden_size"], hp["num_layers"], hp["dropout"], dec_input_size)
        return LSTM_integrate_rm(act_embed, input_dims, emb_dims, num_act, enc_input_size,
                                 hp["hidden_size"], hp["num_layers"], hp["dropout"])

    def check(log_info):
        check_config(cfg, log_info, "LSTM", dataset_name)

    return run_pipeline("LSTM", dataset_name, cfg, task_hparams, training, build_model,
                        trace_padding="left", tasks=tasks, seed=seed, train=train,
                        init_weights=init_weights_uni, check=check)
