import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
EOC_INDEX = 3
NUME_PAD = -10000.0

def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)

def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lens_till_eoc(x, eoc_index=EOC_INDEX):
    T = x.shape[1]
    is_eoc = x == eoc_index
    first_eoc = is_eoc.int().argmax(dim=1)
    return torch.where(is_eoc.any(dim=1), first_eoc + 1, torch.full_like(first_eoc, T))

def left_to_right_padding(cate_prefix, nume_prefix):
    T = cate_prefix.shape[1]
    lengths = (cate_prefix[..., 0] != 0).sum(dim=1, keepdim=True)
    positions = torch.arange(T).unsqueeze(0)
    valid = positions < lengths
    src = (positions + T - lengths).clamp(max=T - 1)

    def shift(x, pad_value):
        out = torch.gather(x, 1, src.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        return out.masked_fill(~valid.unsqueeze(-1), pad_value)

    return shift(cate_prefix, 0), shift(nume_prefix, NUME_PAD)

def load_dataset(data_dir, split, log_prefix_len, trace_padding):
    tensor_dir = Path(data_dir) / "Tensors"
    pad = {"left": "l", "right": "r"}[trace_padding]

    def load(name):
        return torch.load(tensor_dir / f"{split}_{name}.pt", weights_only=True)

    log_cate_prefix = load("log_cate_prefix")[:, -log_prefix_len:, :].clone()
    log_nume_prefix = load("log_nume_prefix")[:, -log_prefix_len:, :].clone()
    if trace_padding == "right":
        log_cate_prefix, log_nume_prefix = left_to_right_padding(log_cate_prefix, log_nume_prefix)

    trace_cate_prefix = load(f"trace_cate_prefix_{pad}")
    trace_nume_prefix = load(f"trace_nume_prefix_{pad}")

    act_suffix = load("act_suffix")
    dec_input_act = act_suffix[:, :-1].clone()
    dec_input_act[dec_input_act == EOC_INDEX] = 0
    tgt_act = act_suffix[:, 1:].clone()
    tgt_rm = load("rm_suffix")[:, 0].clone()

    return TensorDataset(log_cate_prefix, log_nume_prefix, trace_cate_prefix, trace_nume_prefix,
                         dec_input_act, tgt_act, tgt_rm)


def load_rm_stats(data_dir):
    mean_dict = torch.load(Path(data_dir) / "mean_dict.pt", weights_only=True)
    std_dict = torch.load(Path(data_dir) / "std_dict.pt", weights_only=True)
    return float(mean_dict["rm"]), float(std_dict["rm"])


def check_config(cfg, log_info, model_name, dataset_name, extra_checks=()):
    checks = [
        ("num_act", cfg["num_act"], log_info["num_act"]),
        ("input_dims", list(cfg["input_dims"]), log_info["input_dims"]),
        ("num_nume_feature", cfg["num_nume_feature"], log_info["num_nume_feature"]),
        *extra_checks,
    ]
    errors = [f"  {name}: config has {got}, tensors require {expected}"
              for name, got, expected in checks if got != expected]

    if cfg["log_prefix_len"] > log_info["log_prefix_len"]:
        errors.append(f"  log_prefix_len: config has {cfg['log_prefix_len']}, "
                      f"but tensors were created with at most {log_info['log_prefix_len']}")

    if errors:
        raise ValueError(f"{model_name}_hparams.yml [{dataset_name}] does not match "
                         f"log_info.yml:\n" + "\n".join(errors))

def save_csv(rows, path):
    if not rows:
        return
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
