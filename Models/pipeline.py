"""Train and evaluate one model on one dataset for the suffix and/or remaining time task."""

import torch
import yaml
from torch.utils.data import DataLoader

from .train_evaluate import training_trial, evaluate
from .utils import read_config, resolve_path, set_seed, load_dataset, load_rm_stats, save_csv

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pipeline(model_name, dataset_name, cfg, task_hparams, training, build_model,
                 trace_padding, tasks=("suffix", "rm"), seed=42, train=True, init_weights=None,
                 check=None):

    data_dir = resolve_path(cfg["data_dir"])
    output_dir = resolve_path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_info = read_config(data_dir / "log_info.yml")
    if check is not None:
        check(log_info)

    device = get_device()
    rm_mean_value, rm_std_value = load_rm_stats(data_dir)
    log_prefix_len = cfg["log_prefix_len"]

    test_dataloader = DataLoader(load_dataset(data_dir, "test", log_prefix_len, trace_padding),
                                 batch_size=training["test_batch_size"], shuffle=False)
    if train:
        train_dataloader = DataLoader(load_dataset(data_dir, "train", log_prefix_len, trace_padding),
                                      batch_size=training["batch_size"], shuffle=True)
        val_dataloader = DataLoader(load_dataset(data_dir, "val", log_prefix_len, trace_padding),
                                    batch_size=training["val_batch_size"], shuffle=False)

    results = {}
    for task in tasks:
        parameter_path = output_dir / f"{task}_seed{seed}_parameters.pt"
        rm_kwargs = dict(rm_mean_value=rm_mean_value, rm_std_value=rm_std_value) if task == "rm" else {}

        if train:
            set_seed(seed)
            model = build_model(task, task_hparams[task]).to(device)
            if init_weights is not None:
                model.apply(init_weights)

            history = training_trial(model, task, training["lr"], training["num_epochs"],
                                     train_dataloader, val_dataloader, device, parameter_path, **rm_kwargs)
            save_csv(history, output_dir / f"{task}_seed{seed}_loss.csv")
            print(f"[{model_name}] {task} model trained for {len(history)} epochs, saved to {parameter_path}")

        model = build_model(task, task_hparams[task]).to(device)
        model.load_state_dict(torch.load(parameter_path, map_location=device, weights_only=True))
        metric_1, metric_2 = evaluate(model, test_dataloader, device, task, **rm_kwargs)

        if task == "suffix":
            results.update(test_dl_distance=metric_1, test_mae_len=metric_2)
        else:
            results.update(test_mae_rm_hours=metric_1 / 3600, test_mae_rm_seconds=metric_1,
                           test_mse_rm=metric_2)

    results_path = output_dir / f"test_results_seed{seed}.yml"
    saved = read_config(results_path) if results_path.exists() else {}
    saved.update({"dataset": dataset_name, "model": model_name, "seed": seed,
                  **{k: float(v) for k, v in results.items()}})
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(saved, f, sort_keys=False)

    return results
