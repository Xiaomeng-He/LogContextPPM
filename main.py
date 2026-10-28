from __future__ import annotations

import argparse
import statistics
import warnings
from dataclasses import dataclass, fields
from typing import Literal

from Preprocessing.create_tensor_pipeline import create_tensors
from Models.LSTM.LSTM_pipeline import train_evaluate_LSTM
from Models.Transformer.Transformer_pipeline import train_evaluate_Transformer

warnings.filterwarnings("ignore", message=".*nested tensors.*")

Model = Literal["LSTM", "Transformer"]
Task = Literal["suffix", "rm", "both"]

@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str = "BPIC12"        # key in the yaml config files
    model: Model = "LSTM"               # predictive model
    task: Task = "both"                 # activity suffix, remaining time, or both
    seeds: tuple = (42,)                # one training run per seed
    create_tensors: bool = True         # set to False to reuse existing tensors
    train: bool = True                  # set to False to evaluate existing checkpoints

def run_experiment(config: ExperimentConfig):
    if config.create_tensors:
        create_tensors(config.dataset_name)
        print(f"Tensors created for {config.dataset_name}")

    tasks = ("suffix", "rm") if config.task == "both" else (config.task,)
    train_evaluate = {"LSTM": train_evaluate_LSTM, "Transformer": train_evaluate_Transformer}[config.model]

    return [train_evaluate(config.dataset_name, tasks=tasks, seed=seed, train=config.train)
            for seed in config.seeds]

def report(config: ExperimentConfig, all_results):
    labels = {"test_dl_distance": "Test DL distance",
              "test_mae_len": "Test MAE length",
              "test_mae_rm_hours": "Test MAE RM (hours)"}

    print(f"{'Dataset':<20}: {config.dataset_name}")
    print(f"{'Model':<20}: {config.model}")
    print(f"{'Seeds':<20}: {', '.join(map(str, config.seeds))}")
    for key, label in labels.items():
        values = [r[key] for r in all_results if key in r]
        if not values:
            continue
        line = f"{label:<20}: {statistics.mean(values):.4f}"
        if len(values) > 1:
            line += f" ± {statistics.stdev(values):.4f}"
        print(line)

def parse_args():
    defaults = {f.name: f.default for f in fields(ExperimentConfig)}
    parser = argparse.ArgumentParser(description="Create tensors, then train and evaluate a model.")
    parser.add_argument("--dataset", default=defaults["dataset_name"])
    parser.add_argument("--model", choices=["LSTM", "Transformer"], default=defaults["model"])
    parser.add_argument("--task", choices=["suffix", "rm", "both"], default=defaults["task"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults["seeds"]))
    parser.add_argument("--skip_tensors", action="store_true", help="reuse existing tensors")
    parser.add_argument("--eval_only", action="store_true", help="evaluate existing checkpoints")
    args = parser.parse_args()

    return ExperimentConfig(dataset_name=args.dataset, model=args.model, task=args.task,
                            seeds=tuple(args.seeds), create_tensors=not args.skip_tensors,
                            train=not args.eval_only)

if __name__ == "__main__":
    config = parse_args()
    results = run_experiment(config)
    report(config, results)
