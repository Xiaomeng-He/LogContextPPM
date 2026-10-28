from .utils import read_config

TRAINING_DEFAULTS = {"batch_size": 64, "lr": 0.0002, "num_epochs": 200,
                     "val_batch_size": 128, "test_batch_size": 256}

def load_model_config(config_path, dataset_name, model_keys):
  
    configs = read_config(config_path)
    if dataset_name not in configs:
        raise KeyError(f"Dataset '{dataset_name}' not found in {config_path}")
    cfg = configs[dataset_name]

    task_hparams = {}
    for task in ("suffix", "rm"):
        hparams = {k: cfg[k] for k in model_keys if k in cfg}
        hparams.update(cfg.get(task) or {})
        missing = [k for k in model_keys if k not in hparams]
        if missing:
            raise KeyError(f"{config_path.name} [{dataset_name}] is missing {missing} for task '{task}'")
        task_hparams[task] = hparams

    training = {**TRAINING_DEFAULTS, **(cfg.get("training") or {})}

    return cfg, task_hparams, training
