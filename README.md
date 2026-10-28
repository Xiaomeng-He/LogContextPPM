# LogContextPPM

This repository contains the code for the paper **"Exploiting Event Log Context for Inter-Case-Aware Predictive Process Monitoring"**.

## Repository Structure

```
├── Data/                          event logs (.csv or .zip)
├── Preprocessing/
│   ├── dataset_configs.yml        tensor creation settings per dataset
│   ├── create_tensor_pipeline.py  event log -> tensors
│   ├── preprocessing.py           cleaning, SOC/EOC insertion, time features, encoding, standardization
│   ├── train_test_split.py        time-based train/validation/test split
│   └── create_prefix_suffix.py    log prefix, trace prefix and suffix tensors
├── Models/
│   ├── LSTM/
│   │   ├── LSTM_hparams.yml       LSTM settings per dataset
│   │   ├── LSTM_pipeline.py
│   │   └── create_LSTM.py
│   ├── Transformer/
│   │   ├── Transformer_hparams.yml
│   │   ├── Transformer_pipeline.py
│   │   └── create_Transformer.py
│   ├── pipeline.py                train, checkpoint and evaluate one model
│   ├── train_evaluate.py          training loop, losses and metrics
│   ├── config.py
│   └── utils.py
└── main.py
```

## Installation

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Run Experiments

Place the event log in `Data/` (e.g. `Data/bpic2012.csv`), then run:

```bash
python main.py --dataset BPIC12 --model LSTM
```

This creates the tensors, then trains and evaluates the suffix model and the remaining time model. Options:

| Option | Description |
|---|---|
| `--dataset` | dataset key in the yaml files (default `BPIC12`) |
| `--model` | `LSTM` or `Transformer` (default `LSTM`) |
| `--task` | `suffix`, `rm` or `both` (default `both`) |
| `--seeds` | one or more seeds; with several seeds the mean ± std is reported (default `42`) |
| `--skip_tensors` | reuse the tensors from a previous run |
| `--eval_only` | evaluate saved checkpoints instead of training |

Examples:

```bash
# create the tensors once, then train the Transformer with 10 seeds
python main.py --dataset BPIC12 --model Transformer --seeds 1 2 3 4 5 6 7 8 9 10
python main.py --dataset BPIC12 --model LSTM --skip_tensors --seeds 1 2 3 4 5 6 7 8 9 10

# re-evaluate the remaining time checkpoint of seed 2
python main.py --dataset BPIC12 --model Transformer --task rm --skip_tensors --eval_only --seeds 2
```

Alternatively, edit `ExperimentConfig` in `main.py` and call `run_experiment(config)`.
This generates a synthetic event log (`Data/toy.csv`), creates its tensors and trains and evaluates both models on both tasks for 3 epochs (dataset key `TOY`).
