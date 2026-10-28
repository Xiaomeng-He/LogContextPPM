import zipfile
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def resolve_path(path):
    """Resolve a path from a config file; relative paths are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_data_file(path):
    """Read an event log stored as .csv or as a .zip archive containing one .csv file."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(f"Expected exactly one .csv file in {path}, found {csv_names}")
            with zf.open(csv_names[0]) as f:
                return pd.read_csv(f)
    raise ValueError(f"Unsupported file type: {path}")
