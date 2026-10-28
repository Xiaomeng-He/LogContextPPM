import warnings
from pathlib import Path

import torch
import yaml

from .preprocessing import clean_log, sort_log, debiasing, map_case_id, add_soc_eoc, \
    create_time_features, fill_missing, standardize, map_event_name, map_cat_feature, EOC_INDEX
from .train_test_split import get_train_test_split_point, create_table_without_discard_case, \
    get_case_list_by_ratio
from .create_prefix_suffix import create_log_prefix, create_trace_prefix, create_trace_suffix, \
    lens_till_eoc
from .utils import read_config, read_data_file, resolve_path

CONFIG_PATH = Path(__file__).resolve().parent / "dataset_configs.yml"
MIN_SUFFIX_LEN = 3  # keep prefixes whose suffix contains at least one more event before EOC


def create_tensors(dataset_name, config_path=CONFIG_PATH):
    cfg = read_config(config_path)[dataset_name]

    output_dir = resolve_path(cfg["output_dir"])
    tensor_dir = output_dir / "Tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    case_id = cfg["case_id"]
    event_name = cfg["event_name"]
    timestamp = cfg["timestamp"]
    event_idx = "event_idx"

    test_ratio = cfg.get("test_ratio", 0.2)
    val_ratio = cfg.get("val_ratio", 0.25)
    case_len_quantile = cfg.get("case_len_quantile", 1.0)
    split_mode = cfg.get("split_mode", "strict")
    if split_mode != "strict":
        raise NotImplementedError(f"Unsupported split mode: {split_mode}")

    log_prefix_len = cfg["log_prefix_len"]
    trace_prefix_len = cfg["trace_prefix_len"]
    trace_suffix_len = cfg["trace_suffix_len"]

    cat_event_feature = cfg["cat_event_feature"]
    cat_case_feature = cfg["cat_case_feature"]
    num_event_feature = cfg["num_event_feature"]
    num_case_feature = cfg["num_case_feature"]

    trace_time_feature = ["trace_ts_pre", "trace_ts_start"]
    log_time_feature = ["log_ts_pre"]

    categorical_features = [event_name] + cat_event_feature + cat_case_feature
    cat_features = cat_event_feature + cat_case_feature
    continuous_features = (log_time_feature + trace_time_feature + num_event_feature
                           + num_case_feature + ["rm"])
    trace_suffix_col_name = [event_name, "trace_ts_pre", "rm"]

    # preprocessing
    df = read_data_file(resolve_path(cfg["input_path"]))
    if event_name == "combined_activity":
        if not {"concept:name", "lifecycle:transition"}.issubset(df.columns):
            raise KeyError("'combined_activity' requires the columns 'concept:name' and 'lifecycle:transition'.")
        df["combined_activity"] = df["concept:name"] + "-" + df["lifecycle:transition"]

    df = clean_log(df, categorical_features, case_id, event_name, timestamp)
    df = sort_log(df, timestamp)
    df, max_case_len = debiasing(df, case_id, event_name, timestamp, case_len_quantile)
    df, _ = map_case_id(df, case_id)
    df = add_soc_eoc(df, case_id, event_name, timestamp)
    df = create_time_features(df, case_id, timestamp, event_idx)
    df, flag_feature = fill_missing(df, num_event_feature + num_case_feature)

    trace_prefix_col_name = (categorical_features + trace_time_feature + num_event_feature
                             + num_case_feature + flag_feature)
    log_prefix_col_name = (categorical_features + log_time_feature + num_event_feature
                           + num_case_feature + flag_feature)

    # dataset split
    train_test_split_time, _ = get_train_test_split_point(df, test_ratio, case_id, timestamp)
    df_no_discard = create_table_without_discard_case(df, test_ratio, case_id, timestamp)
    full_training_df = df_no_discard[df_no_discard[timestamp] < train_test_split_time]
    train_case_list, val_case_list = get_case_list_by_ratio(full_training_df, val_ratio, case_id, timestamp)
    test_case_list = df[df[timestamp] >= train_test_split_time][case_id].unique()

    # encoding and standardization based on the training set
    training_df = df[df[case_id].isin(train_case_list)].copy()
    df, mapping_dicts = map_cat_feature(df, training_df, cat_features)
    df, event_name_dict = map_event_name(df, training_df, event_name)
    df, mean_dict, std_dict = standardize(df, training_df, continuous_features)

    num_act = max(event_name_dict.values()) + 1
    input_dims = [max(list(mapping.values()) + [2]) + 1 for mapping in mapping_dicts.values()]
    num_nume_feature = len(trace_time_feature + num_event_feature + num_case_feature + flag_feature)

    _check_config(cfg, dataset_name, num_act, max_case_len)
    torch.save(mean_dict, output_dir / "mean_dict.pt")
    torch.save(std_dict, output_dir / "std_dict.pt")

    # tensor creation
    df_before_split = df[df[timestamp] < train_test_split_time]
    train_test_split_idx = (df[timestamp] >= train_test_split_time).idxmax()

    splits = {
        "train": (train_case_list, df_before_split,
                  df[df[case_id].isin(train_case_list)].reset_index(drop=True), 0),
        "val": (val_case_list, df_before_split,
                df[df[case_id].isin(val_case_list)].reset_index(drop=True), 0),
        "test": (test_case_list, df, df, train_test_split_idx),
    }

    common = dict(categorical_features=categorical_features, case_id=case_id,
                  event_name=event_name, event_idx=event_idx)
    num_obs = {}

    for split, (case_list, log_df, trace_df, start_idx) in splits.items():
        log_cate, log_nume = create_log_prefix(
            log_df, log_prefix_len, case_list, start_idx, len(log_df),
            log_prefix_col_name, pad_position="left", **common)

        tensors = {"log_cate_prefix": log_cate, "log_nume_prefix": log_nume}
        for pad in ("r", "l"):
            cate, nume = create_trace_prefix(
                trace_df, trace_prefix_len, case_list, start_idx, len(trace_df),
                trace_prefix_col_name, pad_position="right" if pad == "r" else "left", **common)
            tensors[f"trace_cate_prefix_{pad}"] = cate
            tensors[f"trace_nume_prefix_{pad}"] = nume

        act_suffix, time_suffix, rm_suffix = create_trace_suffix(
            trace_df, trace_suffix_len, case_list, start_idx, len(trace_df),
            trace_suffix_col_name, pad_position="right", **common)
        tensors.update(act_suffix=act_suffix, time_suffix=time_suffix, rm_suffix=rm_suffix)

        keep_mask = lens_till_eoc(act_suffix, EOC_INDEX) >= MIN_SUFFIX_LEN
        for name, tensor in tensors.items():
            torch.save(tensor[keep_mask], tensor_dir / f"{split}_{name}.pt")
        num_obs[split] = int(keep_mask.sum())

    log_info = {
        "num_act": int(num_act),
        "input_dims": [int(d) for d in input_dims],
        "num_nume_feature": int(num_nume_feature),
        "max_case_len": int(max_case_len),
        "log_prefix_len": int(log_prefix_len),
        "trace_prefix_len": int(trace_prefix_len),
        "trace_suffix_len": int(trace_suffix_len),
        "num_obs": num_obs,
        "trace_prefix_col_name": trace_prefix_col_name,
        "log_prefix_col_name": log_prefix_col_name + ["same_case"],
    }
    with open(output_dir / "log_info.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(log_info, f, sort_keys=False)

    return log_info


def _check_config(cfg, dataset_name, num_act, max_case_len):
    if cfg.get("num_act") is not None and cfg["num_act"] != num_act:
        warnings.warn(f"[{dataset_name}] num_act in dataset_configs.yml is {cfg['num_act']}, "
                      f"but the training set gives {num_act}. Update the model configs accordingly.")
    if cfg["trace_prefix_len"] < max_case_len + 1:
        warnings.warn(f"[{dataset_name}] trace_prefix_len={cfg['trace_prefix_len']} is smaller than "
                      f"max_case_len + 1 = {max_case_len + 1}; long trace prefixes will be truncated.")
    if cfg["trace_suffix_len"] < max_case_len + 1:
        warnings.warn(f"[{dataset_name}] trace_suffix_len={cfg['trace_suffix_len']} is smaller than "
                      f"max_case_len + 1 = {max_case_len + 1}; long suffixes will be truncated.")
