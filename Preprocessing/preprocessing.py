"""Cleaning, feature engineering, encoding and standardization of event logs."""

import numpy as np
import pandas as pd

SOC_INDEX = 2
EOC_INDEX = 3


def clean_log(df, categorical_features, case_id, event_name, timestamp):
    """Remove rows with missing key values and duplicates; cast case ID and categorical columns to string."""
    df = df.dropna(subset=[case_id, event_name, timestamp])
    df = df.drop_duplicates(keep="first").copy()
    df[case_id] = df[case_id].astype("str")
    for col in categorical_features:
        df[col] = df[col].astype("str")

    return df


def sort_log(df, timestamp):
    """Parse timestamps as UTC and sort the event log chronologically."""
    df[timestamp] = pd.to_datetime(df[timestamp], format="mixed").dt.tz_convert("UTC")
    return df.sort_values(by=timestamp, kind="mergesort").reset_index(drop=True)


def debiasing(orn_df, case_id, event_name, timestamp, case_len_quantile, tpt_quantile=0.995):
    """Remove cases whose throughput time or length exceeds the given quantile."""
    df_agg = orn_df.groupby(case_id).agg(
        tpt=(timestamp, lambda x: (x.max() - x.min()).total_seconds()),
        case_len=(event_name, "count"),
    ).reset_index()
    df_agg["tpt"] = df_agg["tpt"] / (24 * 3600)
    max_tpt = df_agg["tpt"].quantile(tpt_quantile)
    max_len = int(df_agg["case_len"].quantile(case_len_quantile))

    cases_by_tpt = set(df_agg[df_agg["tpt"] <= max_tpt][case_id])
    cases_by_len = set(df_agg[df_agg["case_len"] <= max_len][case_id])
    cases_retained = cases_by_tpt & cases_by_len

    df = orn_df[orn_df[case_id].isin(cases_retained)].reset_index(drop=True)
    df = df.sort_values(by=timestamp, kind="mergesort").reset_index(drop=True)

    return df, max_len


def map_case_id(df, case_id):
    """Map case IDs to integers 1..n in order of first appearance."""
    case_id_dict = {cid: n for n, cid in enumerate(df[case_id].unique(), start=1)}
    df[case_id] = df[case_id].map(case_id_dict)

    return df, case_id_dict


def add_soc_eoc(df, case_id, event_name, timestamp):
    """
    Insert an SOC (start of case) row before the first event and an EOC (end of case)
    row after the last event of every case. Both rows carry the timestamp of the
    adjacent event; all other attributes are missing. The result is sorted
    chronologically and gets an `event_idx` column.
    """
    df = df.sort_values(by=[case_id, timestamp], kind="mergesort").reset_index(drop=True)

    # position of each row in the case-ordered log once SOC/EOC rows are inserted
    case_rank = pd.factorize(df[case_id])[0]
    event_seq = np.arange(len(df)) + 2 * case_rank + 1

    first = ~df[case_id].duplicated(keep="first").to_numpy()
    last = ~df[case_id].duplicated(keep="last").to_numpy()

    def marker_rows(mask, name, seq):
        rows = df.loc[mask, [case_id, timestamp]].reset_index(drop=True)
        rows[event_name] = name
        rows["_seq"] = seq
        return rows

    events = df.copy()
    events["_seq"] = event_seq
    soc_rows = marker_rows(first, "SOC", event_seq[first] - 1)
    eoc_rows = marker_rows(last, "EOC", event_seq[last] + 1)

    new_df = pd.concat([soc_rows, events, eoc_rows], ignore_index=True)[list(df.columns) + ["_seq"]]

    # SOC is placed directly before and EOC directly after the events with the same timestamp
    offset = np.where(new_df[event_name] == "SOC", 1, np.where(new_df[event_name] == "EOC", 3, 2))
    new_df["_order"] = new_df[case_id].astype(np.int64) * 10 + offset

    new_df = new_df.sort_values(by=[timestamp, "_order", "_seq"], kind="mergesort")
    new_df = new_df.drop(columns=["_order", "_seq"]).reset_index(drop=True)
    new_df["event_idx"] = range(1, len(new_df) + 1)

    return new_df


def create_time_features(df, case_id, timestamp, event_idx):
    """
    Create temporal features (in seconds):
    - log_ts_pre: time since the previous event in the log
    - trace_ts_pre: time since the previous event in the case
    - trace_ts_start: time since the start of the case
    - rm: remaining time until the end of the case
    """
    df = df.sort_values(by=event_idx).reset_index(drop=True)
    df["log_ts_pre"] = df[timestamp].diff(periods=1).dt.total_seconds().fillna(0.0)

    df = df.sort_values(by=[case_id, timestamp], kind="mergesort").reset_index(drop=True)
    df["trace_ts_pre"] = df.groupby(case_id)[timestamp].diff().dt.total_seconds().fillna(0.0)

    case_start = df.groupby(case_id)[timestamp].transform("min")
    case_end = df.groupby(case_id)[timestamp].transform("max")
    df["trace_ts_start"] = (df[timestamp] - case_start).dt.total_seconds().fillna(0.0)
    df["rm"] = (case_end - df[timestamp]).dt.total_seconds().fillna(0.0)

    return df.sort_values(by=event_idx).reset_index(drop=True)


def fill_missing(df, numerical_features):
    """Fill missing numerical values with zero and add a binary flag column for each affected feature."""
    flag_feature = []

    for col in numerical_features:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in the event log.")

        if df[col].isna().any():
            flag_col = f"{col}_fl"
            df[flag_col] = df[col].isna().astype(int)
            df[col] = df[col].astype(float).fillna(0)
            flag_feature.append(flag_col)

    return df, flag_feature


def standardize(df, training_df, numerical_features):
    """Standardize numerical features using the mean and standard deviation of the training set."""
    mean_dict = {col: float(training_df[col].mean()) for col in numerical_features}
    std_dict = {col: float(training_df[col].std()) for col in numerical_features}

    for col in numerical_features:
        df[col] = (df[col] - mean_dict[col]) / std_dict[col]

    return df, mean_dict, std_dict


def map_event_name(df, training_df, event_name):
    """
    Map activity labels to integers. 0 is reserved for padding, 1 for labels unseen
    in the training set, 2 for SOC and 3 for EOC.
    """
    event_name_dict = {"SOC": SOC_INDEX, "EOC": EOC_INDEX}
    n = 4
    for name in training_df[event_name].unique():
        if name not in event_name_dict:
            event_name_dict[name] = n
            n += 1

    for name in df[event_name].unique():
        if name not in event_name_dict:
            event_name_dict[name] = 1

    df[event_name] = df[event_name].map(event_name_dict)

    return df, event_name_dict


def map_cat_feature(df, training_df, cat_features):
    """
    Map categorical feature values to integers. 0 is reserved for padding, 1 for values
    unseen in the training set and 2 for missing values.
    """
    all_mappings = {}

    for col in cat_features:
        if col not in training_df.columns:
            raise KeyError(f"Column '{col}' not found in the event log.")

        mapping_dict = {}
        n = 3
        for cat in training_df[col].unique():
            if pd.notna(cat):
                mapping_dict[cat] = n
                n += 1

        for cat in df[col].unique():
            if pd.notna(cat) and cat not in mapping_dict:
                mapping_dict[cat] = 1

        df[col] = df[col].map(lambda x: 2 if pd.isna(x) else mapping_dict[x])
        all_mappings[col] = mapping_dict

    return df, all_mappings
