"""Create prefix and suffix tensors from a preprocessed event log."""

import numpy as np
import torch

from .preprocessing import SOC_INDEX, EOC_INDEX

CATE_PAD = 0
NUME_PAD = -10000.0

def _fill_padded(out, spans_values, seq_len, pad_position, keep="last"):
    """Write variable-length sequences into a preallocated padded array."""
    for r, seq in enumerate(spans_values):
        L = len(seq)
        if L == 0:
            continue
        if L >= seq_len:
            out[r, :] = seq[-seq_len:] if keep == "last" else seq[:seq_len]
        elif pad_position == "right":
            out[r, :L] = seq
        elif pad_position == "left":
            out[r, -L:] = seq
    return out

def _stack(tensors_list):
    return torch.stack(tensors_list, dim=-1) if tensors_list else torch.empty(0)

def _case_indices(df, case_id, case_list):
    groups = df.groupby(case_id, sort=False).indices
    return {c: np.asarray(groups[c]) for c in case_list if c in groups}

def create_log_prefix(df, log_prefix_len, case_list, start_idx, end_idx, log_prefix_col_name,
                      categorical_features, case_id, event_name, event_idx, pad_position):
                        
    df = df.sort_values(by=event_idx).reset_index(drop=True)

    case_values = df[case_id].to_numpy()
    event_values = df[event_name].to_numpy()
    case_list_set = set(case_list)

    spans, span_case_ids = [], []
    for i in range(start_idx, end_idx):
        if case_values[i] in case_list_set and event_values[i] not in (SOC_INDEX, EOC_INDEX):
            spans.append((max(0, i - log_prefix_len + 1), i))
            span_case_ids.append(case_values[i])

    if not spans:
        raise ValueError("No valid log prefix spans were found.")

    cate_tensors_list, nume_tensors_list = [], []
    for col in log_prefix_col_name:
        is_cate = col in categorical_features
        out = np.full((len(spans), log_prefix_len),
                      CATE_PAD if is_cate else NUME_PAD,
                      dtype=np.int64 if is_cate else np.float32)
        col_values = df[col].to_numpy()
        _fill_padded(out, (col_values[s:e + 1] for s, e in spans), log_prefix_len, pad_position)

        if is_cate:
            cate_tensors_list.append(torch.as_tensor(out).to(torch.int16))
        else:
            nume_tensors_list.append(torch.as_tensor(out).to(torch.float32))

    same_case = np.full((len(spans), log_prefix_len), NUME_PAD, dtype=np.float32)
    _fill_padded(same_case,
                 ((case_values[s:e + 1] == c).astype(np.float32) for (s, e), c in zip(spans, span_case_ids)),
                 log_prefix_len, pad_position)
    nume_tensors_list.append(torch.as_tensor(same_case, dtype=torch.float32))

    return _stack(cate_tensors_list), _stack(nume_tensors_list)

def create_trace_prefix(df, trace_prefix_len, case_list, start_idx, end_idx, trace_prefix_col_name,
                        categorical_features, case_id, event_name, event_idx, pad_position):

    df = df.sort_values(by=event_idx).reset_index(drop=True)
    case_to_indices = _case_indices(df, case_id, case_list)

    pos_in_case = -np.ones(len(df), dtype=np.int64)
    case_of_row = np.full(len(df), None, dtype=object)
    for c, idxs in case_to_indices.items():
        pos_in_case[idxs] = np.arange(len(idxs))
        case_of_row[idxs] = c

    event_values = df[event_name].to_numpy()
    spans = []
    for i in range(start_idx, end_idx):
        c = case_of_row[i]
        if c is None or event_values[i] in (SOC_INDEX, EOC_INDEX):
            continue
        pos = pos_in_case[i]
        spans.append((c, max(0, pos - trace_prefix_len + 1), pos))

    if not spans:
        raise ValueError("No valid trace prefix spans were found.")

    cate_tensors_list, nume_tensors_list = [], []
    for col in trace_prefix_col_name:
        is_cate = col in categorical_features
        out = np.full((len(spans), trace_prefix_len),
                      CATE_PAD if is_cate else NUME_PAD,
                      dtype=np.int16 if is_cate else np.float32)
        col_values = df[col].to_numpy()
        case_to_colvals = {c: col_values[idxs] for c, idxs in case_to_indices.items()}
        _fill_padded(out, (case_to_colvals[c][s:e + 1] for c, s, e in spans), trace_prefix_len, pad_position)

        if is_cate:
            cate_tensors_list.append(torch.as_tensor(out).to(torch.int16))
        else:
            nume_tensors_list.append(torch.as_tensor(out).to(torch.float32))

    return _stack(cate_tensors_list), _stack(nume_tensors_list)

def create_trace_suffix(df, trace_suffix_len, case_list, start_idx, end_idx, suffix_col_name,
                        categorical_features, case_id, event_name, event_idx, pad_position="right"):

    df = df.sort_values(by=event_idx).reset_index(drop=True)
    case_to_indices = _case_indices(df, case_id, case_list)

    pos_in_case = -np.ones(len(df), dtype=np.int64)
    case_of_row = np.full(len(df), None, dtype=object)
    for c, idxs in case_to_indices.items():
        pos_in_case[idxs] = np.arange(len(idxs))
        case_of_row[idxs] = c

    event_values = df[event_name].to_numpy()
    spans = []
    for i in range(start_idx, end_idx):
        c = case_of_row[i]
        if c is None or event_values[i] in (SOC_INDEX, EOC_INDEX):
            continue
        spans.append((c, pos_in_case[i], len(case_to_indices[c]) - 1))

    if not spans:
        raise ValueError("No valid suffix spans were found.")

    suffix_tensors_list = []
    for col in suffix_col_name:
        is_cate = col in categorical_features
        out = np.full((len(spans), trace_suffix_len),
                      CATE_PAD if is_cate else NUME_PAD,
                      dtype=np.int16 if is_cate else np.float32)
        col_values = df[col].to_numpy()
        case_to_colvals = {c: col_values[idxs] for c, idxs in case_to_indices.items()}
        _fill_padded(out, (case_to_colvals[c][s:e + 1] for c, s, e in spans), trace_suffix_len,
                     pad_position, keep="first")

        dtype = torch.int16 if is_cate else torch.float32
        suffix_tensors_list.append(torch.as_tensor(out).to(dtype))

    return suffix_tensors_list

def lens_till_eoc(x, eoc_index=EOC_INDEX):
    T = x.shape[1]
    is_eoc = x == eoc_index
    first_eoc = is_eoc.int().argmax(dim=1)
    return torch.where(is_eoc.any(dim=1), first_eoc + 1, torch.full_like(first_eoc, T))
