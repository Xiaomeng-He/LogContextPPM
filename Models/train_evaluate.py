"""Training and evaluation shared by the LSTM and Transformer models."""

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance as norm_dl_distance
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import lens_till_eoc, EOC_INDEX

TASKS = ("suffix", "rm")

def _to_device(batch, device):
    log_cate, log_nume, trace_cate, trace_nume, dec_input_act, tgt_act, tgt_rm = batch
    return (log_cate.long().to(device), log_nume.float().to(device),
            trace_cate.long().to(device), trace_nume.float().to(device),
            dec_input_act.long().to(device), tgt_act.long().to(device), tgt_rm.float().to(device))

def loss_function_suffix(act_logits, tgt_act):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    return criterion(act_logits.view(-1, act_logits.size(-1)), tgt_act.view(-1))

def loss_function_rm(rm_prediction, tgt_rm):
    return nn.L1Loss()(rm_prediction, tgt_rm)

def performance_metrics_suffix(pred_list, pred_len, tgt_act, eoc_index=EOC_INDEX):
    tgt_act = tgt_act.detach().cpu()
    pred_len = pred_len.detach().cpu()
    tgt_len = lens_till_eoc(tgt_act, eoc_index)

    total_dl_distance = sum(norm_dl_distance(pred_list[i], tgt_act[i, :int(tgt_len[i])].tolist())
                            for i in range(tgt_act.shape[0]))
    dl_distance = total_dl_distance / tgt_act.shape[0]
    mae_len = torch.abs(pred_len - tgt_len).float().mean().item()

    return dl_distance, mae_len

def performance_metrics_rm(rm_prediction, tgt_rm, rm_mean_value, rm_std_value):
    tgt_rm = tgt_rm * rm_std_value + rm_mean_value
    rm_prediction = rm_prediction * rm_std_value + rm_mean_value

    mae_rm = nn.L1Loss()(rm_prediction, tgt_rm).item()
    mse_rm = nn.MSELoss()(rm_prediction, tgt_rm).item()

    return mae_rm, mse_rm

def train(model, dataloader, optimizer, device, task="suffix"):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        log_cate, log_nume, trace_cate, trace_nume, dec_input_act, tgt_act, tgt_rm = _to_device(batch, device)
        optimizer.zero_grad()

        if task == "suffix":
            act_logits = model(log_cate, log_nume, trace_cate, trace_nume, dec_input_act)
            loss = loss_function_suffix(act_logits, tgt_act)
        else:
            rm_prediction = model(log_cate, log_nume, trace_cate, trace_nume)
            loss = loss_function_rm(rm_prediction, tgt_rm)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def validate(model, dataloader, device, task="suffix", rm_mean_value=None, rm_std_value=None):
    model.eval()
    epoch_loss, epoch_metric_1, epoch_metric_2 = 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            log_cate, log_nume, trace_cate, trace_nume, dec_input_act, tgt_act, tgt_rm = _to_device(batch, device)

            if task == "suffix":
                act_logits, act_pred_list, act_pred_len = model.argmax(
                    log_cate, log_nume, trace_cate, trace_nume, dec_input_act)
                loss = loss_function_suffix(act_logits, tgt_act)
                metric_1, metric_2 = performance_metrics_suffix(act_pred_list, act_pred_len, tgt_act)
            else:
                rm_prediction = model(log_cate, log_nume, trace_cate, trace_nume)
                loss = loss_function_rm(rm_prediction, tgt_rm)
                metric_1, metric_2 = performance_metrics_rm(rm_prediction, tgt_rm, rm_mean_value, rm_std_value)

            epoch_loss += loss.item()
            epoch_metric_1 += metric_1
            epoch_metric_2 += metric_2

    n = len(dataloader)
    return epoch_loss / n, epoch_metric_1 / n, epoch_metric_2 / n

def evaluate(model, dataloader, device, task="suffix", rm_mean_value=None, rm_std_value=None):
    _, metric_1, metric_2 = validate(model, dataloader, device, task, rm_mean_value, rm_std_value)
    return metric_1, metric_2

def training_trial(model, task, lr, num_epochs, train_dataloader, val_dataloader, device,
                   model_state_path, rm_mean_value=None, rm_std_value=None,
                   patience=20, lr_patience=15, delta=1e-3):

    metric_names = ("val_dl_distance", "val_mae_len") if task == "suffix" else ("val_mae_rm", "val_mse_rm")

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=lr_patience,
                                  threshold=delta, threshold_mode="abs", min_lr=1e-5)
    early_stopper = EarlyStopper(patience=patience)
    best_score = float("inf")
    results = []

    for epoch in tqdm.tqdm(range(num_epochs), desc=f"Training {task} model"):
        train_loss = train(model, train_dataloader, optimizer, device, task)
        val_loss, metric_1, metric_2 = validate(model, val_dataloader, device, task,
                                                rm_mean_value, rm_std_value)
        score = metric_1 if task == "suffix" else val_loss

        if score < best_score - delta:
            best_score = score
            torch.save(model.state_dict(), model_state_path)

        results.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        metric_names[0]: metric_1, metric_names[1]: metric_2,
                        "lr": optimizer.param_groups[0]["lr"]})

        scheduler.step(score)
        if early_stopper.early_stop(score, delta):
            break

    return results

class EarlyStopper:
    """Stop when the validation error has not improved by more than `delta` for `patience` epochs."""

    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.min_error = float("inf")

    def early_stop(self, val_error, delta):
        if val_error < self.min_error - delta:
            self.min_error = val_error
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
