import torch
import numpy as np
from sklearn.metrics import r2_score as sklearn_r2_score


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def rrse_score(y_pred, y_true) -> float:
    """Root Relative Squared Error (RRSE).

    Notice that this metric must be computed over the whole dataset. Batched 
    computation leads to incorrect results.

    Args:
        y_pred: the predicted values over the whole dataset.
        y_true: the true values over the whole dataset.
    """
    if isinstance(y_true, np.ndarray):
        y_bar = y_true.mean(axis=0)
        loss = np.sqrt(((y_pred - y_true)**2).sum()) / np.sqrt(
            ((y_true - y_bar)**2).sum()
        )
        return np.nanmean(loss)
    y_bar = y_true.mean(dim=0)
    loss = torch.sqrt(((y_pred - y_true)**2).sum()) / torch.sqrt(
        ((y_true - y_bar)**2).sum()
    )
    return loss.mean().item()


def r2_score(y_pred, y_true) -> float:
    """R2 score.

    Notice that this metric must be computed over the whole dataset. Batched 
    computation leads to incorrect results.

    Args:
        y_pred: the predicted values over the whole dataset.
        y_true: the true values over the whole dataset.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(len(y_true), -1)
        if y_pred.shape[-1] == 1:
            y_pred = y_pred.squeeze(-1)
        return sklearn_r2_score(y_true, y_pred)
    if isinstance(y_true, list) and isinstance(y_pred, list):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return r2_score(y_true, y_pred)
    raise NotImplementedError(
        f"unsupported data type {type(y_pred)} or {type(y_true)}"
    )
