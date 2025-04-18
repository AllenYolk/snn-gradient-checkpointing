from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import focal_loss


class TMeanCrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_seq, label):
        y = y_seq.mean(dim=0)
        return super().forward(y, label)


class TETLoss(nn.Module):

    def __init__(
        self,
        base_criterion,
        mean: float,
        tet_lambda: float,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.mean = mean
        self.tet_lambda = tet_lambda

        if tet_lambda == 0:
            self.regularization_loss = self._regularization_loss_0
        else:
            self.regularization_loss = self._regularization_loss

    def base_criterion_loss(self, y, label):
        T = y.shape[0]
        l = 0
        for t in range(T):
            l += self.base_criterion(y[t], label)
        return l / T

    def _regularization_loss(self, y):
        reg = torch.full_like(y, self.mean)
        return F.mse_loss(y, reg)

    def _regularization_loss_0(self, y):
        return 0.

    def forward(self, y, label):
        base_loss = self.base_criterion_loss(y, label)
        re_loss = self.regularization_loss(y)
        return (1. - self.tet_lambda) * base_loss + self.tet_lambda * re_loss


def mse_loss_allowing_nan(y_pred, y_true):
    if isinstance(y_true, np.ndarray):
        loss = (y_true - y_pred)**2
        return np.nanmean(loss)
    mask = ~torch.isnan(y_true)
    y_pred = torch.masked_select(y_pred, mask)
    y_true = torch.masked_select(y_true, mask)
    loss = (y_true - y_pred)**2
    loss = loss.mean()
    return loss


def detection_loss(
    targets: List[Dict[str, Tensor]], head_outputs: Dict[str, Tensor],
    anchors: List[Tensor], matched_idxs: List[Tensor], box_coder: nn.Module
) -> Dict[str, Tensor]:
    bbox_regression = head_outputs["bbox_regression"]
    cls_logits = head_outputs["cls_logits"]  # Size=(N, A, K)

    num_foreground_reg = 0
    num_foreground_cls = 0
    bbox_loss, cls_loss = [], []

    # Match original targets with default boxes
    for (
        targets_per_image, bbox_regression_per_image, cls_logits_per_image,
        anchors_per_image, matched_idxs_per_image
    ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
        # produce the matching between boxes and targets
        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0
                                               )[0]  # N_matched(idx_anchor)
        foreground_matched_idxs_per_image = matched_idxs_per_image[
            foreground_idxs_per_image].to("cpu")  # N_matched(idx_gt_box)
        num_foreground_reg += foreground_idxs_per_image.numel()

        # Compute regression loss
        matched_gt_boxes_per_image = targets_per_image["boxes"][
            foreground_matched_idxs_per_image]  # N_matched
        bbox_regression_per_image = bbox_regression_per_image[
            foreground_idxs_per_image, :]  # N_matched
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :
                                             ]  # N_matched
        target_regression = box_coder.encode_single(
            matched_gt_boxes_per_image.to(anchors_per_image.device),
            anchors_per_image
        )  # N_matched

        bbox_loss.append(
            F.smooth_l1_loss(
                bbox_regression_per_image, target_regression, reduction="sum"
            )
        )

        # Compute classification loss (focal loss)
        foreground_idxs_per_image = matched_idxs_per_image >= 0  # N_matched
        num_foreground_cls += foreground_idxs_per_image.sum()
        gt_classes_target = torch.zeros_like(cls_logits_per_image)  # A, K

        gt_classes_target[
            foreground_idxs_per_image,
            targets_per_image["labels"][foreground_matched_idxs_per_image],
        ] = 1.0

        cls_loss.append(
            focal_loss.sigmoid_focal_loss(
                cls_logits_per_image,
                gt_classes_target,
                reduction="sum",
            )
        )

    bbox_loss = torch.stack(bbox_loss)
    cls_loss = torch.stack(cls_loss)

    return {
        "bbox_regression": bbox_loss.sum() / max(1, num_foreground_reg),
        "classification": cls_loss.sum() / max(1, num_foreground_cls),
    }
