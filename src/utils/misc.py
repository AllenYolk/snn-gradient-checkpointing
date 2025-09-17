from typing import Dict, List
from pathlib import Path
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from torch.utils import data
import torch.distributed as dist
import torchvision.ops.boxes as box_ops


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.
    '''
    dataloader = trainloader = data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )

    mean = torch.zeros(3)  # RGB, 3 channels
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Initialize layer parameters.
    '''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


class AverageMeter:
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_one_hot(x: torch.Tensor, l: int):
    x_one_hot = torch.zeros(*x.shape, l, device=x.device)
    x_one_hot.scatter_(1, x.unsqueeze(-1), 1.0)
    return x_one_hot


def txt2list(filename):
    lines_list = []
    with open(filename, 'r') as txt:
        for line in txt:
            lines_list.append(line.rstrip('\n'))
    return lines_list


def count_learnable_parameters(model):
    cnt = 0
    for param in model.parameters():
        if param.requires_grad:
            cnt += param.numel()
    return cnt


def get_detection_matched_idxs(anchors, targets, proposal_matcher):
    matched_idxs = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        if targets_per_image["boxes"].numel() == 0:
            matched_idxs.append(
                torch.full(
                    (anchors_per_image.size(0),),
                    -1,
                    dtype=torch.int64,
                    device=anchors_per_image.device,
                )
            )
            continue
        match_quality_matrix = box_ops.box_iou(
            targets_per_image["boxes"].to(anchors_per_image.device),
            anchors_per_image
        )
        matched_idxs.append(proposal_matcher(match_quality_matrix))  # N_anchors
    return matched_idxs


def postprocess_detections(
    head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], box_coder,
    image_shape, num_classes, score_threshold, topk_candidates, nms_threshold,
    detections_per_img
) -> List[Dict[str, Tensor]]:
    bbox_regression = head_outputs["bbox_regression"]
    pred_logits = head_outputs["cls_logits"]

    detections = []

    for boxes, logits, anchors in zip(
        bbox_regression, pred_logits, image_anchors
    ):
        boxes = box_coder.decode_single(boxes, anchors)  # N_anchors 4(xyxy)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        image_boxes, image_scores, image_labels = [], [], []
        for label in range(num_classes):
            logits_per_class = logits[:, label]
            score = torch.sigmoid(logits_per_class).flatten()

            # remove low scoring boxes
            keep_idxs = score > score_threshold
            score = score[keep_idxs]
            box = boxes[keep_idxs]

            # keep only topk scoring predictions
            num_topk = min(topk_candidates, score.size(0))
            score, idxs = score.topk(num_topk)
            box = box[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(
                torch.full_like(score, fill_value=label, dtype=torch.int64)
            )

        image_boxes = torch.cat(image_boxes, dim=0)  # N_anchors(match) 4
        image_scores = torch.cat(image_scores, dim=0)  # N_anchors(match),
        image_labels = torch.cat(image_labels, dim=0)  # # N_anchors(match),

        # non-maximum suppression
        keep = box_ops.batched_nms(
            image_boxes, image_scores, image_labels, nms_threshold
        )
        keep = keep[:detections_per_img]

        detections.append({
            "boxes": image_boxes[keep],  # N_keep 4
            "scores": image_scores[keep],  # N_keep
            "labels": image_labels[keep],  # N_keep
        })
    return detections


def filter_boxes(tensors, min_box_diag=30, min_box_side=20):
    widths = tensors['boxes'][:, 2] - tensors['boxes'][:, 0]  # get all widths
    heights = tensors['boxes'][:, 3] - tensors['boxes'][:, 1]  # get all heights
    diag_square = widths**2 + heights**2
    mask = (diag_square >= min_box_diag**
            2) * (widths >= min_box_side) * (heights >= min_box_side)
    return {k: v[mask] for k, v in tensors.items()}


def move_state_dict_to_cpu(path):
    """Move the state dict saved in the specified path to CPU.
    """
    path = Path(path)
    sd = torch.load(path)
    if not isinstance(sd, dict):
        raise TypeError(
            "`path` should point to a file containing a model state dict"
        )

    cpu_sd = {k: v.cpu() for k, v in sd.items()}
    torch.save(cpu_sd, path)


def tensor_size(x: torch.Tensor):  # in bytes
    return x.element_size() * x.numel()


def resolve_device() -> str:
    """Resolve the logical device for the current process.

    Priority:
      1) If no cuda available -> cpu
      2) LOCAL_RANK / SLURM_LOCALID / OMPI_COMM_WORLD_LOCAL_RANK env 
      3) If torch.distributed initialized -> use rank % ngpus
      4) torch.cuda.current_device()
      5) fallback to cuda
    """
    if not torch.cuda.is_available():
        return "cpu"

    # common env vars
    for k in (
        "LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK"
    ):
        v = os.environ.get(k)
        if v is not None:
            try:
                return f"cuda:{int(v)}"
            except Exception:
                pass

    # if dist inited, use rank % n_gpus
    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            n_gpu = torch.cuda.device_count()
            if n_gpu > 0:
                return f"cuda:{rank % n_gpu}"
    except Exception:
        pass

    # fallback to current_device (logical ID after CUDA_VISIBLE_DEVICES)
    try:
        return f"cuda:{torch.cuda.current_device()}"
    except Exception:
        return "cuda"
