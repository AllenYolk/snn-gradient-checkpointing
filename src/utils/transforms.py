"""Transformations shared by multiple datasets.
"""

import random
import numbers
import math
from typing import Tuple

import numpy as np
from PIL import ImageOps, Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from .misc import get_one_hot


class TransformedDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        data = torch.FloatTensor(data)
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.dataset)


################################################################################
#                  Mixup and Cutmix for classification tasks                   #
################################################################################


class RandomMixup(nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False
    ) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4 and batch.ndim != 5:
            raise ValueError(f"Batch ndim should be 4 or 5. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(
                f"Batch dtype should be a float tensor. Got {batch.dtype}."
            )
        if target.dtype != torch.int64:
            raise TypeError(
                f"Target dtype should be torch.int64. Got {target.dtype}"
            )

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = get_one_hot(target, self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3.
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


class RandomCutmix(nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False
    ) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch: torch.Tensor,
                target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )

        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4 and batch.ndim != 5:
            raise ValueError(f"Batch ndim should be 4 or 5. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(
                f"Batch dtype should be a float tensor. Got {batch.dtype}."
            )
        if target.dtype != torch.int64:
            raise TypeError(
                f"Target dtype should be torch.int64. Got {target.dtype}"
            )

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = get_one_hot(target, self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(
            torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0]
        )
        W, H = torchvision.transforms.functional.get_image_size(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2-x1) * (y2-y1) / (W*H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s


################################################################################
#                    Cutout for static image classification                    #
################################################################################


class Cutout:
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        max_length (int): If not None, randomly sample the length of the square 
            patch. If None, use the argument `length` instead.
    """

    def __init__(self, n_holes, length=None, max_length=None):
        self.n_holes = n_holes
        self.length = length
        self.max_length = max_length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(-2)
        w = img.size(-1)

        mask = np.ones((h, w), np.float32)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            length = self.length
            if self.max_length is not None:
                length = np.random.randint(1, self.max_length)

            y1 = np.clip(y - length//2, 0, h)
            y2 = np.clip(y + length//2, 0, h)
            x1 = np.clip(x - length//2, 0, w)
            x2 = np.clip(x + length//2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


################################################################################
#         Sequential data augmentation for neuromorphic datasets.              #
#         Apply the transform to each frame in the sequence.                   #
################################################################################


class ToTensor:

    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class ToPILImage:

    def __call__(self, imgmap):
        topilimage = transforms.ToPILImage()
        return [topilimage(i) for i in imgmap]


class Padding:

    def __init__(self, pad):
        self.pad = pad

    def __call__(self, imgmap):
        return [ImageOps.expand(img, border=self.pad, fill=0) for img in imgmap]


class Resize:

    def __init__(self, size):
        self.size = size

    def __call__(self, imgmap):
        resize = transforms.Resize(self.size)
        return [resize(i) for i in imgmap]


class Normalize:

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, imgmap):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        return [normalize(i) for i in imgmap]


class RandomHorizontalFlip:

    def __init__(self, consistent=True, command=None):
        self.consistent = consistent
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5

    def __call__(self, imgmap):
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for i in imgmap:
                if random.random() < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class RandomCrop:

    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap, flowmap=None):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if not flowmap:
                if self.consistent:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                    return result
            elif flowmap is not None:
                assert (not self.consistent)
                result = []
                for idx, i in enumerate(imgmap):
                    proposal = []
                    for j in range(
                        3
                    ):  # number of proposal: use the one with largest optical flow
                        x = random.randint(0, w - tw)
                        y = random.randint(0, h - th)
                        proposal.append([
                            x, y,
                            abs(np.mean(flowmap[idx, y:y + th, x:x + tw, :]))
                        ])
                    [x1, y1, _] = max(proposal, key=lambda x: x[-1])
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
            else:
                raise ValueError('wrong case')
        else:
            return imgmap


class RandomCropWithProb:

    def __init__(self, size, p=0.8, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent
        self.threshold = p

    def __call__(self, imgmap):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if self.consistent:
                if random.random() < self.threshold:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                else:
                    x1 = int(round((w-tw) / 2.))
                    y1 = int(round((h-th) / 2.))
                return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
            else:
                result = []
                for i in imgmap:
                    if random.random() < self.threshold:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    else:
                        x1 = int(round((w-tw) / 2.))
                        y1 = int(round((h-th) / 2.))
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
        else:
            return imgmap


class TimeChannelMask:

    def __init__(self, prob, time_mask_size, channel_mask_size):
        self.prob = prob
        self.time_mask_size = time_mask_size
        self.channel_mask_size = channel_mask_size

    def __call__(self, x):
        # Sample shape: (B, T, C, ...)
        # Time mask
        if np.random.uniform() < self.prob:
            mask_size = np.random.randint(0, self.time_mask_size)
            idx = np.random.randint(0, x.shape[1] - mask_size)
            x[:, idx:idx + mask_size, :] = 0.

        # Neuron mask
        if np.random.uniform() < self.prob:
            mask_size = np.random.randint(0, self.channel_mask_size)
            idx = np.random.randint(0, x.shape[2] - mask_size)
            x[:, :, idx:idx + mask_size] = 0.

        return x
