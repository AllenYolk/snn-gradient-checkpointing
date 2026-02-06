import PIL
import sys

sys.path.append("./src")

import lightning as L
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from spikingjelly.datasets import CIFAR10DVSTEBNSplit

from utils.transforms import Cutout


class CIFAR10DVSNDA:
    def __init__(self, M=1, N=2):
        self.M = M
        self.N = N

    def __call__(self, data):
        c = 15 * self.N
        rotate_tf = transforms.RandomRotation(degrees=c)
        e = 8 * self.N
        cutout_tf = Cutout(n_holes=1, length=e)

        def roll(data, N=1):
            a = N * 2 + 1
            off1 = np.random.randint(-a, a + 1)
            off2 = np.random.randint(-a, a + 1)
            return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        def rotate(data, N):
            return rotate_tf(data)

        def cutout(data, N):
            return cutout_tf(data)

        transforms_list = [roll, rotate, cutout]
        sampled_ops = np.random.choice(transforms_list, self.M)
        for op in sampled_ops:
            data = op(data, self.N)
        return data


class CIFAR10DVSDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str, T: int, batch_size: int = 128, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        CIFAR10DVSTEBNSplit(self.data_dir, train=True, data_type="frame", frames_number=self.T, split_by="number")
        CIFAR10DVSTEBNSplit(self.data_dir, train=False, data_type="frame", frames_number=self.T, split_by="number")

    def setup(self, stage: str):
        self.train_set = CIFAR10DVSTEBNSplit(
            self.data_dir,
            train=True,
            data_type="frame",
            frames_number=self.T,
            split_by="number",
            transform=transforms.Compose([
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                transforms.RandomResizedCrop(
                    128, scale=(0.7, 1.0), interpolation=PIL.Image.NEAREST
                ),
                transforms.Resize(size=(48, 48)),
                transforms.RandomHorizontalFlip(p=0.5),
                CIFAR10DVSNDA(M=1, N=2),
            ])
        )
        self.test_set = CIFAR10DVSTEBNSplit(
            self.data_dir,
            train=False,
            data_type="frame",
            frames_number=self.T,
            split_by="number",
            transform=transforms.Compose([
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                transforms.Resize(size=(48, 48)),
            ])
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
