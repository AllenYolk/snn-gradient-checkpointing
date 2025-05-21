from pathlib import Path
import PIL
import time
import sys

sys.path.append("./src")

import hashlib
import lightning as L
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from timm.data import create_transform

from utils import save_on_master, mkdir


class ImageNetDataModule(L.LightningDataModule):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        data_dir: str,
        cache_dataset: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        for_model: str = "transformer",
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.cache_dataset = cache_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.for_model = for_model

    @staticmethod
    def _get_cache_path(filepath):
        h = hashlib.sha1(str(filepath).encode()).hexdigest()
        cache_path = Path("~/.torch/vision/datasets/imagefolder")
        cache_path = cache_path / (h[:10] + ".pt")
        cache_path = cache_path.expanduser()
        return cache_path

    @staticmethod
    def _build_transform(is_train, for_model):
        normalize = transforms.Normalize(
            ImageNetDataModule.mean, ImageNetDataModule.std
        )
        if for_model == "transformer":
            if is_train:
                return create_transform(
                    input_size=224,
                    is_training=True,
                    auto_augment="rand-m9-mstd0.5-inc1",
                    interpolation='bicubic',
                    re_prob=0.25,
                    re_mode="pixel",
                    re_count=1,
                    mean=ImageNetDataModule.mean,
                    std=ImageNetDataModule.std,
                )
            else:
                t = []
                t.append(
                    transforms.Resize(256, interpolation=PIL.Image.BICUBIC)
                )
                t.append(transforms.CenterCrop(224))
                t.append(transforms.ToTensor())
                t.append(normalize)
                return transforms.Compose(t)
        elif for_model == "sew":
            if is_train:
                return transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            raise ValueError(
                f"`for_model` must be either 'transformer' or 'sew'"
            )

    @staticmethod
    def _load_dataset(dir, cache_dataset, is_train, for_model):
        st = time.time()
        cache_path = ImageNetDataModule._get_cache_path(dir)
        if cache_dataset and cache_path.exists():
            # the transforms are also cached
            print(f"Loading dataset from {cache_path}")
            ds, _ = torch.load(cache_path)
        else:
            ds = datasets.ImageFolder(
                dir,
                transform=ImageNetDataModule._build_transform(
                    is_train, for_model
                )
            )
            if cache_dataset:
                print(f"Saving dataset_train to {cache_path}")
                mkdir(cache_path.parent)
                save_on_master((ds, dir), cache_path)
        print("Took", time.time() - st)
        return ds

    def prepare_data(self):
        if not (self.train_dir.exists() and self.val_dir.exists()):
            raise FileNotFoundError(
                f"{self.train_dir} and {self.val_dir} not found. "
                f"Please download ImageNet dataset and "
                f"place it in {self.data_dir}"
            )

    def setup(self, stage: str):
        print("Loading training data")
        self.dataset_train = self._load_dataset(
            self.train_dir, self.cache_dataset, True, self.for_model
        )
        print("Loading validation data")
        self.dataset_val = self._load_dataset(
            self.val_dir, self.cache_dataset, False, self.for_model
        )
        print(
            f"dataset_train:{len(self.dataset_train)}, "
            f"dataset_val:{len(self.dataset_val)}"
        )

    def train_dataloader(self):
        train_sampler = torch.utils.data.RandomSampler(self.dataset_train)
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=(self.for_model == "transformer")
        )

    def val_dataloader(self):
        val_sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
