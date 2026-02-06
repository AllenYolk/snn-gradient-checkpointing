import lightning as L
import torch
from torch.utils.data.dataloader import default_collate, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import autoaugment
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import CIFAR10, CIFAR100
from utils.transforms import RandomMixup, RandomCutmix

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class SequentialCIFARClassificationPresetTrain:
    def __init__(
        self,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = []
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment(interpolation=interpolation))
            elif auto_augment_policy == "ta_wide":
                trans.append(
                    autoaugment.TrivialAugmentWide(interpolation=interpolation)
                )
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(
                    autoaugment.AutoAugment(
                        policy=aa_policy, interpolation=interpolation
                    )
                )
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class SCIFARDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_classes: int = 10,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        if num_classes == 10:
            self.mu = CIFAR10_MEAN
            self.sigma = CIFAR10_STD
            self.ds_class = CIFAR10
        else:
            self.mu = CIFAR100_MEAN
            self.sigma = CIFAR100_STD
            self.ds_class = CIFAR100

    def prepare_data(self):
        self.ds_class(self.data_dir, train=True, download=True)
        self.ds_class(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        mixup_transforms = []
        mixup_transforms.append(RandomMixup(self.num_classes, p=1.0, alpha=0.2))
        mixup_transforms.append(RandomCutmix(self.num_classes, p=1.0, alpha=1.0))
        mixupcutmix = transforms.RandomChoice(mixup_transforms)
        self.collate_fn = lambda batch: mixupcutmix(*default_collate(batch))

        transform_train = SequentialCIFARClassificationPresetTrain(
            mean=self.mu,
            std=self.sigma,
            interpolation=InterpolationMode("bilinear"),
            auto_augment_policy="ta_wide",
            random_erase_prob=0.1,
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mu, self.sigma),
            ]
        )

        self.train_set = self.ds_class(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )
        self.test_set = self.ds_class(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
