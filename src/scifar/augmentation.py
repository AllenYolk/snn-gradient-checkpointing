import torch
from torchvision import transforms
from torchvision.transforms import autoaugment
from torchvision.transforms.functional import InterpolationMode

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
                trans.append(
                    autoaugment.RandAugment(interpolation=interpolation)
                )
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
        trans.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
