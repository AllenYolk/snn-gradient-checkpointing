import argparse
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append("./src")

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms.functional import InterpolationMode

from utils import set_seed, ModelNameGenerator
from utils import count_learnable_parameters
from utils import LayerWiseFPCUDATimeProfiler
from augmentation import SequentialCIFARClassificationPresetTrain
from augmentation import CIFAR100_MEAN, CIFAR100_STD
from augmentation import CIFAR10_MEAN, CIFAR10_STD
from utils.transforms import RandomMixup, RandomCutmix
import models as models


def prepare_dataloaders(args):
    mixup_transforms = []
    mixup_transforms.append(RandomMixup(args.num_classes, p=1.0, alpha=0.2))
    mixup_transforms.append(RandomCutmix(args.num_classes, p=1.0, alpha=1.))
    mixupcutmix = transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))

    if args.num_classes == 10:
        ds_class = datasets.CIFAR10
        mu = CIFAR10_MEAN
        sigma = CIFAR10_STD
    else:
        ds_class = datasets.CIFAR100
        mu = CIFAR100_MEAN
        sigma = CIFAR100_STD

    transform_train = SequentialCIFARClassificationPresetTrain(
        mean=mu,
        std=sigma,
        interpolation=InterpolationMode('bilinear'),
        auto_augment_policy='ta_wide',
        random_erase_prob=0.1
    )

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mu, sigma),
    ])

    train_set = ds_class(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    test_set = ds_class(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    train_data_loader = data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_data_loader, test_data_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Classify Sequential CIFAR')
    parser.add_argument('-neuron', "--neuron_type", default='LIF', type=str)
    parser.add_argument(
        "-sc",
        "--spike_compressor",
        default="IdentitySpikeCompressor",
        type=str
    )
    parser.add_argument(
        '-net', "--network", default="MESequentialCIFARNet", type=str
    )
    parser.add_argument(
        '--data_dir', type=str, default="/home/ma-user/work/datasets/CIFAR10"
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("-nc", "--num_classes", default=10, type=int)
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)

    args = parser.parse_args()
    args.decay_lambda = 0.5
    args.channels = 128
    args.batch_size = 128
    args.num_workers = 4
    args.lomo = False
    args.amp = False
    args.warmup = 10
    return args


def main():
    args = parse_args()
    print(args)

    set_seed(args.set_seed)

    run_name = ModelNameGenerator(
        proj=f"sequential-cifar{args.num_classes}-me"
    ).generate(args)
    log_path = Path(args.log_dir) / f"SCIFAR{args.num_classes}"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    log_path = log_path / (run_name+".time-prof.txt")

    _, val_data_loader = prepare_dataloaders(args)

    net = getattr(models, args.network)(
        channels=args.channels,
        neuron_type=args.neuron_type,
        spike_compressor=args.spike_compressor,
        num_classes=args.num_classes,
        decay_lambda=args.decay_lambda,
        T=32,  # for PSN
        k=8,  # for SlidingPSN
    )
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))
    net = net.to(args.device)

    profiler = LayerWiseFPCUDATimeProfiler(
        (net.conv, net.fc, net.decode),
        search_mode=("direct_children", "self", "self"),
        model_names=("conv", "fc", "decode"),
        instances=(torch.nn.Module,),
        filename=log_path,
        warmup=args.warmup,
    )

    with torch.no_grad():
        for img, label in tqdm(val_data_loader):
            img, label = img.float().to(args.device), label.to(args.device)
            img = img.permute(3, 0, 1, 2)  # [W, N, C, H]; W acts as T
            y = net(img)

    profiler.clear_hooks()
    profiler.profile()


if __name__ == '__main__':
    main()
