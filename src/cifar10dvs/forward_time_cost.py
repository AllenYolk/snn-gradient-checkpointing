import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import PIL

sys.path.append("./src")

import numpy as np
import torch
import torch.utils.data as data
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

import torchvision.transforms as transforms
from spikingjelly.activation_based import functional
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as SJCIFAR10DVS

from utils import set_seed, ModelNameGenerator
from utils import count_learnable_parameters
from utils.transforms import TransformedDatasetWrapper
from utils.profiler import *
from augmentation import CIFAR10DVSNDA
from cifar10dvs_dataset import CIFAR10DVS, move_data
import models


def prepare_dataloaders(args):
    """Borrowed from OSR and PSN.
    """
    frame_root = Path(args.data_dir) / f"frames_number_{args.T}_split_by_number"
    if not frame_root.exists():
        # download and integrate frames
        ds = SJCIFAR10DVS(
            args.data_dir,
            data_type="frame",
            frames_number=args.T,
            split_by="number"
        )
        del ds

    train_set_root = frame_root / "train"
    test_set_root = frame_root / "test"
    if not train_set_root.exists() or not test_set_root.exists():
        # split the dataset
        move_data(
            Path(args.data_dir) / f"frames_number_{args.T}_split_by_number"
        )

    train_set = CIFAR10DVS(
        args.data_dir,
        train=True,
        data_type='frame',
        frames_number=args.T,
        split_by='number'
    )
    test_set = CIFAR10DVS(
        args.data_dir,
        train=False,
        data_type='frame',
        frames_number=args.T,
        split_by='number'
    )

    function_nda = CIFAR10DVSNDA(M=1, N=2)

    def transform_train(data):
        data = transforms.RandomResizedCrop(
            128, scale=(0.7, 1.0), interpolation=PIL.Image.NEAREST
        )(data)
        resize = transforms.Resize(size=(48, 48))  # 48 48
        data = resize(data).float()
        flip = np.random.random() > 0.5
        if flip:
            data = torch.flip(data, dims=(3,))
        data = function_nda(data)
        return data.float()

    def transform_test(data):
        resize = transforms.Resize(size=(48, 48))  # 48 48
        data = resize(data).float()
        return data.float()

    train_set = TransformedDatasetWrapper(train_set, transform=transform_train)
    test_set = TransformedDatasetWrapper(test_set, transform=transform_test)

    train_data_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_data_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Classify CIFAR10DVS')
    parser.add_argument('-T', '--T', default=10, type=int)
    parser.add_argument("-neuron", "--neuron_type", default='LIF', type=str)
    parser.add_argument(
        "-sc",
        "--spike_compressor",
        default="IdentitySpikeCompressor",
        type=str
    )
    parser.add_argument("-net", "--network", default="CIFAR10DVSVGG", type=str)
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/home/ma-user/work/datasets/CIFAR10DVS"
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)

    args = parser.parse_args()
    args.lomo = False
    args.amp = False
    args.loss = "tet"
    args.decay_lambda = 0.25
    args.batch_size = 32
    args.epochs = 100
    args.num_workers = 4
    args.learning_rate = 0.1
    args.momentum = 0.9
    args.l2_factor = 5e-4
    args.warmup = 10
    return args


def main():
    args = parse_args()
    print(args)

    set_seed(args.set_seed)

    run_name_generator = ModelNameGenerator(proj="cifar10dvs-me")
    run_name = run_name_generator.generate(args)
    log_path = Path(args.log_dir) / "CIFAR10DVS"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    log_path = log_path / (run_name+".time-prof.txt")

    _, val_data_loader = prepare_dataloaders(args)

    net = getattr(models, args.network)(
        T=args.T,  # for tebn and PSN
        neuron_type=args.neuron_type,
        spike_compressor=args.spike_compressor,
        decay_lambda=args.decay_lambda,
        k=4,  # for SlidingPSN
    )
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))
    net = net.to(args.device)

    profiler = LayerWiseFPCUDATimeProfiler(
        (net.features, net.dropout, net.classifier),
        search_mode=("direct_children", "self", "self"),
        model_names=("feature_extractor", "dropout", "classifier"),
        instances=(torch.nn.Module,),
        filename=log_path,
        warmup=args.warmup,
    )

    with torch.no_grad():
        for img, label in tqdm(val_data_loader):
            img, label = img.float().to(args.device), label.to(args.device)
            y = net(img)  # [T, N, Categories]
            functional.reset_net(net)

    profiler.clear_hooks()
    profiler.profile()


if __name__ == '__main__':
    main()
