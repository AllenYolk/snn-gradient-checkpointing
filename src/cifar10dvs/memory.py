import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import PIL

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import numpy as np
import torch
import torch.nn as nn
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

from utils import set_seed, ModelNameGenerator, AverageMeter
from utils import accuracy, TETLoss, count_learnable_parameters
from utils.transforms import TransformedDatasetWrapper
from utils.profiler import *
from augmentation import CIFAR10DVSNDA
from cifar10dvs_dataset import CIFAR10DVS, move_data
import models
from models.optimizer import Lomo
from models.amp import get_autocast_context, GradScaler


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


def prepare_optimizers_and_schedulers(args, net):
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.l2_factor
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    if args.lomo:
        optimizer = Lomo(optimizer)

    return optimizer, lr_scheduler


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
    parser.add_argument(
        '-amp',
        "--amp",
        action='store_true',
        help='automatic mixed precision training'
    )
    parser.add_argument(
        "-loss", "--loss", default="tet", type=str, choices=["ce", "tet"]
    )
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)
    parser.add_argument("-lomo", "--lomo", action='store_true')
    parser.add_argument("-tebn", "--allow_tebn", action='store_true')

    args = parser.parse_args()
    args.decay_lambda = 0.25
    args.batch_size = 32
    args.epochs = 2
    args.num_workers = 4
    args.learning_rate = 0.1
    args.momentum = 0.9
    args.l2_factor = 5e-4
    return args


def train_step(
    net,
    train_data_loader,
    criterion,
    optimizer,
    lr_scheduler,
    device,
    scaler,
    profiler,
    current_epoch,
    total_epoch,
):
    net.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    use_amp = scaler is not None

    with tqdm(
        train_data_loader,
        desc=f"Epoch {current_epoch + 1}/{total_epoch}",
        leave=False,
        unit="batch"
    ) as pbar:
        for img, label in pbar:
            img, label = img.float().to(device), label.to(device)

            with get_autocast_context(use_amp):
                y = net(img)  # [T, N, Categories]
                batch_loss = criterion(y, label)
                profiler.profile(sort_by="forward_peak_memory")

            if use_amp:
                scaler.scale(batch_loss).backward()
                profiler.profile(sort_by="backward_peak_memory")
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                profiler.profile(sort_by="backward_peak_memory")
                optimizer.step()
            optimizer.zero_grad()

            functional.reset_net(net)
            prec1, prec5 = accuracy(y.mean(dim=0).data, label.data, topk=(1, 5))
            losses.update(batch_loss.item(), label.size(0))
            top1.update(prec1.item(), label.size(0))
            top5.update(prec5.item(), label.size(0))

            pbar.set_postfix({
                "loss": losses.avg,
                "top1_acc": top1.avg,
                "top5_acc": top5.avg,
            })

    if lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "loss": losses.avg,
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
    }


def val_step(net, test_data_loader, device, criterion, profiler):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for img, label in test_data_loader:
            img, label = img.float().to(device), label.to(device)

            y = net(img)  # [T, N, Categories]
            batch_loss = criterion(y, label)
            profiler.profile(sort_by="forward_peak_memory")

            functional.reset_net(net)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(y.mean(dim=0).data, label.data, topk=(1, 5))
            losses.update(batch_loss.item(), label.size(0))
            top1.update(prec1.item(), label.size(0))
            top5.update(prec5.item(), label.size(0))

    return {
        "loss": losses.avg,
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
    }


def main():
    args = parse_args()
    print(args)

    set_seed(args.set_seed)

    run_name_generator = ModelNameGenerator(proj="cifar10dvs-me")
    run_name = run_name_generator.generate(args)
    log_path = Path(args.log_dir) / "CIFAR10DVS" / (run_name+"prof.txt")

    train_data_loader, val_data_loader = prepare_dataloaders(args)

    net = getattr(models, args.network)(
        T=args.T,  # for tebn and PSN
        neuron_type=args.neuron_type,
        spike_compressor=args.spike_compressor,
        allow_tebn=args.allow_tebn,
        decay_lambda=args.decay_lambda,
        k=8,  # for SlidingPSN
    )
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))
    net = net.to(args.device)

    if args.loss == "ce":
        criterion = nn.Sequential(
            models.AverageT(),
            torch.nn.CrossEntropyLoss(),
        )
    else:
        criterion = TETLoss(
            base_criterion=torch.nn.CrossEntropyLoss(),
            mean=1.,
            tet_lambda=1e-3,
        )
    optimizer, lr_scheduler = prepare_optimizers_and_schedulers(args, net)

    scaler = None
    if args.amp:
        scaler = GradScaler()

    print("Stage 1: Peak Memory Checking")
    profiler = MemoryProfilerList()

    torch.cuda.reset_peak_memory_stats(args.device)
    mem_stats = torch.cuda.memory_stats(args.device)
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        "Before training: "
        f"peak_allocated={peak_allocated:.2f} MB, "
        f"peak_reserved={peak_reserved:.2f} MB"
    )

    max_val_accuracy = 0.
    for epoch in range(args.epochs):
        train_results = train_step(
            net,
            train_data_loader,
            criterion,
            optimizer,
            lr_scheduler,
            args.device,
            scaler,
            profiler,
            epoch,
            args.epochs,
        )
        val_results = val_step(
            net,
            val_data_loader,
            args.device,
            criterion,
            profiler,
        )
        mem_stats = torch.cuda.memory_stats(args.device)
        peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
        peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_results['loss']}, "
            f"train_top1_acc={train_results['top1_acc']}, "
            f"train_top5_acc={train_results['top5_acc']}, "
            f"val_loss={val_results['loss']}, "
            f"val_top1_acc={val_results['top1_acc']}, "
            f"val_top5_acc={val_results['top5_acc']}, "
            f"peak_allocated={peak_allocated:.2f} MB, "
            f"peak_reserved={peak_reserved:.2f} MB"
        )
        if val_results["top1_acc"] > max_val_accuracy:
            max_val_accuracy = val_results["top1_acc"]

    print("Stage 2: Memory Profiling")
    profiler = MemoryProfilerList(
        CategoryMemoryProfiler(net, optimizer, filename=log_path),
        LayerWiseMemoryProfiler(
            (net.features, net),
            instances=(torch.nn.Module,),
            filename=log_path,
            direct_children_only=(True, True),
        ),
    )

    torch.cuda.reset_peak_memory_stats(args.device)
    mem_stats = torch.cuda.memory_stats(args.device)
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        "Before training: "
        f"peak_allocated={peak_allocated:.2f} MB, "
        f"peak_reserved={peak_reserved:.2f} MB"
    )

    for epoch in range(args.epochs, 2 * args.epochs):
        train_results = train_step(
            net,
            train_data_loader,
            criterion,
            optimizer,
            lr_scheduler,
            args.device,
            scaler,
            profiler,
            epoch,
            2 * args.epochs,
        )
        val_results = val_step(
            net,
            val_data_loader,
            args.device,
            criterion,
            profiler,
        )
        mem_stats = torch.cuda.memory_stats(args.device)
        peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
        peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
        print(
            f"Epoch {epoch}/{args.epochs*2}: "
            f"train_loss={train_results['loss']}, "
            f"train_top1_acc={train_results['top1_acc']}, "
            f"train_top5_acc={train_results['top5_acc']}, "
            f"val_loss={val_results['loss']}, "
            f"val_top1_acc={val_results['top1_acc']}, "
            f"val_top5_acc={val_results['top5_acc']}, "
            f"peak_allocated={peak_allocated:.2f} MB, "
            f"peak_reserved={peak_reserved:.2f} MB"
        )
        if val_results["top1_acc"] > max_val_accuracy:
            max_val_accuracy = val_results["top1_acc"]


if __name__ == '__main__':
    main()
