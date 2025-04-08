import argparse
import sys
import threading
from tqdm import tqdm

sys.path.append("./src")
sys.path.append("./src/scifar")

import torch
import torch.nn.functional as F
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
from spikingjelly.activation_based import functional

from utils import set_seed, AverageMeter
from utils import accuracy, CategoryMemoryProfiler
from utils import LayerWiseMemoryProfiler, MemoryProfilerList
from augmentation import SequentialCIFARClassificationPresetTrain
from augmentation import CIFAR100_MEAN, CIFAR100_STD
from augmentation import CIFAR10_MEAN, CIFAR10_STD
from utils.transforms import RandomMixup, RandomCutmix
import models
from models import get_autocast_context, GradScaler, Lomo


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


def prepare_optimizers_and_schedulers(args, net):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=args.learning_rate,
        )
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    if args.lomo:
        optimizer = Lomo(optimizer)

    return optimizer, lr_scheduler


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
    parser.add_argument('-dl', "--decay_lambda", default=0.5, type=float)
    parser.add_argument(
        '--data_dir', type=str, default="/home/ma-user/work/datasets/CIFAR10"
    )
    parser.add_argument("-nc", "--num_classes", default=10, type=int)
    parser.add_argument(
        '-amp',
        "--amp",
        action='store_true',
        help='automatic mixed precision training'
    )
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)
    parser.add_argument("-lomo", "--lomo", action='store_true')

    args = parser.parse_args()
    args.epochs = 3
    args.channels = 128
    args.batch_size = 128
    args.num_workers = 4
    args.learning_rate = 0.1
    args.optimizer = "SGD"
    args.momentum = 0.9
    return args


def train_step(
    net, train_data_loader, optimizer, lr_scheduler, device, scaler, profiler,
    current_epoch, total_epoch
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
            img = img.permute(3, 0, 1, 2)  # [W, N, C, H]; W acts as T

            with get_autocast_context(use_amp):
                y = net(img)
                batch_loss = F.cross_entropy(y, label)
                profiler.profile()

            if use_amp:
                scaler.scale(batch_loss).backward()
                profiler.profile()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                profiler.profile()
                optimizer.step()
            optimizer.zero_grad()

            functional.reset_net(net)
            prec1, prec5 = accuracy(y.data, label.argmax(1).data, topk=(1, 5))
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


def val_step(net, test_data_loader, device, profiler):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for img, label in test_data_loader:
            img, label = img.float().to(device), label.to(device)
            img = img.permute(3, 0, 1, 2)  # [W, N, C, H]; W acts as T

            y = net(img)
            batch_loss = F.cross_entropy(y, label)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(y.data, label.data, topk=(1, 5))
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
    torch.cuda.reset_peak_memory_stats(args.device)

    set_seed(args.set_seed)

    train_data_loader, val_data_loader = prepare_dataloaders(args)

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
    net = net.to(args.device)

    optimizer, lr_scheduler = prepare_optimizers_and_schedulers(args, net)

    scaler = None
    if args.amp:
        scaler = GradScaler()

    profiler = MemoryProfilerList(
        CategoryMemoryProfiler(net, optimizer, filename='snn_memory.prof'),
        LayerWiseMemoryProfiler(
            net, instances=(torch.nn.Module,), filename='snn_memory.prof'
        ),
    )
    mem_stats = torch.cuda.memory_stats(args.device)
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"Before training: "
        f"Peak allocated memory: {peak_allocated} MB, "
        f"Peak reserved memory: {peak_reserved} MB"
    )
    profiler.profile()

    max_val_accuracy = 0.
    torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(args.epochs):
        train_results = train_step(
            net,
            train_data_loader,
            optimizer,
            lr_scheduler,
            args.device,
            scaler,
            profiler,
            epoch,
            args.epochs,
        )
        profiler.profile()
        val_results = val_step(
            net,
            val_data_loader,
            args.device,
            profiler,
        )
        profiler.profile()

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
            f"peak_allocated={peak_allocated} MB, "
            f"peak_reserved={peak_reserved} MB"
        )
        if val_results["top1_acc"] > max_val_accuracy:
            max_val_accuracy = val_results["top1_acc"]

    mem_stats = torch.cuda.memory_stats(args.device)
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"Peak allocated memory: {peak_allocated} MB\n"
        f"Peak reserved memory: {peak_reserved} MB"
    )


if __name__ == '__main__':
    main()
