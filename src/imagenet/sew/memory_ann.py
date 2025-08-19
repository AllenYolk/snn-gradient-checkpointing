import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append("./src")

import torch
import torch.utils.data as data
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from spikingjelly.activation_based import functional

from utils import set_seed, ModelNameGenerator, AverageMeter
from utils import accuracy, save_on_master, mkdir, count_learnable_parameters
from utils import TETLoss, TMeanCrossEntropyLoss, Lomo
from utils.profiler import *
from modules import get_autocast_context, GradScaler
import models


def prepare_dataloaders(args):

    def _get_cache_path(filepath):
        import hashlib
        h = hashlib.sha1(str(filepath).encode()).hexdigest()
        cache_path = Path("~/.torch/vision/datasets/imagefolder")
        cache_path = cache_path / (h[:10] + ".pt")
        cache_path = cache_path.expanduser()
        return cache_path

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    print("Loading data")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(train_dir)
    if args.cache_dataset and cache_path.exists():
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset_train, _ = torch.load(cache_path)
    else:
        dataset_train = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            mkdir(cache_path.parent)
            save_on_master((dataset_train, train_dir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(val_dir)
    if args.cache_dataset and cache_path.exists():
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        )
        if args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            mkdir(cache_path.parent)
            save_on_master((dataset_test, val_dir), cache_path)

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    print(
        f'dataset_train:{len(dataset_train)}, dataset_test:{len(dataset_test)}'
    )
    train_loader = data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def prepare_optimizers_and_schedulers(args, net, scaler):
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    if args.lomo:
        optimizer = Lomo(optimizer, scaler=scaler)

    return optimizer, lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify ImageNet (or its subset)'
    )
    parser.add_argument("-T", "--T", default=4, type=int)
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/export/home/data_allenyolk/ImageNet0_03125"
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument('-net', "--network", default="SEWResNet34", type=str)
    parser.add_argument('-neuron', "--neuron_type", default='SJLIF', type=str)
    parser.add_argument(
        "-sc",
        "--spike_compressor",
        default="IdentitySpikeCompressor",
        type=str
    )
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

    args = parser.parse_args()
    args.batch_size = 32
    args.epochs = 320
    args.num_workers = 4
    args.learning_rate = 0.1
    args.momentum = 0.9
    args.cache_dataset = True
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
    early_exit=False,
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
        for i, (img, label) in enumerate(pbar):
            img, label = img.float().to(device), label.to(device)

            with get_autocast_context(use_amp):
                y = net(img)
                batch_loss = criterion(y, label)

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
            prec1, prec5 = accuracy(y.data, label.data, topk=(1, 5))
            losses.update(batch_loss.item(), label.size(0))
            top1.update(prec1.item(), label.size(0))
            top5.update(prec5.item(), label.size(0))

            pbar.set_postfix({
                "loss": losses.avg,
                "top1_acc": top1.avg,
                "top5_acc": top5.avg,
            })

            if early_exit and i > 10:
                break

    if lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "loss": losses.avg,
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
    }


def val_step(net, test_data_loader, device, criterion):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.float().to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            y = net(img)
            batch_loss = criterion(y, label)

            # measure accuracy and record loss
            functional.reset_net(net)
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

    set_seed(args.set_seed)

    run_name_generator = ModelNameGenerator(proj=f"imagenet-sew-me")
    run_name = run_name_generator.generate(args)
    log_path = Path(args.log_dir) / "ImageNet-sew"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    mem_data_path = log_path / (run_name+".prof.pt")
    log_path = log_path / (run_name+".prof.txt")

    train_data_loader, val_data_loader = prepare_dataloaders(args)

    net = getattr(models, args.network)(
        neuron_type=args.neuron_type,
        T=args.T,
        spike_compressor=args.spike_compressor,
        decay_lambda=0.5,
        detach_reset=True,
        k=4,  # for SlidingPSN
    )
    print(net)
    print("Number of learnable parameters: ", count_learnable_parameters(net))
    net = net.to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    scaler = None
    if args.amp:
        scaler = GradScaler()
    optimizer, lr_scheduler = prepare_optimizers_and_schedulers(
        args, net, scaler
    )

    print("Stage 1: Peak Memory Checking")
    profiler = ProfilerList()

    torch.cuda.reset_peak_memory_stats(args.device)
    mem_stats = torch.cuda.memory_stats(args.device)
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        "Before training: "
        f"peak_allocated={peak_allocated:.2f} MB, "
        f"peak_reserved={peak_reserved:.2f} MB"
    )

    real_epochs = 1
    max_val_accuracy = 0.
    for epoch in range(real_epochs):
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
            real_epochs,
            early_exit=True,
        )
        val_results = val_step(
            net,
            val_data_loader,
            args.device,
            criterion,
        )
        mem_stats = torch.cuda.memory_stats(args.device)
        peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
        peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
        print(
            f"Epoch {epoch + 1}/{real_epochs}: "
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
    profiler = ProfilerList(
        CategoryMemoryProfiler(net, optimizer, log_path=log_path),
        LayerWiseMemoryProfiler(
            (
                net.layer1, net.layer2, net.layer3, net.layer4, net.avgpool,
                net.fc
            ),
            model_names=(
                "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"
            ),
            search_mode=(
                "direct_children", "direct_children", "direct_children",
                "direct_children", "self", "self"
            ),
            instances=(torch.nn.Module,),
            log_path=log_path,
            data_path=mem_data_path,
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

    for epoch in range(real_epochs, real_epochs + 1):
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
            real_epochs + 1,
            early_exit=True,
        )
        profiler.export()
        print(
            f"Profiling Epoch: "
            f"train_loss={train_results['loss']}, "
            f"train_top1_acc={train_results['top1_acc']}, "
            f"train_top5_acc={train_results['top5_acc']}, "
        )


if __name__ == '__main__':
    main()
