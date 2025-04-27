import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import PIL

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
from timm.data import create_transform, Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler_v2
from timm.optim import create_optimizer_v2

from utils import set_seed, ModelNameGenerator, AverageMeter, Lomo
from utils import accuracy, save_on_master, mkdir, count_learnable_parameters
from utils.profiler import *
from modules import get_autocast_context, GradScaler
import models


def prepare_dataloaders(args):

    def _build_transform(is_train):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if is_train:
            # train transform
            transform = create_transform(
                input_size=224,
                is_training=True,
                auto_augment="rand-m9-mstd0.5-inc1",
                interpolation='bicubic',
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                mean=mean,
                std=std,
            )
            return transform
        else:
            # eval transform
            t = []
            t.append(transforms.Resize(256, interpolation=PIL.Image.BICUBIC))
            t.append(transforms.CenterCrop(224))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            return transforms.Compose(t)

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

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(train_dir)
    if args.cache_dataset and cache_path.exists():
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset_train, _ = torch.load(cache_path)
    else:
        dataset_train = datasets.ImageFolder(
            train_dir, transform=_build_transform(is_train=True)
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
            val_dir, transform=_build_transform(is_train=False)
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
        pin_memory=True,
        drop_last=True
    )

    test_loader = data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    if args.mixup:
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.smoothing,
            num_classes=1000,
        )
    else:
        mixup_fn = None

    return train_loader, test_loader, mixup_fn


def prepare_optimizers_and_schedulers(
    args, net, scaler, train_loader_length=None
):
    optimizer = create_optimizer_v2(
        net,
        opt="adamw",
        lr=args.learning_rate,
        weight_decay=args.l2_factor,
    )

    lr_scheduler, total_epochs = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=args.epochs,
        min_lr=1e-5,
        warmup_epochs=20,
        warmup_lr=1e-6,
        cooldown_epochs=10,
        step_on_epochs=False,
        updates_per_epoch=train_loader_length,
    )
    if args.epochs != total_epochs:
        print(
            f"Number of epochs changed from {args.epochs} to {total_epochs}"
            f" due to the scheduler."
        )
        args.epochs = total_epochs

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
    parser.add_argument('-net', "--network", default="Spikformer", type=str)
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
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)
    parser.add_argument("-lomo", "--lomo", action='store_true')

    args = parser.parse_args()
    args.mixup = False
    args.batch_size = 32
    args.epochs = 200
    args.num_workers = 4
    args.learning_rate = 1e-3
    args.l2_factor = 5e-2
    args.smoothing = 0.1
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
    mixup_fn,
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
            img = img.float().to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            if mixup_fn is not None:
                img, label = mixup_fn(img, label)

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
            label_idx = label.argmax(dim=1) if mixup_fn is not None else label
            prec1, prec5 = accuracy(y.data, label_idx.data, topk=(1, 5))
            losses.update(batch_loss.item(), label.size(0))
            top1.update(prec1.item(), label.size(0))
            top5.update(prec5.item(), label.size(0))

            pbar.set_postfix({
                "loss": losses.avg,
                "top1_acc": top1.avg,
                "top5_acc": top5.avg,
            })

            if lr_scheduler is not None:
                # will prevent scheduler update that should happen
                # at the end of the epoch
                lr_scheduler.step_update(
                    i + current_epoch * len(train_data_loader)
                )

            if early_exit and i > 10:
                break

    if lr_scheduler is not None:
        # will prevent scheduler update that should happen at each iteration
        lr_scheduler.step(current_epoch + 1)

    return {
        "loss": losses.avg,
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
    }


def val_step(net, test_data_loader, device):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss()

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

    run_name_generator = ModelNameGenerator(proj=f"imagenet-transformer-me")
    run_name = run_name_generator.generate(args)
    log_path = Path(args.log_dir) / "ImageNet-transformer"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    mem_data_path = log_path / (run_name+".prof.pt")
    log_path = log_path / (run_name+".prof.txt")

    train_data_loader, val_data_loader, mixup_fn = prepare_dataloaders(args)

    net = getattr(models, args.network)(
        neuron_type=args.neuron_type,
        T=args.T,
        spike_compressor=args.spike_compressor,
        decay_lambda=0.5,
        detach_reset=True,
        k=4,  # for SlidingPSN
    )
    print(net)
    print(f"Number of learnable parameters: {count_learnable_parameters(net)}")
    net = net.to(args.device)

    if mixup_fn is not None:
        print(f"Mixup enabled!")
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print(f"Criterion: {criterion}")

    scaler = None
    if args.amp:
        scaler = GradScaler()
    optimizer, lr_scheduler = prepare_optimizers_and_schedulers(
        args, net, scaler, len(train_data_loader)
    )

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
            mixup_fn,
            profiler,
            epoch,
            real_epochs,
        )
        val_results = val_step(
            net,
            val_data_loader,
            args.device,
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
    if args.network.endswith("Spikformer"):
        model_list = (net.patch_embed, *[b for b in net.block], net.head)
        search_mode_list = (
            "direct_children", *["direct_children" for _ in net.block], "self"
        )
        model_name_list = (
            "patch_embed", *[f"block{i}" for i in range(net.depths)], "head"
        )
    elif args.network.endswith("QKFormer"):
        model_list = (
            net.patch_embed1, *[b for b in net.block1], net.patch_embed2,
            *[b for b in net.block2], net.patch_embed3,
            *[b for b in net.block3], net.head
        )
        search_mode_list = (
            "direct_children", *["direct_children" for _ in net.block1],
            "direct_children", *["direct_children" for _ in net.block2],
            "direct_children", *["direct_children" for _ in net.block3], "self"
        )
        model_name_list = (
            "patch_embed1",
            *[f"block1_{i}" for i in range(1)],
            "patch_embed2",
            *[f"block2_{i}" for i in range(2)],
            "patch_embed3",
            *[f"block3_{i}" for i in range(net.depths - 3)],
            "head",
        )
    profiler = MemoryProfilerList(
        CategoryMemoryProfiler(net, optimizer, filename=log_path),
        LayerWiseMemoryProfiler(
            model_list,
            search_mode=search_mode_list,
            model_names=model_name_list,
            instances=(torch.nn.Module,),
            filename=log_path,
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
            mixup_fn,
            profiler,
            epoch,
            real_epochs + 1,
            early_exit=True,
        )
        profiler.save_data((None, mem_data_path))
        print(
            f"Profiling Epoch: "
            f"train_loss={train_results['loss']}, "
            f"train_top1_acc={train_results['top1_acc']}, "
            f"train_top5_acc={train_results['top5_acc']}, "
        )


if __name__ == '__main__':
    main()
