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
from modules import get_autocast_context, GradScaler, BaseCheckpointingBlock
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Classify ImageNet (or its subset)'
    )
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
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)

    args = parser.parse_args()
    args.lomo = False
    args.amp = False
    args.T = 4
    args.mixup = False
    args.batch_size = 32
    args.epochs = 200
    args.num_workers = 4
    args.learning_rate = 1e-3
    args.l2_factor = 5e-2
    args.smoothing = 0.1
    args.cache_dataset = True
    return args


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
    log_path = log_path / (run_name+".prof-time.txt")

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
            *["direct_children" for _ in range(3 + net.depths)], "self"
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
    profiler = LayerWiseFPCUDATimeProfiler(
        model_list,
        search_mode=search_mode_list,
        model_names=model_name_list,
        instances=(torch.nn.Module,),
        filename=log_path,
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

    with torch.no_grad():
        for img, label in tqdm(val_data_loader):
            img = img.float().to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            y = net(img)
            functional.reset_net(net)

    profiler.clear_hooks()
    profiler.profile()


if __name__ == '__main__':
    main()
