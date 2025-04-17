import os
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import PIL

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import numpy as np
import torch
from torch.cuda import amp
import torch.utils.data as data
import torch.nn.functional as F
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

import torchvision.transforms as transforms
from spikingjelly.activation_based import functional, surrogate
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as SJCIFAR10DVS
# from spikingjelly.datasets import split_to_train_test_set

from utils import set_seed, AverageMeter
from utils import accuracy, tet_loss_step
from utils import get_two_parameter_groups
from utils import get_optimizer_wrapper, OptimizerWrapperList, LRSchedulerList
from utils.transforms import TransformedDatasetWrapper
from augmentation import CIFAR10DVSNDA
from cifar10dvs_dataset import CIFAR10DVS, move_data
from models import spiking_vgg


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

    # load specific split
    trainset = CIFAR10DVS(
        args.data_dir,
        train=True,
        data_type='frame',
        frames_number=args.T,
        split_by='number'
    )
    testset = CIFAR10DVS(
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

    trainset = TransformedDatasetWrapper(trainset, transform=transform_train)
    testset = TransformedDatasetWrapper(testset, transform=transform_test)

    train_data_loader = data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_data_loader = data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_data_loader, test_data_loader


def prepare_optimizers_and_schedulers(args, net):
    param_group1, param_group2 = get_two_parameter_groups(
        net, r"^((first)|(features\.(?:[0-9]|1[0-5])\.))", verbose=True
    )
    optimizer1 = torch.optim.SGD(
        param_group1,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.l2_factor
    )
    optimizer2 = torch.optim.SGD(
        param_group2,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.l2_factor
    )

    lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=args.T_max
    )
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=args.T_max
    )

    optimizer1 = get_optimizer_wrapper(
        method=args.optimizer_wrapper,
        optimizer=optimizer1,
        rho=args.optimizer_wrapper_rho
    )
    optimizer2 = get_optimizer_wrapper(method="vanilla", optimizer=optimizer2)

    return (
        OptimizerWrapperList([optimizer1, optimizer2]),
        LRSchedulerList([lr_scheduler1, lr_scheduler2])
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Classify CIFAR10DVS')
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/home/ma-user/work/datasets/CIFAR10DVS"
    )
    parser.add_argument('-ou', '--online_update', action='store_true')
    parser.add_argument(
        "-oorl", "--offline_overall_rate_loss", action="store_true"
    )
    parser.add_argument(
        "-bn", "--batch_normalization", type=str, default="None"
    )
    parser.add_argument(
        "-ots", "--online_threshold_stabilization", action="store_true"
    )
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-r", "--learning_rule", default="OTTT", type=str)
    parser.add_argument(
        "-ow", "--optimizer_wrapper", default="vanilla", type=str
    )
    parser.add_argument(
        "-owr", "--optimizer_wrapper_rho", default=0.5, type=float
    )

    args = parser.parse_args()
    args.T = 10
    args.decay_lambda = 0.5
    args.batch_size = 128
    args.epochs = 2
    args.num_workers = 0
    args.amp = False
    args.learning_rate = 0.1
    args.momentum = 0.9
    args.l2_factor = 5e-4
    args.T_max = 2
    args.loss_lambda = 0.001
    args.rate_till_now_loss = False
    args.set_seed = 2024
    args.dropout = 0.1
    args.block_rho = 0.0
    args.model = "online_spiking_vgg11"

    return args


def train_step_online(
    net,
    train_data_loader,
    T,
    optimizer,
    lr_scheduler,
    device,
    online_update,
    use_amp,
    loss_lambda,
    rate_till_now_loss,
    num_classes,
    scaler,
    current_epoch,
    total_epoch,
):
    net.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(
        train_data_loader,
        desc=f"Epoch {current_epoch + 1}/{total_epoch}",
        leave=False,
        unit="batch"
    ) as pbar:
        for img, label in pbar:
            img, label = img.float().to(device), label.to(device)
            batch_loss = 0.
            total_spike_count = 0.  # \sum_{t}(WS[t]+b); detached

            if not online_update:
                optimizer.zero_grad()
            optimizer.reset_state()

            for t in range(T):
                if online_update:
                    optimizer.zero_grad()
                with amp.autocast(enabled=use_amp):
                    # calculate loss and total_spike_count
                    if rate_till_now_loss:
                        # Even if online_update is True, we use the same method
                        # as that of online_update=False to calculate rate_till_now,
                        # since we assume that the online update is small and
                        # has negligible effects on rate_till_now. See the OTTT
                        # paper for more details.
                        frame = img[:, t]
                        out_spike = net(frame, init=(t == 0))  # WS[t]+b
                        rate_till_now = total_spike_count + out_spike
                        rate_till_now = rate_till_now / (t+1)
                        loss = tet_loss_step(
                            rate_till_now,
                            label,
                            loss_lambda,
                            target_is_label=True,
                            num_classes=num_classes
                        ) / T
                        total_spike_count += out_spike.clone().detach()
                    else:
                        frame = img[:, t]
                        out_spike = net(frame, init=(t == 0))  # WS[t]+b
                        loss = tet_loss_step(
                            out_spike,
                            label,
                            loss_lambda,
                            target_is_label=True,
                            num_classes=num_classes
                        ) / T
                        total_spike_count += out_spike.clone().detach()

                if use_amp and (scaler is not None):
                    scaler.scale(loss).backward()
                    optimizer.minor_step(t)
                    if online_update:
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss.backward()
                    optimizer.minor_step(t)
                    if online_update:
                        optimizer.step()
                batch_loss += loss.item()

            if not online_update:
                if use_amp and (scaler is not None):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            functional.reset_net(net)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(
                total_spike_count.data, label.data, topk=(1, 5)
            )
            losses.update(batch_loss, frame.size(0))
            top1.update(prec1.item(), frame.size(0))
            top5.update(prec5.item(), frame.size(0))

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


def train_step_offline(
    net, train_data_loader, T, optimizer, lr_scheduler, device, use_amp,
    loss_lambda, rate_till_now_loss, offline_overall_rate_loss, scaler,
    current_epoch, total_epoch, num_classes
):
    net.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(
        train_data_loader,
        desc=f"Epoch {current_epoch + 1}/{total_epoch}",
        leave=False,
        unit="batch"
    ) as pbar:
        for img, label in pbar:
            optimizer.zero_grad()
            optimizer.reset_state()

            img, label = img.float().to(device), label.to(device)
            label = label.to(torch.int64)

            with amp.autocast(enabled=use_amp):
                batch_loss = 0.
                total_spike_count = 0.  # should be detached from the graph
                if offline_overall_rate_loss:
                    overall_firing_rate = 0.
                    for t in range(T):
                        frame = img[:, t]
                        out_spike = net(frame, init=(t == 0))
                        total_spike_count += out_spike.clone().detach()
                        overall_firing_rate += out_spike
                    overall_firing_rate /= T
                    batch_loss = F.cross_entropy(overall_firing_rate, label)
                else:
                    for t in range(T):
                        frame = img[:, t]
                        out_spike = net(frame, init=(t == 0))
                        if rate_till_now_loss:
                            rate_till_now = total_spike_count + out_spike
                            rate_till_now = rate_till_now / (t+1)
                            loss = tet_loss_step(
                                rate_till_now,
                                label,
                                loss_lambda,
                                target_is_label=True,
                                num_classes=num_classes
                            ) / T
                        else:
                            loss = tet_loss_step(
                                out_spike,
                                label,
                                loss_lambda,
                                target_is_label=True,
                                num_classes=num_classes
                            ) / T
                        batch_loss += loss
                        total_spike_count += out_spike.clone().detach()

            if scaler is not None:
                scaler.scale(batch_loss).backward()
                optimizer.minor_step(t)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.minor_step(t)
                optimizer.step()

            functional.reset_net(net)
            prec1, prec5 = accuracy(
                total_spike_count.data, label.data, topk=(1, 5)
            )
            losses.update(batch_loss.item(), frame.size(0))
            top1.update(prec1.item(), frame.size(0))
            top5.update(prec5.item(), frame.size(0))

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


def val_step_online(
    net,
    test_data_loader,
    T,
    device,
    loss_lambda,
    rate_till_now_loss,
    num_classes,
):
    net.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for img, label in test_data_loader:
            img, label = img.float().to(device), label.to(device)
            batch_loss = 0.
            total_spike_count = 0.  # \sum_{t}(WS[t]+b)

            for t in range(T):
                frame = img[:, t]
                out_spike = net(frame, init=(t == 0))  # WS[t]+b
                # calculate loss
                if rate_till_now_loss:
                    rate_till_now = total_spike_count + out_spike
                    rate_till_now = rate_till_now / (t+1)
                    loss = tet_loss_step(
                        rate_till_now,
                        label,
                        loss_lambda,
                        target_is_label=True,
                        num_classes=num_classes
                    ) / T
                else:
                    loss = tet_loss_step(
                        out_spike,
                        label,
                        loss_lambda,
                        target_is_label=True,
                        num_classes=num_classes
                    ) / T
                total_spike_count += out_spike.clone().detach()
                batch_loss += loss.item()

            functional.reset_net(net)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(
                total_spike_count.data, label.data, topk=(1, 5)
            )
            losses.update(batch_loss, frame.size(0))
            top1.update(prec1.item(), frame.size(0))
            top5.update(prec5.item(), frame.size(0))

    return {
        "loss": losses.avg,
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
    }


def val_step_offline(
    net, test_data_loader, T, device, loss_lambda, rate_till_now_loss,
    offline_overall_rate_loss, num_classes
):
    net.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for img, label in test_data_loader:
            img, label = img.float().to(device), label.to(device)
            label = label.to(torch.int64)

            batch_loss = 0.
            total_spike_count = 0.
            if offline_overall_rate_loss:
                for t in range(T):
                    frame = img[:, t]
                    out_spike = net(frame, init=(t == 0))
                    total_spike_count += out_spike.clone().detach()
                batch_loss = F.cross_entropy(total_spike_count / T, label)
            else:
                for t in range(T):
                    frame = img[:, t]
                    out_spike = net(frame, init=(t == 0))
                    if rate_till_now_loss:
                        rate_till_now = total_spike_count + out_spike
                        rate_till_now = rate_till_now / (t+1)
                        loss = tet_loss_step(
                            rate_till_now,
                            label,
                            loss_lambda,
                            target_is_label=True,
                            num_classes=num_classes
                        ) / T
                    else:
                        loss = tet_loss_step(
                            out_spike,
                            label,
                            loss_lambda,
                            target_is_label=True,
                            num_classes=num_classes
                        ) / T
                    batch_loss += loss
                    total_spike_count += out_spike.clone().detach()

            functional.reset_net(net)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(
                total_spike_count.data, label.data, topk=(1, 5)
            )
            losses.update(batch_loss.item(), frame.size(0))
            top1.update(prec1.item(), frame.size(0))
            top5.update(prec5.item(), frame.size(0))

    return {
        "loss": losses.avg,
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
    }


def main():
    args = parse_args()
    print(args)

    set_seed(args.set_seed)

    train_data_loader, val_data_loader = prepare_dataloaders(args)

    net = spiking_vgg.__dict__[args.model](
        c_in=2,
        fc_hw=1,
        bn=args.batch_normalization,
        num_classes=10,
        learning_rule=args.learning_rule,
        light_classifier=True,
        T=args.T,
        osr_bound=50.,
        block_rho=args.block_rho,
        decay_lambda=args.decay_lambda,
        surrogate_function=surrogate.Sigmoid(alpha=4.),
        online_threshold_stabilization=args.online_threshold_stabilization,
        v_reset=None,  # soft reset
        dropout=args.dropout,
    )
    functional.set_step_mode(net, step_mode='s')
    net = net.to(args.device)

    optimizer, lr_scheduler = prepare_optimizers_and_schedulers(args, net)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    torch.cuda.reset_peak_memory_stats(args.device)
    for epoch in range(args.epochs):
        if args.learning_rule == "BPTT":
            train_results = train_step_offline(
                net,
                train_data_loader,
                args.T,
                optimizer,
                lr_scheduler,
                args.device,
                args.amp,
                args.loss_lambda,
                args.rate_till_now_loss,
                args.offline_overall_rate_loss,
                scaler,
                epoch,
                args.epochs,
                10,
            )
            val_results = val_step_offline(
                net,
                val_data_loader,
                args.T,
                args.device,
                args.loss_lambda,
                args.rate_till_now_loss,
                args.offline_overall_rate_loss,
                10,
            )
        else:
            train_results = train_step_online(
                net,
                train_data_loader,
                args.T,
                optimizer,
                lr_scheduler,
                args.device,
                args.online_update,
                args.amp,
                args.loss_lambda,
                args.rate_till_now_loss,
                10,
                scaler,
                epoch,
                args.epochs,
            )
            val_results = val_step_online(
                net,
                val_data_loader,
                args.T,
                args.device,
                args.loss_lambda,
                args.rate_till_now_loss,
                10,
            )

        mem_stats = torch.cuda.memory_stats(args.device)
        peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
        peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_results['loss']:.2f}, "
            f"train_top1_acc={train_results['top1_acc']:.2f}, "
            f"train_top5_acc={train_results['top5_acc']:.2f}, "
            f"val_loss={val_results['loss']:.2f}, "
            f"val_top1_acc={val_results['top1_acc']:.2f}, "
            f"val_top5_acc={val_results['top5_acc']:.2f}, "
            f"peak_allocated={peak_allocated:.2f} MB, "
            f"peak_reserved={peak_reserved:.2f} MB"
        )

    mem_stats = torch.cuda.memory_stats(args.device)
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"Peak allocated memory: {peak_allocated:.2f} MB, "
        f"Peak reserved memory: {peak_reserved:.2f} MB"
    )


if __name__ == '__main__':
    main()
