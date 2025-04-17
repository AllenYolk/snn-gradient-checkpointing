import os
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import PIL

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import numpy as np
import wandb
import torch
from torch.cuda import amp
import torch.utils.data as data
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

import torchvision.transforms as transforms
import torchinfo
from spikingjelly.activation_based import functional, surrogate
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as SJCIFAR10DVS

from utils import set_seed, ModelNameGenerator, AverageMeter
from utils import accuracy, tet_loss_step
from utils import get_two_parameter_groups
from utils import get_optimizer_wrapper, OptimizerWrapperList, LRSchedulerList
from utils import move_state_dict_to_cpu
from utils.transforms import TransformedDatasetWrapper
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


def prepare_optimizers_and_schedulers(args, net):
    param_group1, param_group2 = get_two_parameter_groups(
        net, r"^((first)|(features\.(?:[0-9]|1[0-5])\.))", verbose=True
    )
    if args.optimizer == 'SGD':
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
    elif args.optimizer == 'Adam':
        optimizer1 = torch.optim.AdamW(
            param_group1, lr=args.learning_rate, weight_decay=args.l2_factor
        )
        optimizer2 = torch.optim.AdamW(
            param_group2, lr=args.learning_rate, weight_decay=args.l2_factor
        )
    else:
        raise NotImplementedError(args.opt)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler1 = torch.optim.lr_scheduler.StepLR(
            optimizer1, step_size=args.step_size, gamma=args.gamma
        )
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(
            optimizer2, step_size=args.step_size, gamma=args.gamma
        )
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=args.T_max
        )
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=args.T_max
        )
    else:
        raise NotImplementedError(args.lr_scheduler)

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
    parser.add_argument('-T', '--T', default=10, type=int)
    parser.add_argument('-dl', "--decay_lambda", default=0.5, type=float)
    parser.add_argument(
        '-b', "--batch_size", default=128, type=int, help='batch size'
    )
    parser.add_argument(
        '-e',
        '--epochs',
        default=300,
        type=int,
    )
    parser.add_argument(
        '-nw',
        "--num_workers",
        default=4,
        type=int,
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default="/home/ma-user/work/datasets/CIFAR10DVS"
    )
    parser.add_argument(
        '-amp',
        "--amp",
        action='store_true',
        help='automatic mixed precision training'
    )
    parser.add_argument(
        '-opt',
        "--optimizer",
        type=str,
        help='use which optimizer. SGD or Adam',
        default='SGD'
    )
    parser.add_argument(
        '-lr', "--learning_rate", default=0.1, type=float, help='learning rate'
    )
    parser.add_argument(
        '-mom', '--momentum', default=0.9, type=float, help='momentum for SGD'
    )
    parser.add_argument(
        '-sch',
        '--lr_scheduler',
        default='CosALR',
        type=str,
        help='use which schedule. StepLR or CosALR'
    )
    parser.add_argument(
        '-step_size',
        '--step_size',
        default=100,
        type=float,
        help='step_size for StepLR'
    )
    parser.add_argument(
        '-gamma', '--gamma', default=0.1, type=float, help='gamma for StepLR'
    )
    parser.add_argument(
        '-T_max',
        '--T_max',
        default=300,
        type=int,
        help='T_max for CosineAnnealingLR'
    )
    parser.add_argument(
        '-m', '--model', type=str, default='online_spiking_vgg11'
    )
    parser.add_argument('-ou', '--online_update', action='store_true')
    parser.add_argument(
        "-bn", "--batch_normalization", type=str, default="None"
    )
    parser.add_argument(
        "-ots", "--online_threshold_stabilization", action="store_true"
    )
    parser.add_argument('-l2', '--l2_factor', type=float, default=5e-4)
    parser.add_argument('-ll', '--loss_lambda', type=float, default=0.001)
    parser.add_argument("-rtnl", "--rate_till_now_loss", action="store_true")
    parser.add_argument('-d', '--device', default='cuda:0', type=str)
    parser.add_argument("-ss", "--set_seed", type=int, default=2024)
    parser.add_argument("-r", "--learning_rule", default="OTTT", type=str)
    parser.add_argument("-dr", "--dropout", default=0.1, type=float)
    parser.add_argument(
        "-ow", "--optimizer_wrapper", default="vanilla", type=str
    )
    parser.add_argument(
        "-owr", "--optimizer_wrapper_rho", default=0.5, type=float
    )
    parser.add_argument("-br", "--block_rho", default=0.0, type=float)

    return parser.parse_args()


def train_step(
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


def val_step(
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


def main():
    args = parse_args()
    print(args)

    set_seed(args.set_seed)

    run_name_generator = ModelNameGenerator(proj="cifar10dvs-online")
    run_name = run_name_generator.generate(args)
    model_name = run_name[:100] if len(run_name) > 100 else run_name
    wandb.require("core")
    wandb.init(
        project="cifar10dvs-online",
        entity="pkuml-spiking",
        config=args,
        name=run_name,
        # add the following line if 'InitStartError' occurs:
        # settings=wandb.Settings(start_method='fork')
    )
    log_dir = Path(wandb.run.dir)

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
    try:
        net(torch.randn(1, 2, 48, 48), init=True)
        torchinfo.summary(net, input_size=(1, 2, 48, 48))
    except Exception as e:
        print(net)
    net = net.to(args.device)

    optimizer, lr_scheduler = prepare_optimizers_and_schedulers(args, net)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    max_val_accuracy = 0.
    assert args.learning_rule != "BPTT"
    for epoch in range(args.epochs):
        train_results = train_step(
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
        val_results = val_step(
            net,
            val_data_loader,
            args.T,
            args.device,
            args.loss_lambda,
            args.rate_till_now_loss,
            10,
        )
        wandb.log({
            "loss/train_loss": train_results["loss"],
            "acc/train_top1_acc": train_results["top1_acc"],
            "acc/train_top5_acc": train_results["top5_acc"],
            "loss/val_loss": val_results["loss"],
            "acc/val_top1_acc": val_results["top1_acc"],
            "acc/val_top5_acc": val_results["top5_acc"],
        })
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_results['loss']}, "
            f"train_top1_acc={train_results['top1_acc']}, "
            f"train_top5_acc={train_results['top5_acc']}, "
            f"val_loss={val_results['loss']}, "
            f"val_top1_acc={val_results['top1_acc']}, "
            f"val_top5_acc={val_results['top5_acc']}"
        )
        if val_results["top1_acc"] > max_val_accuracy:
            max_val_accuracy = val_results["top1_acc"]
            torch.save(net.state_dict(), log_dir / f"{model_name}.pth")

    wandb.summary["acc/max_val_top1_acc"] = max_val_accuracy
    move_state_dict_to_cpu(log_dir / f"{model_name}.pth")
    wandb.log_model(log_dir / f"{model_name}.pth")
    os.remove(log_dir / f"{model_name}.pth")

    wandb.finish()


if __name__ == '__main__':
    main()
