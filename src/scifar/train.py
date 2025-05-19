import time
import sys
from tqdm import tqdm

sys.path.append("./src")

import torch
import torch.nn.functional as F
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from spikingjelly.activation_based import functional
from lightning.pytorch.cli import LightningCLI

from utils import ModelNameGenerator, AverageMeter
from utils import accuracy, Lomo
from modules import get_autocast_context
import models as models
from modules import ClassificationLightningModule
from data_module import SCIFARDataModule


class SCIFARLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        num_classes: int,
        network: str,
        channels: int,
        neuron_type: str,
        spike_compressor: str,
        decay_lambda: float,
        learning_rate: float,
        momentum: float,
        lomo: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            network=network,
            channels=channels,
            neuron_type=neuron_type,
            spike_compressor=spike_compressor,
            decay_lambda=decay_lambda,
            learning_rate=learning_rate,
            momentum=momentum,
            lomo=lomo
        )

    def configure_network(self):
        return getattr(models, self.hparams.network)(
            channels=self.hparams.channels,
            neuron_type=self.hparams.neuron_type,
            spike_compressor=self.hparams.spike_compressor,
            num_classes=self.hparams.num_classes,
            decay_lambda=self.hparams.decay_lambda,
            T=32,  # for PSN
            k=8,  # for SlidingPSN
        )

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
        )
        if self.hparams.lomo:
            optimizer = Lomo(optimizer, scaler=self.trainer.scaler)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )

        return ([optimizer], [lr_scheduler])


def train_step(
    net, train_data_loader, optimizer, lr_scheduler, device, scaler,
    current_epoch, total_epoch
):
    epoch_start_time = time.time()
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

            if use_amp:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
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

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time  # unit: second
    return {
        "loss": losses.avg,
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
        "time": epoch_time
    }


def val_step(net, test_data_loader, device):
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
    cli = LightningCLI(SCIFARLightningModule, SCIFARDataModule, run=False)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
