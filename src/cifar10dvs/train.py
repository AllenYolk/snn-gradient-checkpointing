import sys

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks

from utils import Lomo, TMeanCrossEntropyLoss, TETLoss
import models
from modules import ClassificationLightningModule
from modules.lightning_callbacks import *
from data_module import CIFAR10DVSDataModule


class CIFAR10DVSLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        network: str,
        T: int,
        neuron_type: str,
        spike_compressor: str,
        decay_lambda: float,
        learning_rate: float,
        momentum: float,
        l2_factor: float,
        lomo: bool = False,
        loss: str = "tet",
    ):
        super().__init__(
            num_classes=10,
            network=network,
            T=T,
            neuron_type=neuron_type,
            spike_compressor=spike_compressor,
            decay_lambda=decay_lambda,
            learning_rate=learning_rate,
            momentum=momentum,
            l2_factor=l2_factor,
            lomo=lomo,
            loss=loss,
            y_with_T=True,
        )

    def configure_network(self):
        return getattr(models, self.hparams.network)(
            T=self.hparams.T,  # for PSN and tebn
            neuron_type=self.hparams.neuron_type,
            spike_compressor=self.hparams.spike_compressor,
            decay_lambda=self.hparams.decay_lambda,
            k=4,  # for SlidingPSN
        )

    def configure_criterion(self):
        if self.hparams.loss == "ce":
            return TMeanCrossEntropyLoss()
        else:
            return TETLoss(
                base_criterion=torch.nn.CrossEntropyLoss(),
                mean=1.,
                tet_lambda=1e-3,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.l2_factor
        )
        if self.hparams.lomo:
            optimizer = Lomo(optimizer, scaler=self.trainer.scaler)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return ([optimizer], [lr_scheduler])


def main():
    cli = LightningCLI(
        CIFAR10DVSLightningModule, CIFAR10DVSDataModule, run=False
    )
    cli.trainer.callbacks += [
        callbacks.ModelSummary(max_depth=-1),
        callbacks.ModelCheckpoint(
            filename="best-{epoch}-{train_acc:.4f}-{valid_acc:.4f}",
            save_top_k=1,
            monitor="val_acc",
            mode="max"
        ),
        callbacks.ModelCheckpoint(
            filename="lastest-{epoch}",
            save_top_k=1,
            monitor="epoch",
            mode="max"
        ),
        GlobalMeanBatchTimeCallback(reset_per_epoch=True),
        SamplePerSecondCallback(),
        PeakMemoryTillNowCallback(),
    ]
    if cli.trainer.is_global_zero:
        print(cli.model)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
