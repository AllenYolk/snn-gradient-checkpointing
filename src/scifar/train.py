import sys

sys.path.append("./src")
sys.path.append("./src/scifar")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks

from utils import Lomo
import models
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
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


def main():
    cli = LightningCLI(SCIFARLightningModule, SCIFARDataModule, run=False)
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
