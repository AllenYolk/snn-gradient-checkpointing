import sys

sys.path.append("./src")
sys.path.append("./src/scifar")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()

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
        channels: int,
        neuron_type: str,
        num_classes: int,
        compress_x: bool,
        level: int,
        decay_lambda: float,
        learning_rate: float,
        momentum: float,
        lomo: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            channels=channels,
            neuron_type=neuron_type,
            compress_x=compress_x,
            level=level,
            decay_lambda=decay_lambda,
            learning_rate=learning_rate,
            momentum=momentum,
            lomo=lomo,
        )

    def configure_network(self):
        return models.GCSequentialCIFARNet(
            channels=self.hparams.channels,
            neuron_type=self.hparams.neuron_type,
            num_classes=self.hparams.num_classes,
            compress_x=self.hparams.compress_x,
            level=self.hparams.level,
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
    cli = LightningCLI(
        SCIFARLightningModule,
        SCIFARDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {"save_dir": "./logs", "name": "SCIFAR"},
            },
            "enable_model_summary": False,
            "enable_checkpointing": False,
        },
    )
    assert cli.model.hparams.num_classes == cli.datamodule.num_classes
    cli.trainer.callbacks += [
        callbacks.ModelSummary(max_depth=-1),
        callbacks.ModelCheckpoint(
            filename="best-{epoch}-{train_acc:.4f}-{val_acc:.4f}",
            save_top_k=1,
            monitor="val_acc",
            mode="max",
        ),
        GlobalMeanBatchTimeCallback(reset_per_epoch=True),
        SamplePerSecondCallback(),
        PeakMemoryTillNowCallback(),
    ]
    if cli.trainer.is_global_zero:
        print(cli.model)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
