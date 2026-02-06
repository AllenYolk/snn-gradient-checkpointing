"""Modified from SEW ResNet source code."""

import sys

sys.path.append("./src")
sys.path.append("./src/imagenet")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks

import models
from data_module import ImageNetDataModule
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
from utils import TETLoss, TMeanCrossEntropyLoss, Lomo


class SEWImageNetLightningModule(ClassificationLightningModule):
    def __init__(
        self,
        neuron_type: str,
        compress_x: bool,
        level: int,
        T: int,
        learning_rate: float,
        momentum: float,
        loss: str,
        lomo: bool = False,
    ):
        super().__init__(
            num_classes=1000,
            y_with_T=True,
            neuron_type=neuron_type,
            compress_x=compress_x,
            level=level,
            T=T,
            learning_rate=learning_rate,
            momentum=momentum,
            loss=loss,
            lomo=lomo,
        )

    def configure_network(self):
        return models.GCSEWResNet34(
            neuron_type=self.hparams.neuron_type,
            compress_x=self.hparams.compress_x,
            level=self.hparams.level,
            T=self.hparams.T,
            decay_lambda=0.5,
            detach_reset=True,
            k=4,  # for SlidingPSN
        )

    def configure_criterion(self):
        if self.hparams.loss == "ce":
            return TMeanCrossEntropyLoss()
        elif self.hparams.loss == "tet":
            return TETLoss(base_criterion=torch.nn.CrossEntropyLoss(), tet_lambda=0.0)
        else:
            raise ValueError(f"`loss` should be either 'ce' or 'tet'")

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
        SEWImageNetLightningModule,
        ImageNetDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {"save_dir": "./logs", "name": "ImageNet-sew"},
            },
            "enable_model_summary": False,
            "enable_checkpointing": False,
        },
    )
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
