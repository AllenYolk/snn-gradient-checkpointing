import sys

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks

from utils import Lomo, TMeanCrossEntropyLoss, TETLoss
import models
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
from data_module import CIFAR10DVSDataModule


class CIFAR10DVSLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        T: int,
        neuron_type: str,
        compress_x: bool,
        level: int,
        decay_lambda: float,
        learning_rate: float,
        momentum: float,
        l2_factor: float,
        lomo: bool = False,
        loss: str = "tet",
    ):
        super().__init__(
            num_classes=10,
            T=T,
            neuron_type=neuron_type,
            compress_x=compress_x,
            level=level,
            decay_lambda=decay_lambda,
            learning_rate=learning_rate,
            momentum=momentum,
            l2_factor=l2_factor,
            lomo=lomo,
            loss=loss,
            y_with_T=True,
        )

    def configure_network(self):
        return models.FLGCCIFAR10DVSVGG(
            T=self.hparams.T,
            neuron_type=self.hparams.neuron_type,
            compress_x=self.hparams.compress_x,
            level=self.hparams.level,
            decay_lambda=self.hparams.decay_lambda,
            k=2,  # for SlidingPSN
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
        CIFAR10DVSLightningModule,
        CIFAR10DVSDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {
                    "save_dir": "./logs",
                    "name": "CIFAR10DVS"
                }
            },
            "enable_model_summary": False,
            "enable_checkpointing": False,
        }
    )
    assert cli.model.hparams.T == cli.datamodule.T
    cli.trainer.callbacks += [
        callbacks.ModelSummary(max_depth=-1),
        callbacks.ModelCheckpoint(
            filename="best-{epoch}-{train_acc:.4f}-{val_acc:.4f}",
            save_top_k=1,
            monitor="val_acc",
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
