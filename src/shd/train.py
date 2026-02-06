import sys

sys.path.append("./src")
sys.path.append("./src/shd")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks

import models
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
from data_module import SHDDataModule


class SHDLightningModule(ClassificationLightningModule):
    def __init__(
        self,
        network: str,
        T: int,
        compress_x: bool,
        level: int,
        temporal_split_factor: int,
    ):
        super().__init__(
            num_classes=20,
            network=network,
            T=T,
            compress_x=compress_x,
            level=level,
            temporal_split_factor=temporal_split_factor,
        )

    def configure_network(self):
        network = self.hparams.network
        if not network.startswith("GC"):
            network = "GC" + network
        net_class = getattr(models, network)
        return net_class(
            T=self.hparams.T,
            compress_x=self.hparams.compress_x,
            level=self.hparams.level,
            temporal_split_factor=self.hparams.temporal_split_factor,
        )

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        learning_rate = 1e-2
        base_params, other_params = [], []
        for name, param in self.named_parameters():
            if name.endswith(".tau_m"):
                other_params.append(param)
            else:
                base_params.append(param)
        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": learning_rate},
                {"params": other_params, "lr": learning_rate * 2},
            ],
            lr=learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return ([optimizer], [scheduler])


def main():
    cli = LightningCLI(
        SHDLightningModule,
        SHDDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {"save_dir": "./logs", "name": "SHD"},
            },
            "enable_model_summary": False,
            "enable_checkpointing": False,
        },
    )
    assert cli.model.hparams.T * cli.datamodule.dt == 1000
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
