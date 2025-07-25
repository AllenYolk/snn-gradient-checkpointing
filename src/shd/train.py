import sys

sys.path.append("./src")
sys.path.append("./src/shd")

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
from data_module import SHDDataModule


class SHDLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        network: str,
        neuron_type: str,
        spike_compressor: str,
        lomo: bool = False,
    ):
        super().__init__(
            num_classes=20,
            network=network,
            neuron_type=neuron_type,
            spike_compressor=spike_compressor,
            lomo=lomo
        )

    def configure_network(self):
        return models.PLIFSFNN()

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        learning_rate = 1e-2
        base_params = [
            self.net.dense_2.dense.weight,
            self.net.dense_2.dense.bias,
            self.net.dense_1.dense.weight,
            self.net.dense_1.dense.bias,
        ]
        optimizer = torch.optim.Adam(
            [
                {
                    'params': base_params,
                    'lr': learning_rate
                },
                {
                    'params': self.net.dense_2.tau_m,
                    'lr': learning_rate * 2
                },
                {
                    'params': self.net.dense_1.tau_m,
                    'lr': learning_rate * 2
                },
            ],
            lr=learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=.5
        )
        return ([optimizer], [scheduler])


def main():
    cli = LightningCLI(
        SHDLightningModule,
        SHDDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {
                    "save_dir": "./logs",
                    "name": "SHD"
                }
            },
            "enable_model_summary": False,
            "enable_checkpointing": False,
        }
    )
    cli.trainer.callbacks += [
        callbacks.ModelSummary(max_depth=-1),
        callbacks.ModelCheckpoint(
            filename="best-{epoch}-{train_acc:.4f}-{valid_acc:.4f}",
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
