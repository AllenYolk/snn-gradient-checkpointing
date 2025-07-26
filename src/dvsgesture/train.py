import sys

sys.path.append("./src")
sys.path.append("./src/dvsgesture")

import torch
import torch.nn as nn
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks

import models
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
from data_module import DVSGestureDataModule


class DVSGestureLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        T: int,
        network: str,
        neuron_type: str,
        spike_compressor: str,
    ):
        super().__init__(
            num_classes=11,
            T=T,
            network=network,
            neuron_type=neuron_type,
            spike_compressor=spike_compressor,
        )

    def configure_network(self):
        return getattr(models, self.hparams.network)(
            neuron_type=self.hparams.neuron_type,
            T=self.hparams.T,
            spike_compressor=self.hparams.spike_compressor,
            decay_lambda=0.5,
            detach_reset=True,
            k=4,
        )

    def configure_criterion(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=64, gamma=0.1
        )
        return ([optimizer], [lr_scheduler])


def main():
    cli = LightningCLI(
        DVSGestureLightningModule,
        DVSGestureDataModule,
        run=False,
        trainer_defaults={
            "logger": {
                "class_path": "CSVLogger",
                "init_args": {
                    "save_dir": "./logs",
                    "name": "DVSGesture"
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
    assert cli.model.hparams.T == cli.datamodule.T
    if cli.trainer.is_global_zero:
        print(cli.model)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == '__main__':
    main()
