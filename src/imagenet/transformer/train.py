"""Modified from Spikformer and QKFormer source code.
"""
import sys

sys.path.append("./src")
sys.path.append("./src/imagenet")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import callbacks
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler_v2
from timm.optim import create_optimizer_v2

import models
from data_module import ImageNetDataModule
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
from utils import Lomo


class TransformerImageNetLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        network: str,
        neuron_type: str,
        T: int,
        spike_compressor: str,
        learning_rate: float,
        l2_factor: float,
        smoothing: float,
        lomo: bool = False,
    ):
        super().__init__(
            num_classes=1000,
            network=network,
            neuron_type=neuron_type,
            T=T,
            spike_compressor=spike_compressor,
            learning_rate=learning_rate,
            l2_factor=l2_factor,
            smoothing=smoothing,
            lomo=lomo,
        )
        # this property should be properly assigned before `configure_optimizers`
        self.batch_per_training_epoch = None

    def configure_network(self):
        return getattr(models, self.hparams.network)(
            neuron_type=self.hparams.neuron_type,
            T=self.hparams.T,
            spike_compressor=self.hparams.spike_compressor,
            decay_lambda=0.5,
            detach_reset=True,
            k=4,  # for SlidingPSN
        )

    def configure_criterion(self):
        if self.hparams.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(
                smoothing=self.hparams.smoothing
            )
        else:
            criterion = torch.nn.CrossEntropyLoss()
        print(f"Criterion: {criterion}")
        return criterion

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt="adamw",
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_factor,
        )  # timm's optimizers are inherited from torch's optimizer
        if self.hparams.lomo:
            optimizer = Lomo(optimizer, scaler=self.trainer.scaler)

        if self.batch_per_training_epoch is None:
            raise ValueError(
                "`TransformerImageNetLightningModule.batch_per_training_batch`"
                f"should be assigned the length of `train_dataloader`. "
                f"We suggest making this assignment by overriding "
                f"`LightningCLI.instantiate_classes`."
            )
        else:
            print(
                f"{self.batch_per_training_epoch} batches per training epoch."
            )
        lr_scheduler, total_epochs = create_scheduler_v2(
            optimizer,
            sched="cosine",
            num_epochs=self.trainer.max_epochs,
            min_lr=1e-5,
            warmup_epochs=20,
            warmup_lr=1e-6,
            cooldown_epochs=10,
            step_on_epochs=False,
            updates_per_epoch=self.batch_per_training_epoch
        )
        if self.trainer.max_epochs != total_epochs:
            print(
                f"Number of epochs changed from {self.trainer.max_epochs} "
                f"to {total_epochs} due to the scheduler."
            )
            self.trainer.fit_loop.max_epochs = total_epochs

        return ([optimizer], [{
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }])

    def lr_scheduler_step(self, scheduler, metric):
        """timm's scheduler is not a subclass of torch's scheduler.
        So, we have to rewrite LightningModule.lr_scheduler_step()
        """
        scheduler.step_update(self.trainer.global_step)


class CustomLightningCLI(LightningCLI):

    def instantiate_classes(self) -> None:
        """Assign LightningModule.batch_per_training_epoch 
        once LightningModule is instantiated.
        """
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")
        self.model = self._get(self.config_init, "model")
        # tell LightningModule what's the length of train_dataloader
        self.model.batch_per_training_epoch = self.datamodule.batch_per_training_epoch
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()


def main():
    cli = CustomLightningCLI(
        TransformerImageNetLightningModule, ImageNetDataModule, run=False
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
