"""Modified from Spikformer and QKFormer source code.
"""
from pathlib import Path
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
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler_v2
from timm.optim import create_optimizer_v2

import models
from data_module import ImageNetDataModule
from modules import ClassificationLightningModule
from modules.lightning_callbacks import *
from utils import Lomo
from utils.profiler import *

PROFILE_LOG_DIR = "./profile_logs"
WARMUP_ITERATIONS = 10


class TransformerImageNetLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        num_classes: int,
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
            num_classes=num_classes,
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
        TransformerImageNetLightningModule,
        ImageNetDataModule,
        run=False,
        trainer_defaults={"enable_progress_bar": False}
    )
    if cli.trainer.is_global_zero:
        print(cli.model)

    args = cli.config.model
    run_name = (
        f"{args.neuron_type}_{args.network}_{args.spike_compressor}_"
        f"lomo{args.lomo}_amp{cli.trainer.precision}"
    )
    log_path = Path(PROFILE_LOG_DIR) / f"ImageNet-transformer"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    profile_log_path = log_path / (run_name+".time-prof.txt")

    net = cli.model.net
    if args.network.endswith("Spikformer"):
        model_list = (net.patch_embed, *[b for b in net.block], net.head)
        search_mode_list = (
            "direct_children", *["direct_children" for _ in net.block], "self"
        )
        model_name_list = (
            "patch_embed", *[f"block{i}" for i in range(net.depths)], "head"
        )
    elif args.network.endswith("QKFormer"):
        model_list = (
            net.patch_embed1, *[b for b in net.block1], net.patch_embed2,
            *[b for b in net.block2], net.patch_embed3,
            *[b for b in net.block3], net.head
        )
        search_mode_list = (
            *["direct_children" for _ in range(3 + net.depths)], "self"
        )
        model_name_list = (
            "patch_embed1",
            *[f"block1_{i}" for i in range(1)],
            "patch_embed2",
            *[f"block2_{i}" for i in range(2)],
            "patch_embed3",
            *[f"block3_{i}" for i in range(net.depths - 3)],
            "head",
        )
    profiler = LayerWiseFPCUDATimeProfiler(
        model_list,
        search_mode=search_mode_list,
        model_names=model_name_list,
        instances=(torch.nn.Module,),
        filename=profile_log_path,
        warmup=WARMUP_ITERATIONS,
    )

    cli.trainer.validate(cli.model, datamodule=cli.datamodule)
    profiler.clear_hooks()
    profiler.profile()


if __name__ == '__main__':
    main()
