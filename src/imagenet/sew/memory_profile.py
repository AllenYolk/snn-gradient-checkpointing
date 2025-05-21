"""Modified from SEW ResNet source code.
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

import models
from data_module import ImageNetDataModule
from modules import ClassificationLightningModule
from modules.lightning_callbacks import *
from utils import TETLoss, TMeanCrossEntropyLoss, Lomo
from utils.profiler import *

PROFILE_LOG_DIR = "./profile_logs"


class SEWImageNetLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        num_classes: int,
        network: str,
        neuron_type: str,
        T: int,
        spike_compressor: str,
        learning_rate: float,
        momentum: float,
        loss: str,
        lomo: bool = False,
    ):
        super().__init__(
            num_classes=num_classes,
            network=network,
            neuron_type=neuron_type,
            T=T,
            spike_compressor=spike_compressor,
            learning_rate=learning_rate,
            momentum=momentum,
            loss=loss,
            lomo=lomo,
            y_with_T=True,
        )

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
        )
        if self.hparams.lomo:
            optimizer = Lomo(optimizer, scaler=self.trainer.scaler)
        self.profiled_optimizer = optimizer  # to access optimizer for profiling easily

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
        if self.hparams.loss == "ce":
            return TMeanCrossEntropyLoss()
        elif self.hparams.loss == "tet":
            return TETLoss(
                base_criterion=torch.nn.CrossEntropyLoss(), tet_lambda=0.
            )
        else:
            raise ValueError(f"`loss` should be either 'ce' or 'tet'")

    def configure_optimizers(self):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.profiled_optimizer, T_max=self.trainer.max_epochs
        )

        return ([self.profiled_optimizer], [lr_scheduler])


def main():
    cli = LightningCLI(
        SEWImageNetLightningModule,
        ImageNetDataModule,
        run=False,
        trainer_defaults={
            "max_steps": 5,
            "enable_progress_bar": False
        }
    )
    if cli.trainer.is_global_zero:
        print(cli.model)

    args = cli.config.model
    run_name = (
        f"{args.neuron_type}_{args.network}_{args.spike_compressor}_"
        f"lomo{args.lomo}_amp{cli.trainer.precision}_loss{args.loss}"
    )
    log_path = Path(PROFILE_LOG_DIR) / f"ImageNet-sew"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    mem_data_path = log_path / (run_name+".prof.pt")
    profile_log_path = log_path / (run_name+".prof.txt")

    net = cli.model.net
    optimizer = cli.model.profiled_optimizer
    profiler = MemoryProfilerList(
        CategoryMemoryProfiler(net, optimizer, filename=profile_log_path),
        LayerWiseMemoryProfiler(
            (
                net.pre_conv, net.layer1, net.layer2, net.layer3, net.layer4,
                net.avgpool, net.fc
            ),
            search_mode=(
                "self", "direct_children", "direct_children", "direct_children",
                "direct_children", "self", "self"
            ),
            model_names=(
                "pre_conv", "layer1", "layer2", "layer3", "layer4", "avgpool",
                "fc"
            ),
            instances=(torch.nn.Module,),
            filename=profile_log_path,
        ),
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    profiler.profile(sort_by="backward_peak_memory")
    profiler.save_data((None, mem_data_path))


if __name__ == '__main__':
    main()
