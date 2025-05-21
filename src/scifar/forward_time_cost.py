import sys

sys.path.append("./src")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from lightning.pytorch.cli import LightningCLI

from utils import Lomo
import models as models
from modules import ClassificationLightningModule
from modules.lightning_callbacks import *
from utils.profiler import *
from data_module import SCIFARDataModule

PROFILE_LOG_DIR = "./profile_logs"
WARMUP_ITERATIONS = 10


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
    cli = LightningCLI(
        SCIFARLightningModule,
        SCIFARDataModule,
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
    log_path = Path(PROFILE_LOG_DIR) / f"SCIFAR{cli.config.data.num_classes}"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    profile_log_path = log_path / (run_name+".time-prof.txt")

    net = cli.model.net
    profiler = LayerWiseFPCUDATimeProfiler(
        (net.conv, net.fc, net.decode),
        search_mode=("direct_children", "self", "self"),
        model_names=("conv", "fc", "decode"),
        instances=(torch.nn.Module,),
        filename=profile_log_path,
        warmup=WARMUP_ITERATIONS,
    )

    cli.trainer.validate(cli.model, datamodule=cli.datamodule)
    profiler.clear_hooks()
    profiler.profile()


if __name__ == '__main__':
    main()
