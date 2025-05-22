import sys

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()
if npu_available:
    print("NPU is available.")
else:
    print("NPU is not available.")

from lightning.pytorch.cli import LightningCLI

from utils import Lomo, TMeanCrossEntropyLoss, TETLoss
import models
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
from data_module import CIFAR10DVSDataModule
from utils.profiler import *

PROFILE_LOG_DIR = "./profile_logs"


class CIFAR10DVSLightningModule(ClassificationLightningModule):

    def __init__(
        self,
        network: str,
        T: int,
        neuron_type: str,
        spike_compressor: str,
        decay_lambda: float,
        learning_rate: float,
        momentum: float,
        l2_factor: float,
        lomo: bool = False,
        loss: str = "tet",
    ):
        super().__init__(
            num_classes=10,
            network=network,
            T=T,
            neuron_type=neuron_type,
            spike_compressor=spike_compressor,
            decay_lambda=decay_lambda,
            learning_rate=learning_rate,
            momentum=momentum,
            l2_factor=l2_factor,
            lomo=lomo,
            loss=loss,
            y_with_T=True,
        )

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.l2_factor
        )
        if self.hparams.lomo:
            optimizer = Lomo(optimizer, scaler=self.trainer.scaler)
        self.profiled_optimizer = optimizer  # to access optimizer for profiling easily

    def configure_network(self):
        return getattr(models, self.hparams.network)(
            T=self.hparams.T,  # for PSN and tebn
            neuron_type=self.hparams.neuron_type,
            spike_compressor=self.hparams.spike_compressor,
            decay_lambda=self.hparams.decay_lambda,
            k=4,  # for SlidingPSN
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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.profiled_optimizer, T_max=self.trainer.max_epochs
        )
        return ([self.profiled_optimizer], [lr_scheduler])


def main():
    cli = LightningCLI(
        CIFAR10DVSLightningModule,
        CIFAR10DVSDataModule,
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
        f"T{args.T}_lomo{args.lomo}_amp{cli.trainer.precision}_loss{args.loss}"
    )
    log_path = Path(PROFILE_LOG_DIR) / f"CIFAR10DVS"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    mem_data_path = log_path / (run_name+".prof.pt")
    profile_log_path = log_path / (run_name+".prof.txt")

    net = cli.model.net
    optimizer = cli.model.profiled_optimizer
    profiler = MemoryProfilerList(
        CategoryMemoryProfiler(net, optimizer, filename=profile_log_path),
        LayerWiseMemoryProfiler(
            (net.features, net.dropout, net.classifier),
            search_mode=("direct_children", "self", "self"),
            model_names=("feature_extractor", "dropout", "classifier"),
            instances=(torch.nn.Module,),
            filename=profile_log_path,
        ),
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    profiler.profile(sort_by="backward_peak_memory")
    profiler.save_data((None, mem_data_path))


if __name__ == '__main__':
    main()
