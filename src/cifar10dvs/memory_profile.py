import sys

sys.path.append("./src")
sys.path.append("./src/cifar10dvs")

import torch
from utils import use_torch_npu

npu_available = use_torch_npu()

from lightning.pytorch.cli import LightningCLI

from utils import Lomo, TMeanCrossEntropyLoss, TETLoss
import models
from modules import ClassificationLightningModule
from utils.lightning_callbacks import *
from data_module import CIFAR10DVSDataModule
from utils.profiler import *


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
        return models.GCCIFAR10DVSVGG(
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

    def training_step(self, batch, batch_idx):
        x, label = batch[0].float(), batch[1]
        x.requires_grad = True  # so that the BP peak memory of the 1st layer can be properly recorded
        y = self(x)
        batch_loss = self.criterion(y, label)  # must properly handle the sizes!
        if self.y_with_T:
            y = y.mean(dim=0)
        if label.ndim > 1:
            label = label.argmax(dim=1)
        self.train_acc.update(y, label)
        self.train_loss.update(batch_loss.data)
        self.log("train_loss", self.train_loss.compute(), prog_bar=True)
        self.log("train_acc", self.train_acc.compute() * 100, prog_bar=True)
        return batch_loss


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
            "max_steps": 3,
            "enable_progress_bar": False,
        }
    )
    if cli.trainer.is_global_zero:
        print(cli.model)

    args = cli.config.model
    run_name = f"{args.neuron_type}_{args.level}"
    if not args.compress_x:
        run_name += "_no-compress"
    log_path = Path("profile_logs") / f"CIFAR10DVS"
    if not log_path.exists():
        log_path.mkdir(parents=True)
    mem_data_path = log_path / (run_name+".prof.pt")
    profile_log_path = log_path / (run_name+".prof.txt")

    net = cli.model.net
    profiler = LayerWiseMemoryProfiler(
        (net.features, net.classifier),
        model_names=("feature_extractor", "classifier"),
        search_mode=("direct_children", "direct_children"),
        instances=(torch.nn.Module,),
        log_path=profile_log_path,
        data_path=mem_data_path,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    profiler.export(output=True)


if __name__ == '__main__':
    main()
