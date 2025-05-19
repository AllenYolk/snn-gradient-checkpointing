import lightning as L
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy
import torch.nn as nn


class ClassificationLightningModule(L.LightningModule):

    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        kwargs.update({
            "num_classes": num_classes,
        })
        self.save_hyperparameters(kwargs)

        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.net = self.configure_network()
        self.criterion = self.configure_criterion()

    def configure_network(self) -> nn.Module:
        raise NotImplementedError(
            "ClassificationLightningModule.get_network() is not implemented."
        )

    def configure_criterion(self) -> nn.Module:
        raise NotImplementedError(
            "ClassificationLightningModule.get_criterion() is not implemented."
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            "ClassificationLightningModule.configure_optimizers() is not "
            "implemented."
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, label = batch[0].float(), batch[1]
        y = self(x)
        batch_loss = self.criterion(y, label)
        if y.numel() == label.numel():  # mixup!
            label = label.argmax(dim=1)
        self.train_acc.update(y, label)
        self.train_loss.update(batch_loss.data)
        self.log("train_loss", self.train_loss.compute(), prog_bar=True)
        self.log("train_acc", self.train_acc.compute() * 100, prog_bar=True)
        return batch_loss

    def on_train_epoch_end(self):
        train_acc = self.train_acc.compute()
        train_loss = self.train_loss.compute()
        self.log("train_loss", train_loss, on_epoch=True)
        self.log("train_acc", train_acc, on_epoch=True)
        self.train_acc.reset()
        self.train_loss.reset()
        if self.global_rank == 0:
            print(
                f"Epoch {self.current_epoch}/{self.trainer.max_epochs}: "
                f"train_loss={train_loss:.3f}, "
                f"train_acc={train_acc*100:.3f}, "
            )

    def validation_step(self, batch, batch_idx):
        x, label = batch
        y = self(x)
        batch_loss = self.criterion(y, label)
        if y.numel() == label.numel():  # mixup!
            label = label.argmax(dim=1)
        self.val_acc.update(y, label)
        self.val_loss.update(batch_loss.data)
        return batch_loss

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_loss = self.val_loss.compute()
        self.log("val_acc", val_acc, on_epoch=True)
        self.log("val_loss", val_loss, on_epoch=True)
        self.val_acc.reset()
        self.val_loss.reset()
        if self.global_rank == 0:
            print(f"val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, ")
