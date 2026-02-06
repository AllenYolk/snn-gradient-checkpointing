from pathlib import Path
import sys

sys.path.append("./src")

import lightning as L
from torch.utils.data.dataloader import DataLoader
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture


class DVSGestureDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str, T: int, batch_size: int = 128, num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        DVS128Gesture(
            root=self.data_dir,
            train=True,
            data_type="frame",
            frames_number=self.T,
            split_by="number",
        )
        DVS128Gesture(
            root=self.data_dir,
            train=False,
            data_type="frame",
            frames_number=self.T,
            split_by="number",
        )

    def setup(self, stage: str):
        self.train_set = DVS128Gesture(
            root=self.data_dir,
            train=True,
            data_type="frame",
            frames_number=self.T,
            split_by="number",
        )
        self.test_set = DVS128Gesture(
            root=self.data_dir,
            train=False,
            data_type="frame",
            frames_number=self.T,
            split_by="number",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
