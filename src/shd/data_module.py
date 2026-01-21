from pathlib import Path

import lightning as L
from tqdm import tqdm
import torch
from torch.utils import data
import h5py
import numpy as np


def binary_image_readout(times, units, dt=1e-3):
    img = []
    N = int(1 / dt)
    for i in range(N):
        idxs = np.argwhere(times <= i * dt).flatten()
        vals = units[idxs]
        vals = vals[vals > 0]
        vector = np.zeros(700)
        vector[700 - vals] = 1
        times = np.delete(times, idxs)
        units = np.delete(units, idxs)
        img.append(vector)
    return np.array(img)


def generate_dataset(file_path, output_dir, dt=1e-3):
    print("generating SHD dataset at: ", file_path)
    with h5py.File(file_path, 'r') as fileh:
        units = fileh["spikes"]["units"]
        times = fileh["spikes"]["times"]
        labels = fileh["labels"]

        print("Number of samples: ", len(times))
        for i in tqdm(range(len(times))):
            x_tmp = binary_image_readout(times[i], units[i], dt=dt)
            y_tmp = labels[i]
            output_file_name = f"ID:{i}_{y_tmp}.npz"
            output_file_name = Path(output_dir) / output_file_name
            np.savez_compressed(output_file_name, x=x_tmp)
        print("Done!")


class MyDataset(data.Dataset):

    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        p = str(self.data_paths[index])
        x = torch.from_numpy(np.load(p)["x"]).to(torch.float32)
        y_ = p.split('_')[-1]
        y_ = int(y_.split('.')[0])
        y = torch.tensor(int(y_))
        if self.transform:
            x = self.transform(x)
        return x, y


class SHDDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        dt: int,  # unit: ms
        batch_size: int = 128,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dt = dt

        self.train_h5 = self.data_dir / "shd_train.h5"
        self.test_h5 = self.data_dir / "shd_test.h5"
        self.train_dir = self.data_dir / f"train_{dt}ms"
        self.test_dir = self.data_dir / f"test_{dt}ms"

    def prepare_data(self):
        if not self.train_dir.exists():
            self.train_dir.mkdir(parents=True)
            generate_dataset(self.train_h5, self.train_dir, dt=self.dt / 1000)
        if not self.test_dir.exists():
            self.test_dir.mkdir(parents=True)
            generate_dataset(self.test_h5, self.test_dir, dt=self.dt / 1000)

    def setup(self, stage: str):
        train_files = list(self.train_dir.glob("*.npz"))
        test_files = list(self.test_dir.glob("*.npz"))
        self.train_set = MyDataset(train_files)
        self.test_set = MyDataset(test_files)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
