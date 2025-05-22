from typing import Callable, Dict, Optional, Tuple
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import PIL
import sys

sys.path.append("./src")

import lightning as L
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.utils import extract_archive
import spikingjelly.datasets as sjds
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as SJCIFAR10DVS

from utils.transforms import Cutout, TransformedDatasetWrapper

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31


def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_raw_events(
    fp, bytes_skip=0, bytes_trim=0, filter_dvs=False, times_first=False
):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def parse_raw_address(
    addr,
    x_mask=x_mask,
    x_shift=x_shift,
    y_mask=y_mask,
    y_shift=y_shift,
    polarity_mask=polarity_mask,
    polarity_shift=polarity_shift
):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_events(
    fp,
    filter_dvs=False,
    # bytes_skip=0,
    # bytes_trim=0,
    # times_first=False,
    **kwargs
):
    timestamp, addr = load_raw_events(
        fp,
        filter_dvs=filter_dvs,
        #   bytes_skip=bytes_skip,
        #   bytes_trim=bytes_trim,
        #   times_first=times_first
    )
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity


def move_data(root):
    class_num = (
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
        'ship', 'truck'
    )

    for cn in class_num:
        source = os.path.join(root, cn)

        target = os.path.join(root, 'test', cn)
        if not os.path.exists(target):
            os.makedirs(target)
            for i in range(100):
                os.symlink(
                    os.path.join(source, f'cifar10_{cn}_{i}.npz'),
                    os.path.join(target, f'cifar10_{cn}_{i}.npz')
                )

        target = os.path.join(root, 'train', cn)
        if not os.path.exists(target):
            os.makedirs(target)
            for i in range(100, 1000):
                os.symlink(
                    os.path.join(source, f'cifar10_{cn}_{i}.npz'),
                    os.path.join(target, f'cifar10_{cn}_{i}.npz')
                )


class CIFAR10DVS(sjds.NeuromorphicDatasetFolder):

    def __init__(
        self,
        root: str,
        train: bool = None,
        data_type: str = 'event',
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        '''
        :param root: root path of the dataset
        :type root: str
        :param data_type: `event` or `frame`
        :type data_type: str
        :param frames_number: the integrated frame number
        :type frames_number: int
        :param split_by: `time` or `number`
        :type split_by: str
        :param duration: the time duration of each frame
        :type duration: int
        :param transform: a function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        :type transform: callable
        :param target_transform: a function/transform that takes
            in the target and transforms it.
        :type target_transform: callable

        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``.

        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :class:`cal_fixed_frames_number_segment_index` for
            more details.

        If ``data_type == 'frame'``, ``frames_number`` is ``None``, and ``duration`` is not ``None``
            events will be integrated to frames with fixed time duration.

        '''
        super().__init__(
            root, train, data_type, frames_number, split_by, duration,
            transform, target_transform
        )

    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        return [
            (
                'airplane.zip',
                'https://ndownloader.figshare.com/files/7712788',
                '0afd5c4bf9ae06af762a77b180354fdd'
            ),
            (
                'automobile.zip',
                'https://ndownloader.figshare.com/files/7712791',
                '8438dfeba3bc970c94962d995b1b9bdd'
            ),
            (
                'bird.zip', 'https://ndownloader.figshare.com/files/7712794',
                'a9c207c91c55b9dc2002dc21c684d785'
            ),
            (
                'cat.zip', 'https://ndownloader.figshare.com/files/7712812',
                '52c63c677c2b15fa5146a8daf4d56687'
            ),
            (
                'deer.zip', 'https://ndownloader.figshare.com/files/7712815',
                'b6bf21f6c04d21ba4e23fc3e36c8a4a3'
            ),
            (
                'dog.zip', 'https://ndownloader.figshare.com/files/7712818',
                'f379ebdf6703d16e0a690782e62639c3'
            ),
            (
                'frog.zip', 'https://ndownloader.figshare.com/files/7712842',
                'cad6ed91214b1c7388a5f6ee56d08803'
            ),
            (
                'horse.zip', 'https://ndownloader.figshare.com/files/7712851',
                'e7cbbf77bec584ffbf913f00e682782a'
            ),
            (
                'ship.zip', 'https://ndownloader.figshare.com/files/7712836',
                '41c7bd7d6b251be82557c6cce9a7d5c9'
            ),
            (
                'truck.zip', 'https://ndownloader.figshare.com/files/7712839',
                '89f3922fd147d9aeff89e76a2b0b70a7'
            )
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return True

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        with ThreadPoolExecutor(
            max_workers=min(multiprocessing.cpu_count(), 10)
        ) as tpe:
            for zip_file in os.listdir(download_root):
                zip_file = os.path.join(download_root, zip_file)
                print(f'Extract [{zip_file}] to [{extract_root}].')
                tpe.submit(extract_archive, zip_file, extract_root)

    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''
        with open(file_name, 'rb') as fp:
            t, x, y, p = load_events(
                fp,
                x_mask=0xfE,
                x_shift=1,
                y_mask=0x7f00,
                y_shift=8,
                polarity_mask=1,
                polarity_shift=None
            )
            # return {'t': t, 'x': 127 - x, 'y': y, 'p': 1 - p.astype(int)}  # this will get the same data with http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/dat2mat.m
            # see https://github.com/jackd/events-tfds/pull/1 for more details about this problem
            return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128

    @staticmethod
    def read_aedat_save_to_np(bin_file: str, np_file: str):
        events = CIFAR10DVS.load_origin_data(bin_file)
        np.savez(
            np_file, t=events['t'], x=events['x'], y=events['y'], p=events['p']
        )
        print(f'Save [{bin_file}] to [{np_file}].')

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        t_ckp = time.time()
        with ThreadPoolExecutor(
            max_workers=min(multiprocessing.cpu_count(), 64)
        ) as tpe:
            for class_name in os.listdir(extract_root):
                aedat_dir = os.path.join(extract_root, class_name)
                np_dir = os.path.join(events_np_root, class_name)
                os.mkdir(np_dir)
                print(f'Mkdir [{np_dir}].')
                for bin_file in os.listdir(aedat_dir):
                    source_file = os.path.join(aedat_dir, bin_file)
                    target_file = os.path.join(
                        np_dir,
                        os.path.splitext(bin_file)[0] + '.npz'
                    )
                    print(
                        f'Start to convert [{source_file}] to [{target_file}].'
                    )
                    tpe.submit(
                        CIFAR10DVS.read_aedat_save_to_np, source_file,
                        target_file
                    )
        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')


class CIFAR10DVSNDA:

    def __init__(self, M=1, N=2):
        self.M = M
        self.N = N

    def __call__(self, data):
        c = 15 * self.N
        rotate_tf = transforms.RandomRotation(degrees=c)
        e = 8 * self.N
        cutout_tf = Cutout(n_holes=1, length=e)

        def roll(data, N=1):
            a = N*2 + 1
            off1 = np.random.randint(-a, a + 1)
            off2 = np.random.randint(-a, a + 1)
            return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        def rotate(data, N):
            return rotate_tf(data)

        def cutout(data, N):
            return cutout_tf(data)

        transforms_list = [roll, rotate, cutout]
        sampled_ops = np.random.choice(transforms_list, self.M)
        for op in sampled_ops:
            data = op(data, self.N)
        return data


class CIFAR10DVSDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_dir: str,
        T: int,
        batch_size: int = 128,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        frame_root = Path(
            self.data_dir
        ) / f"frames_number_{self.T}_split_by_number"
        if not frame_root.exists():
            # download and integrate frames
            ds = SJCIFAR10DVS(
                self.data_dir,
                data_type="frame",
                frames_number=self.T,
                split_by="number"
            )
            del ds

        train_set_root = frame_root / "train"
        test_set_root = frame_root / "test"
        if not train_set_root.exists() or not test_set_root.exists():
            # split the dataset
            move_data(
                Path(self.data_dir) / f"frames_number_{self.T}_split_by_number"
            )

    def setup(self, stage: str):
        train_set = CIFAR10DVS(
            self.data_dir,
            train=True,
            data_type='frame',
            frames_number=self.T,
            split_by='number'
        )
        test_set = CIFAR10DVS(
            self.data_dir,
            train=False,
            data_type='frame',
            frames_number=self.T,
            split_by='number'
        )

        function_nda = CIFAR10DVSNDA(M=1, N=2)

        def transform_train(data):
            data = transforms.RandomResizedCrop(
                128, scale=(0.7, 1.0), interpolation=PIL.Image.NEAREST
            )(data)
            resize = transforms.Resize(size=(48, 48))  # 48 48
            data = resize(data).float()
            flip = np.random.random() > 0.5
            if flip:
                data = torch.flip(data, dims=(3,))
            data = function_nda(data)
            return data.float()

        def transform_test(data):
            resize = transforms.Resize(size=(48, 48))  # 48 48
            data = resize(data).float()
            return data.float()

        self.train_set = TransformedDatasetWrapper(
            train_set, transform=transform_train
        )
        self.test_set = TransformedDatasetWrapper(
            test_set, transform=transform_test
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
