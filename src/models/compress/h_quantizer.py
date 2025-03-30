import abc
from typing import Tuple

import torch

from ..amp import AUTOCAST_DTYPE, is_autocast_enabled


class BaseHQuantizer(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def _quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        pass

    def quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._quantize(x_seq)

    def dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._dequantize(x_seq)


class IdentityHQuantizer(BaseHQuantizer):

    def __init__(self):
        super().__init__()

    def _quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq

    def _dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq


class ClampFloatHQuantizer(BaseHQuantizer):

    def __init__(
        self, clamp_range: Tuple[float, float], dtype=torch.float8_e4m3fn
    ):
        super().__init__()
        self.clamp_min = clamp_range[0]
        self.clamp_max = clamp_range[1]
        assert self.clamp_min < self.clamp_max

        self.dtype = dtype
        dtype_info = torch.finfo(dtype)
        self.dtype_min = dtype_info.min
        self.dtype_max = dtype_info.max
        print(dtype_info)

    def _quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.original_dtype = x_seq.dtype
        x_seq = torch.clamp(x_seq, self.clamp_min, self.clamp_max)
        # proj: [clamp_min, clamp_max] -> [dtype_min, dtype_max]
        x_seq = (x_seq - self.clamp_min) / (self.clamp_max - self.clamp_min)
        x_seq = x_seq * (self.dtype_max - self.dtype_min) + self.dtype_min
        # cast to dtype
        return x_seq.to(dtype=self.dtype)

    def _dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_seq = x_seq.to(dtype=self.original_dtype)
        # proj: [dtype_min, dtype_max] -> [clamp_min, clamp_max]
        x_seq = (x_seq - self.dtype_min) / (self.dtype_max - self.dtype_min)
        x_seq = x_seq * (self.clamp_max - self.clamp_min) + self.clamp_min
        return x_seq


class ClampIntHQuantizer(BaseHQuantizer):

    def __init__(self, clamp_range: Tuple[float, float], dtype=torch.uint8):
        super().__init__()
        self.clamp_min = clamp_range[0]
        self.clamp_max = clamp_range[1]
        assert self.clamp_min < self.clamp_max

        self.dtype = dtype
        dtype_info = torch.iinfo(dtype)
        self.dtype_min = dtype_info.min
        self.dtype_max = dtype_info.max
        print(dtype_info)

    def _quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.original_dtype = x_seq.dtype
        x_seq = torch.clamp(x_seq, self.clamp_min, self.clamp_max)
        # proj: [clamp_min, clamp_max] -> [dtype_min, dtype_max]
        x_seq = (x_seq - self.clamp_min) / (self.clamp_max - self.clamp_min)
        x_seq = x_seq * (self.dtype_max - self.dtype_min) + self.dtype_min
        # cast to dtype
        return x_seq.to(dtype=self.dtype)

    def _dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        x_seq = x_seq.to(dtype=self.original_dtype)
        # proj: [dtype_min, dtype_max] -> [clamp_min, clamp_max]
        x_seq = (x_seq - self.dtype_min) / (self.dtype_max - self.dtype_min)
        x_seq = x_seq * (self.clamp_max - self.clamp_min) + self.clamp_min
        return x_seq
