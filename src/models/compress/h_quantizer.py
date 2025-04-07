import abc

import torch

from ..amp import AUTOCAST_DTYPE, is_autocast_enabled

MIN_POS_FLOAT = {
    torch.float16: 2**(-24),
}
TORCH_FLOAT8E4M3FN_AVAILABLE = hasattr(torch, "float8_e4m3fn")
if TORCH_FLOAT8E4M3FN_AVAILABLE:
    MIN_POS_FLOAT.update({
        torch.float8_e4m3fn: 2**(-9),
        torch.float8_e5m2: 2**(-16),
    })
    DEFAULT_HQ_DTYPE = torch.float8_e4m3fn
else:
    DEFAULT_HQ_DTYPE = torch.float16
print("DEFAULT_HQ_DTYPE:", DEFAULT_HQ_DTYPE)


def get_h_quantizer(h_quantizer: str, **kwargs):
    return globals()[h_quantizer](**kwargs)


class BaseHQuantizer(abc.ABC):

    def __init__(self, dtype=DEFAULT_HQ_DTYPE):
        self.dtype = dtype

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


class NullHQuantizer(BaseHQuantizer):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dtype = AUTOCAST_DTYPE if is_autocast_enabled() else torch.float32

    def _quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq

    def _dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq


class TypecastHQuantizer(BaseHQuantizer):

    def __init__(self, dtype=DEFAULT_HQ_DTYPE, *args, **kwargs):
        super().__init__(dtype=dtype)

    def _quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.original_dtype = x_seq.dtype
        return x_seq.to(dtype=self.dtype)

    def _dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        return x_seq.to(dtype=self.original_dtype)


class ClampProjHQuantizer(BaseHQuantizer):

    def __init__(
        self,
        clamp_abs: float = 12.223,  # searched optimal value
        dtype=DEFAULT_HQ_DTYPE,
        verbose=False
    ):
        super().__init__(dtype=dtype)
        self.clamp_abs = clamp_abs
        assert self.clamp_abs > 0

        if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
            dtype_info = torch.finfo(dtype)
        else:
            raise TypeError(
                f"Unsupported h-quantization dtype {dtype}. "
                f"Only floating point types are supported."
            )
        dtype_min, dtype_max = dtype_info.min, dtype_info.max
        assert dtype_min + dtype_max == 0
        self.dtype_abs = dtype_max

        self.shift = (
            self.clamp_abs / (2 * self.dtype_abs) * MIN_POS_FLOAT[dtype]
        )
        self.scale = self.dtype_abs / self.clamp_abs

        if verbose:
            print(dtype_info)
            print(f"clamp_abs: {self.clamp_abs}")
            print(f"dtype_abs: {self.dtype_abs}")
            print(f"shift: {self.shift}")

    def _quantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        self.original_dtype = x_seq.dtype
        # shift
        x_seq = x_seq - self.shift
        # clamp
        x_seq = torch.clamp(x_seq, -self.clamp_abs, self.clamp_abs)
        # project (use simplified formulation to avoid numerical issues)
        # x_seq = (x_seq + self.clamp_abs) / (2 * self.clamp_abs)
        # x_seq = x_seq * 2 * self.dtype_abs - self.dtype_abs
        x_seq = x_seq * self.scale
        # cast to dtype
        return x_seq.to(dtype=self.dtype)

    def _dequantize(self, x_seq: torch.Tensor) -> torch.Tensor:
        # cast to original dtype
        x_seq = x_seq.to(dtype=self.original_dtype)
        # inverse project (use simplified formulation to avoid numerical issues)
        # x_seq = (x_seq + self.dtype_abs) / (2 * self.dtype_abs)
        # x_seq = x_seq * (2 * self.clamp_abs) - self.clamp_abs
        x_seq = x_seq / self.scale
        # inverse shift
        x_seq = x_seq + self.shift
        return x_seq

    def __repr__(self):
        return (
            f"ClampProjHQuantizer(clamp_abs={self.clamp_abs}, "
            f"dtype={self.dtype})"
        )

    def __str__(self):
        return self.__repr__()
