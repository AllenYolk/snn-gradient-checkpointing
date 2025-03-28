import abc
import torch
from nvidia import nvcomp

from ..amp import AUTOCAST_DTYPE, is_autocast_enabled


def get_nvcomp_compressor(general_compressor: str):
    return globals()[general_compressor]()


NVCOMP_TYPE_DICT = {
    torch.float16: "<f2",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.int16: "<i2",
    torch.int8: "|i1",
    torch.uint64: "<u8",
    torch.uint32: "<u4",
    torch.uint16: "<u2",
    torch.uint8: "|u1",
}


def torch_type_to_nvcomp_type(dtype):
    return NVCOMP_TYPE_DICT.get(dtype, dtype)


class BaseNvcompCompressor(abc.ABC):

    def __init__(self):
        pass

    @property
    def supported_devices(self):
        return ["cpu", "cuda"]

    @property
    def supported_dtypes(self):
        return [
            torch.float32,
            torch.float16,
        ]

    @abc.abstractmethod
    def _compress(self, x: torch.Tensor) -> nvcomp.Array:
        pass

    @abc.abstractmethod
    def _decompress(self, x_nv: nvcomp.Array, shape, dtype) -> torch.Tensor:
        pass

    def compress(self, x: torch.Tensor) -> nvcomp.Array:
        with torch.no_grad():
            return self._compress(x)

    def decompress(self, x_nv: nvcomp.Array, shape, dtype) -> torch.Tensor:
        with torch.no_grad():
            return self._decompress(x_nv, shape, dtype)


class IdentityNvcompCompressor(BaseNvcompCompressor):

    def __init__(self):
        super().__init__()
        self.codec = nvcomp.Codec(
            algorithm="LZ4", bitstream_kind=nvcomp.BitstreamKind.RAW
        )

    def _compress(self, x: torch.Tensor) -> nvcomp.Array:
        x_nv = nvcomp.as_array(x)
        return x_nv

    def _decompress(self, x_nv: nvcomp.Array, shape, dtype) -> nvcomp.Array:
        x = torch.from_dlpack(x_nv.to_dlpack())
        return x.reshape(shape)


class Lz4NvcompCompressor(BaseNvcompCompressor):

    def __init__(self):
        super().__init__()
        self.codec = nvcomp.Codec(
            algorithm="LZ4", bitstream_kind=nvcomp.BitstreamKind.RAW
        )

    def _compress(self, x: torch.Tensor) -> nvcomp.Array:
        x_nv = nvcomp.as_array(x)
        x_nv_comp = self.codec.encode(x_nv)
        return x_nv_comp

    def _decompress(self, x_nv: nvcomp.Array, shape, dtype) -> nvcomp.Array:
        x_nv_decomp = self.codec.decode(
            x_nv, data_type=torch_type_to_nvcomp_type(dtype)
        )
        x_decomp = torch.from_dlpack(x_nv_decomp.to_dlpack())
        return x_decomp.reshape(shape)
