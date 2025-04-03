try:
    from typing import Sequence, Union
    import torch
    from nvidia import nvcomp

    NVCOMP_AVAILABLE = True

    NVCOMP_TYPE_DICT = {
        torch.float16: "<f2",
        torch.int64: "<i8",
        torch.int32: "<i4",
        torch.int16: "<i2",
        torch.int8: "|i1",
        torch.uint8: "|u1",
    }

    def torch_type_to_nvcomp_type(dtype):
        return NVCOMP_TYPE_DICT.get(dtype, dtype)

    DEFAULT_NVCOMP_CODEC_ALGORITHM = "Zstd"
    DEFAULT_NVCOMP_CODEC_KWARGS = {}

    NVCOMP_CHUNK_SIZE_BYTES = 16777216
    NVCOMP_NEED_CHUNK = {
        "Zstd": True,
        "ANS": True,
    }

    # ===================== functional interface ======================
    def nvcomp_compress(
        x: torch.Tensor,
        algorithm: str = DEFAULT_NVCOMP_CODEC_ALGORITHM,
        **kwargs
    ):
        x_size_byte = x.numel() * x.element_size()
        x = torch.tensor((), device=x.device, dtype=torch.uint8).set_(
            x.untyped_storage(),
            x.storage_offset(),
            (x_size_byte,),
        )

        if (
            x_size_byte > NVCOMP_CHUNK_SIZE_BYTES and
            NVCOMP_NEED_CHUNK[algorithm]
        ):
            x = torch.split(x, NVCOMP_CHUNK_SIZE_BYTES)
            x = [nvcomp.as_array(chunk) for chunk in x]
        else:
            x = nvcomp.as_array(x)

        additional_kwargs = DEFAULT_NVCOMP_CODEC_KWARGS
        additional_kwargs.update(kwargs)

        return nvcomp.Codec(
            algorithm=algorithm,
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            **additional_kwargs,
        ).encode(x)

    def nvcomp_decompress(
        x: Union[nvcomp.Array, Sequence[nvcomp.Array]],
        target_shape,
        target_dtype=torch.float32,
        algorithm: str = DEFAULT_NVCOMP_CODEC_ALGORITHM,
        **kwargs
    ):
        additional_kwargs = DEFAULT_NVCOMP_CODEC_KWARGS
        additional_kwargs.update(kwargs)
        x = nvcomp.Codec(
            algorithm=algorithm,
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            **additional_kwargs,
        ).decode(x)

        if isinstance(x, nvcomp.Array):
            x = torch.from_dlpack(x.to_dlpack())
        else:
            x = [torch.from_dlpack(chunk.to_dlpack()) for chunk in x]
            x = torch.cat(x, dim=0)

        y = torch.tensor((), dtype=target_dtype, device=x.device)
        y = y.set_(
            x.untyped_storage(),
            x.storage_offset(),
            (x.numel() * x.element_size() // y.element_size(),),
        )
        return y.reshape(target_shape)

    # ===================== class interface ======================
    class NvcompCompressor:

        def __init__(
            self,
            algorithm: str = DEFAULT_NVCOMP_CODEC_ALGORITHM,
            compressed_dtype=torch.uint8,
            **kwargs  # additional arguments for nvcomp.Codec
        ):
            self.algorithm = algorithm
            self.compressed_dtype = compressed_dtype
            additional_kwargs = DEFAULT_NVCOMP_CODEC_KWARGS
            additional_kwargs.update(kwargs)
            self.codec = nvcomp.Codec(
                algorithm=algorithm,
                bitstream_kind=nvcomp.BitstreamKind.RAW,
                **additional_kwargs,
            )

        def compress(self, x: torch.Tensor) -> nvcomp.Array:
            y = torch.tensor((), device=x.device, dtype=self.compressed_dtype)
            y = y.set_(
                x.untyped_storage(),
                x.storage_offset(),
                (x.numel() * x.element_size() // y.element_size(),),
            )
            y = nvcomp.as_array(y)

            return self.codec.encode(y)

        def decompress(self, x: nvcomp.Array, target_shape, target_dtype):
            x = self.codec.decode(x)
            x = torch.from_dlpack(x.to_dlpack())
            y = torch.tensor((), dtype=target_dtype, device=x.device)
            y = y.set_(
                x.untyped_storage(),
                x.storage_offset(),
                (x.numel() * x.element_size() // y.element_size(),),
            )
            return y.reshape(target_shape)

except Exception:
    NVCOMP_AVAILABLE = False
