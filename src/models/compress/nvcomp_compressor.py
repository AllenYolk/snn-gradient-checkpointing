try:
    import abc
    import torch
    from nvidia import nvcomp
    from ..amp import AUTOCAST_DTYPE, is_autocast_enabled

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

    def nvcomp_compress(x: torch.Tensor, algorithm: str = "Zstd"):
        x = torch.tensor((), device=x.device, dtype=torch.uint8).set_(
            x.untyped_storage(),
            x.storage_offset(),
            x.numel() * x.element_size(),
        )
        x = nvcomp.as_array(x)

        return nvcomp.Codec(
            algorithm=algorithm,
            bitstream_kind=nvcomp.BitstreamKind.RAW,
        ).encode(x)

    def nvcomp_decompress(
        x: nvcomp.Array,
        target_shape,
        target_dtype=torch.float32,
        algorithm: str = "Zstd",
    ):
        x = nvcomp.Codec(
            algorithm=algorithm,
            bitstream_kind=nvcomp.BitstreamKind.RAW,
        ).decode(x)
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
