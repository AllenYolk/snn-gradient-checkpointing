import sys

sys.path.append("./src")

import torch
from nvidia import nvcomp

from models.compress import nvcomp_compress, nvcomp_decompress


def test_manual():
    torch.manual_seed(2025)
    x_f32 = torch.randn(1024, device='cuda', dtype=torch.float32)
    print(x_f32[:10])
    print("Original dtype:", x_f32.dtype)
    print("Original size (bytes):", x_f32.numel() * 4)
    print("Original data_ptr:", x_f32.data_ptr())
    print("Original shape:", x_f32.shape)
    print("Original size (elements):", x_f32.numel())
    print("Original storage offset:", x_f32.storage_offset())

    x_uint8 = torch.tensor((), device='cuda', dtype=torch.uint8).set_(
        x_f32.untyped_storage(), x_f32.storage_offset(), (x_f32.numel() * 4,)
    )
    print(x_uint8)
    print("Shared uint8 dtype:", x_uint8.dtype)
    print("Shared uint8 size:", x_uint8.numel())
    print("Shared uint8 data_ptr:", x_uint8.data_ptr())

    x_nv = nvcomp.as_array(x_uint8)
    print(x_nv.__cuda_array_interface__)
    codec = nvcomp.Codec(
        algorithm="Zstd", bitstream_kind=nvcomp.BitstreamKind.RAW
    )
    x_comp_nv = codec.encode(x_nv)
    print("Compressed nvcomp size (bytes):", x_comp_nv.buffer_size)

    x_dec_nv = codec.decode(x_comp_nv)
    print("Decompressed nvcomp size (bytes):", x_dec_nv.buffer_size)
    x_dec = torch.from_dlpack(x_dec_nv.to_dlpack())
    print("Decompressed tensor data_ptr:", x_dec.data_ptr())
    print("Decompressed tensor dtype:", x_dec.dtype)
    print("Decompressed tensor size (bytes):", x_dec.numel() * 4)
    print("Decompressed tensor shape:", x_dec.shape)
    print("Decompressed tensor size (elements):", x_dec.numel())
    print("Decompressed tensor storage offset:", x_dec.storage_offset())

    x_dec_f32 = torch.tensor((), dtype=torch.float32, device="cuda").set_(
        x_dec.untyped_storage(),
        x_dec.storage_offset(),
        (x_dec.numel() // 4,),
    )
    print("x_dec_f32 data_ptr:", x_dec_f32.data_ptr())
    print("x_dec_f32 dtype:", x_dec_f32.dtype)
    print("x_dec_f32 size (bytes):", x_dec_f32.numel() * 4)
    print("x_dec_f32 shape:", x_dec_f32.shape)
    print("x_dec_f32 size (elements):", x_dec_f32.numel())
    print("x_dec_f32 storage offset:", x_dec_f32.storage_offset())

    print("Restored equal to original?", torch.allclose(x_f32, x_dec_f32))


def test_function():
    torch.manual_seed(2025)
    x_f32 = torch.randn((31, 1024), device='cuda', dtype=torch.float32)
    print(x_f32[:10])
    print("Original dtype:", x_f32.dtype)
    print("Original size (bytes):", x_f32.numel() * 4)
    print("Original data_ptr:", x_f32.data_ptr())
    print("Original shape:", x_f32.shape)
    print("Original size (elements):", x_f32.numel())
    print("Original storage offset:", x_f32.storage_offset())

    x_comp = nvcomp_compress(x_f32, algorithm="Zstd")
    print("Compressed nvcomp size (bytes):", x_comp.buffer_size)

    x_reconstruct = nvcomp_decompress(x_comp, x_f32.shape, x_f32.dtype, "Zstd")

    print("Restored equal to original?", torch.allclose(x_f32, x_reconstruct))


if __name__ == "__main__":
    test_manual()
    test_function()
