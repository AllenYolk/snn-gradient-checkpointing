import triton
import triton.language as tl
import torch


@triton.jit
def kernel(x_ptr, value, x_length, dtype: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x_ptrs = x_ptr + pid*BLOCK + tl.arange(0, BLOCK)
    mask = (pid*BLOCK + tl.arange(0, BLOCK)) < x_length

    xx = tl.full([BLOCK], value, dtype=dtype)
    one = tl.full([1], 11451, dtype=dtype)
    zero = tl.full([1], 0, dtype=dtype)
    xx = tl.where(xx >= 1., one, zero)
    tl.store(x_ptrs, xx, mask=mask)


def f(shape, value, dtype, tl_dtype):
    x = torch.empty(shape, dtype=dtype, device="cuda")
    x_length = x.numel()

    grid = lambda meta: (triton.cdiv(x_length, meta['BLOCK']),)
    with torch.cuda.device(x.device):
        kernel[grid](x, value, x_length, tl_dtype, BLOCK=64)

    return x


print(f([100, 100], 0.0, torch.float32, tl.float32))
print(f([100, 100], 1.0, torch.float16, tl.float16))
print(f([115, 91], 2.0, torch.int64, tl.int64))
