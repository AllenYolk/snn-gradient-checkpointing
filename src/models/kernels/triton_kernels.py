try:
    import triton
    import triton.language as tl
    import torch

    TRITON_AVAILABLE = True

    dc = torch.cuda.get_device_capability()
    if dc[0] < 8:
        print(
            "Triton kernel with bfloat16 is not supported on devices "
            "with compute capability < 8.0. "
            f"Your device's capability is: {dc}."
        )
        TRITON_BFLOAT16_AVAILABLE = False
    else:
        TRITON_BFLOAT16_AVAILABLE = True

    @triton.jit
    def _handwritten_lif_forward_triton_float32(
        x_seq_ptr,
        s_seq_ptr,
        h_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        ncl_indices = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL

        x_offsets_per_time_step = ncl_indices
        mask_x = ncl_indices < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0)
            v = h * (1.-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)
            tl.store(h_ptrs, h, mask=mask_x)

    @triton.jit
    def _handwritten_lif_forward_triton_float16(
        x_seq_ptr,
        s_seq_ptr,
        h_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.float16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.float16)
        one = tl.full([1], 1., dtype=tl.float16)

        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0).to(tl.float16)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)
            tl.store(h_ptrs, h, mask=mask_x)

    @triton.jit
    def _handwritten_lif_forward_triton_bfloat16(
        x_seq_ptr,
        s_seq_ptr,
        h_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.bfloat16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.bfloat16)
        one = tl.full([1], 1., dtype=tl.bfloat16)

        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0).to(tl.bfloat16)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)
            tl.store(h_ptrs, h, mask=mask_x)

    @triton.jit
    def _handwritten_lif_backward_not_detached_triton_float32(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        ncl_indices = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL

        x_offsets_per_time_step = ncl_indices
        mask_x = ncl_indices < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = tl.where(h >= 1., 1., 0)

            sg = pi * (h-1.)
            sg = 1. / (1. + sg*sg)
            grad_v = (grad_s - grad_v*h) * sg + grad_v * (1-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_lif_backward_not_detached_triton_float16(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.float16)
        pi = tl.full([1], pi, dtype=tl.float16)
        one = tl.full([1], 1., dtype=tl.float16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.float16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = tl.where(h >= 1., 1., 0).to(tl.float16)

            sg = pi * (h-one)
            sg = (one / (one + sg*sg)).to(tl.float16)
            grad_v = (grad_s - grad_v*h) * sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_lif_backward_not_detached_triton_bfloat16(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.bfloat16)
        pi = tl.full([1], pi, dtype=tl.bfloat16)
        one = tl.full([1], 1., dtype=tl.bfloat16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.bfloat16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = tl.where(h >= 1., 1., 0).to(tl.bfloat16)

            sg = pi * (h-one)
            sg = (one / (one + sg*sg)).to(tl.bfloat16)
            grad_v = (grad_s - grad_v*h) * sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_lif_backward_detached_triton_float32(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        ncl_indices = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL

        x_offsets_per_time_step = ncl_indices
        mask_x = ncl_indices < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = tl.where(h >= 1., 1., 0)

            sg = pi * (h-1.)
            sg = 1. / (1. + sg*sg)
            grad_v = grad_s*sg + grad_v * (1-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_lif_backward_detached_triton_float16(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.float16)
        pi = tl.full([1], pi, dtype=tl.float16)
        one = tl.full([1], 1., dtype=tl.float16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.float16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = tl.where(h >= 1., 1., 0).to(tl.float16)

            sg = pi * (h-one)
            sg = (one / (one + sg*sg)).to(tl.float16)
            grad_v = grad_s*sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_lif_backward_detached_triton_bfloat16(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.bfloat16)
        pi = tl.full([1], pi, dtype=tl.bfloat16)
        one = tl.full([1], 1., dtype=tl.bfloat16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.bfloat16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + t*T_stride + x_offsets_per_time_step
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = tl.where(h >= 1., 1., 0).to(tl.bfloat16)

            sg = pi * (h-one)
            sg = (one / (one + sg*sg)).to(tl.bfloat16)
            grad_v = grad_s*sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    _handwritten_lif_forward_triton_kernel = {
        torch.float32: _handwritten_lif_forward_triton_float32,
        torch.float16: _handwritten_lif_forward_triton_float16,
        torch.bfloat16: _handwritten_lif_forward_triton_bfloat16,
    }

    _handwritten_lif_backward_not_detached_triton_kernel = {
        torch.float32: _handwritten_lif_backward_not_detached_triton_float32,
        torch.float16: _handwritten_lif_backward_not_detached_triton_float16,
        torch.bfloat16: _handwritten_lif_backward_not_detached_triton_bfloat16,
    }

    _handwritten_lif_backward_detached_triton_kernel = {
        torch.float32: _handwritten_lif_backward_detached_triton_float32,
        torch.float16: _handwritten_lif_backward_detached_triton_float16,
        torch.bfloat16: _handwritten_lif_backward_detached_triton_bfloat16,
    }

    def handwritten_lif_forward_triton(x_seq, decay_lambda, block_ncl=512):
        T = x_seq.shape[0]
        NCL = x_seq[0].numel()
        grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)
        s_seq = torch.empty_like(x_seq)
        h_seq = torch.empty_like(x_seq)

        dtype = x_seq.dtype
        if dtype == torch.bfloat16 and not TRITON_BFLOAT16_AVAILABLE:
            raise RuntimeError(
                "Triton kernel with bfloat16 is not supported on devices "
                "with compute capability < 8.0. Use float16 instead."
            )
        kernel = _handwritten_lif_forward_triton_kernel[dtype]

        with torch.cuda.device(x_seq.device):
            kernel[grid](
                x_seq,
                s_seq,
                h_seq,
                T,
                NCL,
                x_seq.stride(0),
                decay_lambda,
                BLOCK_NCL=block_ncl
            )
        return s_seq, h_seq

    def handwritten_lif_backward_not_detached_triton(
        grad_s_seq, h_seq, decay_lambda, T, block_ncl=512
    ):
        NCL = grad_s_seq[0].numel()
        grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)
        grad_x_seq = torch.empty_like(grad_s_seq)

        dtype = grad_s_seq.dtype
        if dtype == torch.bfloat16 and not TRITON_BFLOAT16_AVAILABLE:
            raise RuntimeError(
                "Triton kernel with bfloat16 is not supported on devices "
                "with compute capability < 8.0. Use float16 instead."
            )
        kernel = _handwritten_lif_backward_not_detached_triton_kernel[dtype]

        with torch.cuda.device(grad_s_seq.device):
            kernel[grid](
                grad_s_seq,
                h_seq,
                grad_x_seq,
                T,
                NCL,
                grad_s_seq.stride(0),
                decay_lambda,
                torch.pi,
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

    def handwritten_lif_backward_detached_triton(
        grad_s_seq, h_seq, decay_lambda, T, block_ncl=512
    ):
        NCL = grad_s_seq[0].numel()
        grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)
        grad_x_seq = torch.empty_like(grad_s_seq)

        dtype = grad_s_seq.dtype
        if dtype == torch.bfloat16 and not TRITON_BFLOAT16_AVAILABLE:
            raise RuntimeError(
                "Triton kernel with bfloat16 is not supported on devices "
                "with compute capability < 8.0. Use float16 instead."
            )
        kernel = _handwritten_lif_backward_detached_triton_kernel[dtype]

        with torch.cuda.device(grad_s_seq.device):
            kernel[grid](
                grad_s_seq,
                h_seq,
                grad_x_seq,
                T,
                NCL,
                grad_s_seq.stride(0),
                decay_lambda,
                torch.pi,
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

except Exception as e:
    TRITON_AVAILABLE = False
    print(f"triton is not available. {e}")
