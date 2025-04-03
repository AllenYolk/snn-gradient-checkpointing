try:
    import triton
    import triton.language as tl
    import torch

    from ..compress import ClampProjHQuantizer, FLOAT8_AVAILABLE

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
    if float(f"{dc[0]}.{dc[1]}") < 8.9:
        print(
            "Triton kernel with float8e4b8 is not supported on devices "
            "with compute capability < 8.9. "
            f"Your devices's capability is: {dc}."
        )
        TRITON_BFLOAT8E4B8_AVAILABLE = False
    else:
        TRITON_BFLOAT8E4B8_AVAILABLE = True

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

    # ========== Hand-written Multistep LIF neuron with H quantization =========
    @triton.jit
    def _handwritten_hqlif_forward_triton_float32_float16(
        x_seq_ptr,
        s_seq_ptr,
        ot_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        clamp_abs,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0)
            v = h * (1.-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)

            ot = h - 1.
            ot = tl.clamp(ot - shift, -clamp_abs, clamp_abs) * scale
            ot = tl.cast(ot, tl.float16)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(ot_ptrs, ot, mask=mask_x)

    @triton.jit
    def _handwritten_hqlif_forward_triton_float32_float8e4m3fn(
        x_seq_ptr,
        s_seq_ptr,
        ot_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        clamp_abs,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0)
            v = h * (1.-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)

            # quantize ot
            ot = h - 1.
            ot = tl.clamp(ot - shift, -clamp_abs, clamp_abs) * scale
            ot = tl.cast(ot, tl.float8e4b8)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(ot_ptrs, ot, mask=mask_x)

    @triton.jit
    def _handwritten_hqlif_forward_triton_float16_float16(
        x_seq_ptr,
        s_seq_ptr,
        ot_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        clamp_abs,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.float16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.float16)
        one = tl.full([1], 1., dtype=tl.float16)
        scale = tl.full([1], scale, dtype=tl.float16)
        shift = tl.full([1], shift, dtype=tl.float16)
        clamp_abs = tl.full([1], clamp_abs, dtype=tl.float16)
        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0).to(tl.float16)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)

            ot = h - one
            ot = tl.clamp(ot - shift, -clamp_abs, clamp_abs) * scale
            ot = tl.cast(ot, tl.float16)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(ot_ptrs, ot, mask=mask_x)

    @triton.jit
    def _handwritten_hqlif_forward_triton_float16_float8e4m3fn(
        x_seq_ptr,
        s_seq_ptr,
        ot_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        clamp_abs,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.float16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.float16)
        one = tl.full([1], 1., dtype=tl.float16)
        scale = tl.full([1], scale, dtype=tl.float16)
        shift = tl.full([1], shift, dtype=tl.float16)
        clamp_abs = tl.full([1], clamp_abs, dtype=tl.float16)
        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0).to(tl.float16)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)

            # quantize ot
            ot = h - one
            ot = tl.clamp(ot - shift, -clamp_abs, clamp_abs) * scale
            ot = tl.cast(ot, tl.float8e4b8)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(ot_ptrs, ot, mask=mask_x)

    @triton.jit
    def _handwritten_hqlif_forward_triton_bfloat16_float16(
        x_seq_ptr,
        s_seq_ptr,
        ot_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        clamp_abs,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.bfloat16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.bfloat16)
        one = tl.full([1], 1., dtype=tl.bfloat16)
        scale = tl.full([1], scale, dtype=tl.bfloat16)
        shift = tl.full([1], shift, dtype=tl.bfloat16)
        clamp_abs = tl.full([1], clamp_abs, dtype=tl.bfloat16)
        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0).to(tl.bfloat16)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)

            ot = h - one
            ot = tl.clamp(ot - shift, -clamp_abs, clamp_abs) * scale
            ot = tl.cast(ot, tl.float16)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(ot_ptrs, ot, mask=mask_x)

    @triton.jit
    def _handwritten_hqlif_forward_triton_bfloat16_float8e4m3fn(
        x_seq_ptr,
        s_seq_ptr,
        ot_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        clamp_abs,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=tl.bfloat16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.bfloat16)
        one = tl.full([1], 1., dtype=tl.bfloat16)
        scale = tl.full([1], scale, dtype=tl.bfloat16)
        shift = tl.full([1], shift, dtype=tl.bfloat16)
        clamp_abs = tl.full([1], clamp_abs, dtype=tl.bfloat16)
        for t in range(0, T, 1):
            x_ptrs = x_seq_ptr + t*T_stride + x_offsets_per_time_step
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = tl.where(h >= 1., 1., 0).to(tl.bfloat16)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(s_ptrs, s, mask=mask_x)

            # quantize ot
            ot = h - one
            ot = tl.clamp(ot - shift, -clamp_abs, clamp_abs) * scale
            ot = tl.cast(ot, tl.float8e4b8)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(ot_ptrs, ot, mask=mask_x)

    @triton.jit
    def _handwritten_hqlif_backward_not_detached_triton_float32(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.float32)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)

            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = tl.cast(ot, tl.float32)
            ot = (ot/scale) + shift

            s = tl.where(ot >= 0., 1., 0.)

            sg = pi * ot
            grad_v = (grad_s - grad_v * (ot+1.)) / (1. + sg*sg) + grad_v * (1-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_hqlif_backward_not_detached_triton_float16(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.float16)
        pi = tl.full([1], pi, dtype=tl.float16)
        one = tl.full([1], 1., dtype=tl.float16)
        scale = tl.full([1], scale, dtype=tl.float16)
        shift = tl.full([1], shift, dtype=tl.float16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.float16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = tl.cast(ot, tl.float16)
            ot = (ot/scale) + shift

            s = tl.where(ot >= 0., 1., 0.).to(tl.float16)

            sg = pi * ot
            sg = (one / (one + sg*sg)).to(tl.float16)
            grad_v = (grad_s - grad_v * (ot+one)) * sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_hqlif_backward_not_detached_triton_bfloat16(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.bfloat16)
        pi = tl.full([1], pi, dtype=tl.bfloat16)
        one = tl.full([1], 1., dtype=tl.bfloat16)
        scale = tl.full([1], scale, dtype=tl.bfloat16)
        shift = tl.full([1], shift, dtype=tl.bfloat16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.bfloat16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = tl.cast(ot, tl.bfloat16)
            ot = (ot/scale) + shift

            s = tl.where(ot >= 0., 1., 0.).to(tl.bfloat16)

            sg = pi * ot
            sg = (one / (one + sg*sg)).to(tl.bfloat16)
            grad_v = (grad_s - grad_v * (ot+one)) * sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_hqlif_backward_detached_triton_float32(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        shift,
        scale,
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

            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = tl.cast(ot, tl.float32)
            ot = (ot/scale) + shift

            s = tl.where(ot >= 0., 1., 0.)

            sg = pi * ot
            grad_v = grad_s / (1. + sg*sg) + grad_v * (1-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_hqlif_backward_detached_triton_float16(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.float16)
        pi = tl.full([1], pi, dtype=tl.float16)
        one = tl.full([1], 1., dtype=tl.float16)
        scale = tl.full([1], scale, dtype=tl.float16)
        shift = tl.full([1], shift, dtype=tl.float16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.float16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = tl.cast(ot, tl.float16)
            ot = (ot/scale) + shift

            s = tl.where(ot >= 0., 1., 0.).to(tl.float16)

            sg = pi * ot
            sg = (one / (one + sg*sg)).to(tl.float16)
            grad_v = grad_s*sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_hqlif_backward_detached_triton_bfloat16(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T,
        NCL,
        T_stride,
        decay_lambda,
        pi,
        shift,
        scale,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=tl.bfloat16)
        pi = tl.full([1], pi, dtype=tl.bfloat16)
        one = tl.full([1], 1., dtype=tl.bfloat16)
        scale = tl.full([1], scale, dtype=tl.bfloat16)
        shift = tl.full([1], shift, dtype=tl.bfloat16)
        decay_lambda = tl.full([1], decay_lambda, dtype=tl.bfloat16)
        for t in range(T - 1, -1, -1):
            grad_s_ptrs = grad_s_seq_ptr + t*T_stride + x_offsets_per_time_step
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            ot_ptrs = ot_seq_ptr + t*T_stride + x_offsets_per_time_step
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = tl.cast(ot, tl.bfloat16)
            ot = (ot/scale) + shift

            s = tl.where(ot >= 0., 1., 0.).to(tl.bfloat16)

            sg = pi * ot
            sg = (one / (one + sg*sg)).to(tl.bfloat16)
            grad_v = grad_s*sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + t*T_stride + x_offsets_per_time_step
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    _handwritten_hqlif_forward_triton_kernel = {
        (torch.float32, torch.float16):
            _handwritten_hqlif_forward_triton_float32_float16,
        (torch.float16, torch.float16):
            _handwritten_hqlif_forward_triton_float16_float16,
        (torch.bfloat16, torch.float16):
            _handwritten_hqlif_forward_triton_bfloat16_float16,
    }
    if FLOAT8_AVAILABLE:
        _handwritten_hqlif_forward_triton_kernel.update({
            (torch.float32, torch.float8_e4m3fn):
                _handwritten_hqlif_forward_triton_float32_float8e4m3fn,
            (torch.float16, torch.float8_e4m3fn):
                _handwritten_hqlif_forward_triton_float16_float8e4m3fn,
            (torch.bfloat16, torch.float8_e4m3fn):
                _handwritten_hqlif_forward_triton_bfloat16_float8e4m3fn,
        })

    _handwritten_hqlif_backward_not_detached_triton_kernel = {
        torch.float32:
            _handwritten_hqlif_backward_not_detached_triton_float32,
        torch.float16:
            _handwritten_hqlif_backward_not_detached_triton_float16,
        torch.bfloat16:
            _handwritten_hqlif_backward_not_detached_triton_bfloat16,
    }

    _handwritten_hqlif_backward_detached_triton_kernel = {
        torch.float32: _handwritten_hqlif_backward_detached_triton_float32,
        torch.float16: _handwritten_hqlif_backward_detached_triton_float16,
        torch.bfloat16: _handwritten_hqlif_backward_detached_triton_bfloat16,
    }

    def handwritten_hqlif_forward_triton(
        x_seq, decay_lambda, h_quantizer, block_ncl=512
    ):
        T = x_seq.shape[0]
        NCL = x_seq[0].numel()
        grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

        s_seq = torch.empty_like(x_seq)
        ot_seq = torch.empty_like(x_seq, dtype=h_quantizer.dtype)

        if not isinstance(h_quantizer, ClampProjHQuantizer):
            raise ValueError(
                "Only ClampProjHQuantizer is supported for Triton kernel."
            )

        dtype, cdtype = x_seq.dtype, h_quantizer.dtype
        if (
            dtype == torch.bfloat16 or cdtype == torch.bfloat16
        ) and not TRITON_BFLOAT16_AVAILABLE:
            raise RuntimeError(
                "Triton kernel with bfloat16 is not supported on devices "
                "with compute capability < 8.0. Use float16 instead."
            )
        kernel = _handwritten_hqlif_forward_triton_kernel[(dtype, cdtype)]

        with torch.cuda.device(x_seq.device):
            kernel[grid](
                x_seq,
                s_seq,
                ot_seq,
                T,
                NCL,
                x_seq.stride(0),
                decay_lambda,
                h_quantizer.clamp_abs,
                h_quantizer.shift,
                h_quantizer.scale,
                BLOCK_NCL=block_ncl
            )
        return s_seq, ot_seq

    def handwritten_hqlif_backward_not_detached_triton(
        grad_s_seq, ot_seq, decay_lambda, T, h_quantizer, block_ncl=512
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
        kernel = _handwritten_hqlif_backward_not_detached_triton_kernel[dtype]

        with torch.cuda.device(grad_s_seq.device):
            kernel[grid](
                grad_s_seq,
                ot_seq,
                grad_x_seq,
                T,
                NCL,
                grad_s_seq.stride(0),
                decay_lambda,
                torch.pi,
                h_quantizer.shift,
                h_quantizer.scale,
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

    def handwritten_hqlif_backward_detached_triton(
        grad_s_seq, ot_seq, decay_lambda, T, h_quantizer, block_ncl=512
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
        kernel = _handwritten_hqlif_backward_detached_triton_kernel[dtype]

        with torch.cuda.device(grad_s_seq.device):
            kernel[grid](
                grad_s_seq,
                ot_seq,
                grad_x_seq,
                T,
                NCL,
                grad_s_seq.stride(0),
                decay_lambda,
                torch.pi,
                h_quantizer.shift,
                h_quantizer.scale,
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

except Exception as e:
    TRITON_AVAILABLE = False
    print(f"triton is not available. {e}")
