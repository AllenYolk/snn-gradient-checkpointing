try:
    import triton
    import triton.language as tl
    import torch

    TRITON_AVAILABLE = True

    type_dict = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
    }
    dc = torch.cuda.get_device_capability()
    if dc[0] < 8 or not hasattr(tl, "bfloat16"):
        print(
            "Triton kernel with bfloat16 is not supported on devices "
            "with compute capability < 8.0. "
            f"Your device's capability is: {dc}."
        )
        TRITON_BFLOAT16_AVAILABLE = False
    else:
        TRITON_BFLOAT16_AVAILABLE = True
        type_dict[torch.bfloat16] = tl.bfloat16

    TORCH_FLOAT8E4M3FN_AVAILABLE = hasattr(torch, "float8_e4m3fn")
    if float(f"{dc[0]}.{dc[1]}") < 8.9 or not hasattr(tl, "float8e4nv"):
        print(
            "Triton kernel with float8e4nv (float8_e4m3fn) is not supported on "
            "devices with compute capability < 8.9. "
            f"Your devices's capability is: {dc}."
        )
        TRITON_FLOAT8E4NV_AVAILABLE = False
    else:
        TRITON_FLOAT8E4NV_AVAILABLE = True
        if TORCH_FLOAT8E4M3FN_AVAILABLE:
            type_dict[torch.float8_e4m3fn] = tl.float8e4nv

    DEFAULT_BLOCK_SIZE = 512
    assert DEFAULT_BLOCK_SIZE % 8 == 0, "BLOCK_SIZE must be dividable by 8"

    @triton.jit
    def _handwritten_lif_forward_triton(
        x_seq_ptr,
        s_seq_ptr,
        h_seq_ptr,
        T: tl.constexpr,
        NCL,
        T_stride,
        decay_lambda,
        dtype: tl.constexpr,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=dtype)
        decay_lambda = tl.full([1], decay_lambda, dtype=dtype)
        one = tl.full([1], 1., dtype=dtype)

        for t in tl.static_range(0, T, 1):
            x_offsets = t*T_stride + x_offsets_per_time_step
            x_ptrs = x_seq_ptr + x_offsets
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = (h >= 1.).to(dtype)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + x_offsets
            h_ptrs = h_seq_ptr + x_offsets
            tl.store(s_ptrs, s, mask=mask_x)
            tl.store(h_ptrs, h, mask=mask_x)

    @triton.jit
    def _handwritten_lif_backward_not_detached_triton(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T: tl.constexpr,
        NCL,
        T_stride,
        decay_lambda,
        pi: tl.constexpr,
        dtype: tl.constexpr,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
        pi = tl.full([1], pi, dtype=dtype)
        one = tl.full([1], 1., dtype=dtype)
        decay_lambda = tl.full([1], decay_lambda, dtype=dtype)
        for t in tl.static_range(T - 1, -1, -1):
            x_offsets = t*T_stride + x_offsets_per_time_step
            grad_s_ptrs = grad_s_seq_ptr + x_offsets
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + x_offsets
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = (h >= 1.).to(dtype)

            sg = pi * (h-one)
            sg = (one / (one + sg*sg)).to(dtype)
            grad_v = (grad_s - grad_v*h) * sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + x_offsets
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_lif_backward_detached_triton(
        grad_s_seq_ptr,
        h_seq_ptr,
        grad_x_seq_ptr,
        T: tl.constexpr,
        NCL,
        T_stride,
        decay_lambda,
        pi: tl.constexpr,
        dtype: tl.constexpr,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
        pi = tl.full([1], pi, dtype=dtype)
        one = tl.full([1], 1., dtype=dtype)
        decay_lambda = tl.full([1], decay_lambda, dtype=dtype)
        for t in tl.static_range(T - 1, -1, -1):
            x_offsets = t*T_stride + x_offsets_per_time_step
            grad_s_ptrs = grad_s_seq_ptr + x_offsets
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            h_ptrs = h_seq_ptr + x_offsets
            h = tl.load(h_ptrs, mask=mask_x, other=0.)
            s = (h >= 1.).to(dtype)

            sg = pi * (h-one)
            sg = (one / (one + sg*sg)).to(dtype)
            grad_v = grad_s*sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + x_offsets
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    def handwritten_lif_forward_triton(
        x_seq, decay_lambda, block_ncl=DEFAULT_BLOCK_SIZE
    ):
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

        with torch.cuda.device(x_seq.device):
            _handwritten_lif_forward_triton[grid](
                x_seq,
                s_seq,
                h_seq,
                T,
                NCL,
                x_seq.stride(0),
                decay_lambda,
                type_dict[dtype],
                BLOCK_NCL=block_ncl
            )
        return s_seq, h_seq

    def handwritten_lif_backward_not_detached_triton(
        grad_s_seq, h_seq, decay_lambda, T, block_ncl=DEFAULT_BLOCK_SIZE
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

        with torch.cuda.device(grad_s_seq.device):
            _handwritten_lif_backward_not_detached_triton[grid](
                grad_s_seq,
                h_seq,
                grad_x_seq,
                T,
                NCL,
                grad_s_seq.stride(0),
                decay_lambda,
                torch.pi,
                type_dict[dtype],
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

    def handwritten_lif_backward_detached_triton(
        grad_s_seq, h_seq, decay_lambda, T, block_ncl=DEFAULT_BLOCK_SIZE
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

        with torch.cuda.device(grad_s_seq.device):
            _handwritten_lif_backward_detached_triton[grid](
                grad_s_seq,
                h_seq,
                grad_x_seq,
                T,
                NCL,
                grad_s_seq.stride(0),
                decay_lambda,
                torch.pi,
                type_dict[dtype],
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

    # ========== Hand-written Multistep LIF neuron with H quantization =========
    @triton.jit
    def _handwritten_hqlif_forward_triton(
        x_seq_ptr,
        s_seq_ptr,
        ot_seq_ptr,
        T: tl.constexpr,
        NCL,
        T_stride,
        decay_lambda,
        clamp_abs: tl.constexpr,
        shift: tl.constexpr,
        scale: tl.constexpr,
        dtype: tl.constexpr,
        compressed_dtype: tl.constexpr,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        v = tl.zeros([BLOCK_NCL], dtype=dtype)
        decay_lambda = tl.full([1], decay_lambda, dtype=dtype)
        one = tl.full([1], 1., dtype=dtype)
        scale = tl.full([1], scale, dtype=dtype)
        shift = tl.full([1], shift, dtype=dtype)
        clamp_abs = tl.full([1], clamp_abs, dtype=dtype)
        for t in tl.static_range(0, T, 1):
            x_offsets = t*T_stride + x_offsets_per_time_step
            x_ptrs = x_seq_ptr + x_offsets
            x = tl.load(x_ptrs, mask=mask_x, other=0.)

            h = decay_lambda*v + x
            s = (h >= 1.).to(dtype)
            v = h * (one-s)

            s_ptrs = s_seq_ptr + x_offsets
            tl.store(s_ptrs, s, mask=mask_x)

            # quantize ot
            ot = h - one
            ot = tl.clamp(ot - shift, -clamp_abs, clamp_abs) * scale
            ot = ot.to(compressed_dtype, fp_downcast_rounding="rtne")
            ot_ptrs = ot_seq_ptr + x_offsets
            tl.store(ot_ptrs, ot, mask=mask_x)

    @triton.jit
    def _handwritten_hqlif_backward_not_detached_triton(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T: tl.constexpr,
        NCL,
        T_stride,
        decay_lambda,
        pi: tl.constexpr,
        shift: tl.constexpr,
        scale: tl.constexpr,
        dtype: tl.constexpr,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
        pi = tl.full([1], pi, dtype=dtype)
        one = tl.full([1], 1., dtype=dtype)
        scale = tl.full([1], scale, dtype=dtype)
        shift = tl.full([1], shift, dtype=dtype)
        decay_lambda = tl.full([1], decay_lambda, dtype=dtype)
        for t in tl.static_range(T - 1, -1, -1):
            x_offsets = t*T_stride + x_offsets_per_time_step
            grad_s_ptrs = grad_s_seq_ptr + x_offsets
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            ot_ptrs = ot_seq_ptr + x_offsets
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = ot.to(dtype)
            ot = (ot/scale) + shift

            s = (ot >= 0.).to(dtype)

            sg = pi * ot
            sg = (one / (one + sg*sg)).to(dtype)
            grad_v = (grad_s - grad_v * (ot+one)) * sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + x_offsets
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    @triton.jit
    def _handwritten_hqlif_backward_detached_triton(
        grad_s_seq_ptr,
        ot_seq_ptr,
        grad_x_seq_ptr,
        T: tl.constexpr,
        NCL,
        T_stride,
        decay_lambda,
        pi: tl.constexpr,
        shift: tl.constexpr,
        scale: tl.constexpr,
        dtype: tl.constexpr,
        BLOCK_NCL: tl.constexpr,
    ):
        pid_ncl = tl.program_id(0)
        x_offsets_per_time_step = tl.arange(0, BLOCK_NCL) + pid_ncl*BLOCK_NCL
        mask_x = x_offsets_per_time_step < NCL

        grad_v = tl.zeros([BLOCK_NCL], dtype=dtype)
        pi = tl.full([1], pi, dtype=dtype)
        one = tl.full([1], 1., dtype=dtype)
        scale = tl.full([1], scale, dtype=dtype)
        shift = tl.full([1], shift, dtype=dtype)
        decay_lambda = tl.full([1], decay_lambda, dtype=dtype)
        for t in tl.static_range(T - 1, -1, -1):
            x_offsets = t*T_stride + x_offsets_per_time_step
            grad_s_ptrs = grad_s_seq_ptr + x_offsets
            grad_s = tl.load(grad_s_ptrs, mask=mask_x, other=0.)
            ot_ptrs = ot_seq_ptr + x_offsets
            ot = tl.load(ot_ptrs, mask=mask_x, other=0.)
            # dequantize ot
            ot = ot.to(dtype)
            ot = (ot/scale) + shift

            s = (ot >= 0.).to(dtype)

            sg = pi * ot
            sg = (one / (one + sg*sg)).to(dtype)
            grad_v = grad_s*sg + grad_v * (one-s)

            grad_x_ptrs = grad_x_seq_ptr + x_offsets
            tl.store(grad_x_ptrs, grad_v, mask=mask_x)
            grad_v = grad_v * decay_lambda

    def handwritten_hqlif_forward_triton(
        x_seq, decay_lambda, h_quantizer, block_ncl=DEFAULT_BLOCK_SIZE
    ):
        T = x_seq.shape[0]
        NCL = x_seq[0].numel()
        grid = lambda meta: (triton.cdiv(NCL, meta['BLOCK_NCL']),)

        s_seq = torch.empty_like(x_seq)
        ot_seq = torch.empty_like(x_seq, dtype=h_quantizer.dtype)

        if not hasattr(h_quantizer, "clamp_abs"):
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

        with torch.cuda.device(x_seq.device):
            _handwritten_hqlif_forward_triton[grid](
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
                type_dict[dtype],
                type_dict[cdtype],
                BLOCK_NCL=block_ncl
            )
        return s_seq, ot_seq

    def handwritten_hqlif_backward_not_detached_triton(
        grad_s_seq,
        ot_seq,
        decay_lambda,
        T,
        h_quantizer,
        block_ncl=DEFAULT_BLOCK_SIZE
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

        with torch.cuda.device(grad_s_seq.device):
            _handwritten_hqlif_backward_not_detached_triton[grid](
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
                type_dict[dtype],
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

    def handwritten_hqlif_backward_detached_triton(
        grad_s_seq,
        ot_seq,
        decay_lambda,
        T,
        h_quantizer,
        block_ncl=DEFAULT_BLOCK_SIZE
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

        with torch.cuda.device(grad_s_seq.device):
            _handwritten_hqlif_backward_detached_triton[grid](
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
                type_dict[dtype],
                BLOCK_NCL=block_ncl
            )
        return grad_x_seq

    # =========================== BitSpikeCompressor ===========================
    @triton.jit
    def _bit_spike_compress_triton(
        s_seq_ptr,  # fp32, 0 or 1
        s_seq_compressed_ptr,
        n_elements,
        n_compressed_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        store_offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        store_mask = store_offsets < n_compressed_elements

        s_seq_compressed = tl.zeros([
            BLOCK_SIZE,
        ], dtype=tl.uint8)

        for i in tl.static_range(8):
            load_offsets = i + store_offsets*8
            load_mask = load_offsets < n_elements
            s_seq = tl.load(s_seq_ptr + load_offsets, mask=load_mask, other=0.)
            s_seq = s_seq.to(tl.uint8)
            s_seq_compressed = s_seq_compressed | (s_seq << i)

        tl.store(
            s_seq_compressed_ptr + store_offsets,
            s_seq_compressed,
            mask=store_mask
        )

    @triton.jit
    def _bit_spike_decompress_triton(
        s_seq_compressed_ptr,
        s_seq_decompressed_ptr,
        n_compressed_elements,
        n_decompressed_elements,
        BLOCK_SIZE: tl.constexpr,  # must be dividable by 8
    ):
        pid = tl.program_id(0)
        load_offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        load_mask = load_offsets < n_compressed_elements

        s_seq_compressed = tl.load(
            s_seq_compressed_ptr + load_offsets,
            mask=load_mask,
            other=0,
        )

        for i in tl.static_range(8):
            store_offsets = i + load_offsets*8
            store_mask = store_offsets < n_decompressed_elements
            tl.store(
                s_seq_decompressed_ptr + store_offsets,
                (s_seq_compressed >> i) & 1,
                mask=store_mask,
            )

    def bit_spike_compress_triton(s_seq, block_size=DEFAULT_BLOCK_SIZE // 8):
        # s_seq: float32, ndim=1
        s_seq = s_seq.reshape(-1)
        n_elements = s_seq.numel()
        n_compressed_elements = (n_elements+7) // 8
        s_seq_compressed = torch.zeros(
            n_compressed_elements, dtype=torch.uint8, device=s_seq.device
        )
        grid = lambda meta: (
            triton.cdiv(n_compressed_elements, meta['BLOCK_SIZE']),
        )

        with torch.cuda.device(s_seq.device):
            _bit_spike_compress_triton[grid](
                s_seq,
                s_seq_compressed,
                n_elements,
                n_compressed_elements,
                BLOCK_SIZE=block_size
            )
        return s_seq_compressed

    def bit_spike_decompress_triton(
        s_seq_compressed, shape, block_size=DEFAULT_BLOCK_SIZE // 8
    ):
        # s_seq: uint8, ndim=1
        n_compressed_elements = s_seq_compressed.numel()
        n_decompressed_elements = shape.numel()
        s_seq_decompressed = torch.zeros(
            n_decompressed_elements,
            dtype=torch.uint8,
            device=s_seq_compressed.device
        )
        grid = lambda meta: (
            triton.cdiv(n_compressed_elements, meta['BLOCK_SIZE']),
        )

        with torch.cuda.device(s_seq_compressed.device):
            _bit_spike_decompress_triton[grid](
                s_seq_compressed,
                s_seq_decompressed,
                n_compressed_elements,
                n_decompressed_elements,
                BLOCK_SIZE=block_size
            )
        return s_seq_decompressed.reshape(shape)

except Exception as e:
    TRITON_AVAILABLE = False
    TRITON_BFLOAT16_AVAILABLE = False
    TRITON_FLOAT8E4NV_AVAILABLE = False
    TORCH_FLOAT8E4M3FN_AVAILABLE = hasattr(torch, "float8_e4m3fn")
    print(f"triton is not available. {e}")
