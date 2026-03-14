# =========================
# mxfp8.py
# =========================
#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType, SwizzleType

from quant import quant_mxfp8, require

@torch.no_grad()
def mxfp8_mm(
    A_bf16: torch.Tensor,
    B_bf16: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    apply_rmsnorm_lhs: bool = False,
    rms_eps: float = 1e-6,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Fully dynamic MXFP8 GEMM: C = A @ B^T."""
    require(A_bf16.is_cuda and B_bf16.is_cuda, "mxfp8_mm requires CUDA tensors.")

    inv_rms = None

    if apply_rmsnorm_lhs:
        a_fp8, s_a_swz, inv_rms = quant_mxfp8(
            A_bf16,
            apply_rmsnorm=True,
            epsilon=rms_eps,
        )
    else:
        a_fp8, s_a_swz = quant_mxfp8(A_bf16)

    b_fp8, s_b_swz = quant_mxfp8(B_bf16)

    C = F.scaled_mm(
        a_fp8,
        b_fp8.t(),
        s_a_swz,
        ScalingType.BlockWise1x32,
        s_b_swz,
        ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=out_dtype,
        contraction_dim=(1, 0),
    )

    if apply_rmsnorm_lhs:
        return C, inv_rms
    return C