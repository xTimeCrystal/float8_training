# =========================
# quant.py
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cupy as cp
import torch


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


require(hasattr(torch, "float8_e4m3fn"), "Your torch build lacks torch.float8_e4m3fn.")
require(hasattr(torch, "float8_e8m0fnu"), "Your torch build lacks torch.float8_e8m0fnu.")


_KERNEL_MXFP8_FUSED = r"""
extern "C" {

__device__ __forceinline__ unsigned char cvt_e4m3(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}

__device__ __forceinline__ int exp_unbiased_rceil_448(float amax) {
    if (!(amax > 0.0f)) return 0;
    int bits = __float_as_int(amax);
    int biased_e = (bits >> 23) & 0xff;
    int e = biased_e - 127;
    int mant_bits = (bits & 0x007fffff) | 0x3f800000;
    float m = __int_as_float(mant_bits);
    int exp = (e - 8) + ((m > 1.75f) ? 1 : 0);
    return exp;
}

__device__ __forceinline__ unsigned char exp_to_e8m0_biased(int exp_unbiased) {
    int e = exp_unbiased + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return (unsigned char)e;
}

__device__ __forceinline__ float inv_scale_from_exp(int exp_unbiased) {
    int be = 127 - exp_unbiased;
    if (be <= 0) return 0.0f;
    if (be >= 255) return __int_as_float(0x7f800000); // inf
    return __int_as_float((be & 0xff) << 23);
}

__global__ void quant_mxfp8_fused_opt(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_u8,
    unsigned char* __restrict__ s_u8,
    int num_elements,
    int cols
){
    int num_blocks = num_elements >> 5;

    int tid  = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = tid & 31;
    int warp = tid >> 5;

    int block_base    = warp << 4;
    int block_in_warp = lane >> 1;
    int blk           = block_base + block_in_warp;
    if (blk >= num_blocks) return;

    int half = lane & 1;
    int elem_base = (blk << 5) + (half << 4);

    const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
    uint4 v0 = in4[0];
    uint4 v1 = in4[1];

    unsigned int w[8];
    w[0]=v0.x; w[1]=v0.y; w[2]=v0.z; w[3]=v0.w;
    w[4]=v1.x; w[5]=v1.y; w[6]=v1.z; w[7]=v1.w;

    float f[16];
    float local_max = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        unsigned int u = w[i];
        unsigned short lo = (unsigned short)(u & 0xffff);
        unsigned short hi = (unsigned short)(u >> 16);

        float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
        float a1 = __int_as_float((int)(((unsigned int)hi) << 16));

        f[2*i+0] = a0;
        f[2*i+1] = a1;

        local_max = fmaxf(local_max, fabsf(a0));
        local_max = fmaxf(local_max, fabsf(a1));
    }

    float other = __shfl_xor_sync(0xffffffff, local_max, 1);
    float amax  = fmaxf(local_max, other);

    int exp_unbiased = exp_unbiased_rceil_448(amax);
    unsigned char e8 = (amax > 0.0f) ? exp_to_e8m0_biased(exp_unbiased) : (unsigned char)0;
    float inv_scale  = (amax > 0.0f) ? inv_scale_from_exp(exp_unbiased) : 1.0f;

    if (half == 0) {
        int r = blk / cols;
        int c = blk % cols;

        int r_blk = r >> 7;
        int c_blk = c >> 2;
        int r_in  = r & 127;
        int c_in  = c & 3;

        int cols_div_4 = cols >> 2;
        int tile_idx = r_blk * cols_div_4 + c_blk;

        int new_row = r_in & 31;
        int new_col = ((r_in >> 5) << 2) + c_in;

        int swizzled_blk = (tile_idx << 9) + (new_row << 4) + new_col;
        s_u8[swizzled_blk] = e8;
    }

    unsigned int p0=0, p1=0, p2=0, p3=0;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        unsigned char q = cvt_e4m3(f[i] * inv_scale);
        int sh = (i & 3) * 8;
        if (i < 4)       p0 |= ((unsigned int)q) << sh;
        else if (i < 8)  p1 |= ((unsigned int)q) << sh;
        else if (i < 12) p2 |= ((unsigned int)q) << sh;
        else             p3 |= ((unsigned int)q) << sh;
    }
    uint4 out; out.x=p0; out.y=p1; out.z=p2; out.w=p3;
    ((uint4*)(q_u8 + elem_base))[0] = out;
}
}
"""

_KERNEL_RMSNORM_MXFP8_SMEM = r"""
extern "C" {

__device__ __forceinline__ unsigned char cvt_e4m3(float v) {
    unsigned short packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(packed) : "f"(v));
    return (unsigned char)(packed & 0x00FF);
}

__device__ __forceinline__ int exp_unbiased_rceil_448(float amax) {
    if (!(amax > 0.0f)) return 0;
    int bits = __float_as_int(amax);
    int biased_e = (bits >> 23) & 0xff;
    int e = biased_e - 127;
    int mant_bits = (bits & 0x007fffff) | 0x3f800000;
    float m = __int_as_float(mant_bits);
    int exp = (e - 8) + ((m > 1.75f) ? 1 : 0);
    return exp;
}

__device__ __forceinline__ unsigned char exp_to_e8m0_biased(int exp_unbiased) {
    int e = exp_unbiased + 127;
    if (e < 0) e = 0;
    if (e > 254) e = 254;
    return (unsigned char)e;
}

__device__ __forceinline__ float inv_scale_from_exp(int exp_unbiased) {
    int be = 127 - exp_unbiased;
    if (be <= 0) return 0.0f;
    if (be >= 255) return __int_as_float(0x7f800000); // inf
    return __int_as_float((be & 0xff) << 23);
}

__global__ void rmsnorm_quant_mxfp8_smem_opt(
    const unsigned short* __restrict__ x_bf16,
    unsigned char* __restrict__ q_u8,
    unsigned char* __restrict__ s_u8,
    float* __restrict__ inv_rms_out,
    int R, int K, int pad_cols, float epsilon
){
    extern __shared__ unsigned short smem_row[];
    int r = blockIdx.x;
    if (r >= R) return;

    int tid = threadIdx.x;
    int cols16 = K / 16;

    // Phase 1: RMS
    float sum_sq = 0.0f;
    for (int c = tid; c < cols16; c += blockDim.x) {
        int elem_base = r * K + (c * 16);
        const uint4* in4 = (const uint4*)(x_bf16 + elem_base);
        uint4 v0 = in4[0], v1 = in4[1];

        uint4* smem_out = (uint4*)(smem_row + (c * 16));
        smem_out[0] = v0;
        smem_out[1] = v1;

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);
            float a0 = __int_as_float((int)(((unsigned int)lo) << 16));
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16));
            sum_sq += a0 * a0 + a1 * a1;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    static __shared__ float shared_sum[32];
    int lane = tid % 32;
    int wid = tid / 32;

    if (lane == 0) shared_sum[wid] = sum_sq;
    __syncthreads();

    float block_sum = (tid < (blockDim.x / 32)) ? shared_sum[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (tid == 0) {
            float irms = rsqrtf((block_sum / (float)K) + epsilon);
            shared_sum[0] = irms;
            inv_rms_out[r] = irms;
        }
    }
    __syncthreads();

    float inv_rms = shared_sum[0];

    // Phase 2: Quantize
    int mxfp8_cols = K / 32;

    for (int blk_idx = tid / 2; blk_idx < mxfp8_cols; blk_idx += blockDim.x / 2) {
        int half = tid & 1;
        int c = blk_idx;

        int smem_idx = (blk_idx * 32) + (half * 16);
        const uint4* in4 = (const uint4*)(smem_row + smem_idx);
        uint4 v0 = in4[0], v1 = in4[1];

        unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float f[16];
        float local_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short lo = (unsigned short)(w[i] & 0xffff);
            unsigned short hi = (unsigned short)(w[i] >> 16);

            float a0 = __int_as_float((int)(((unsigned int)lo) << 16)) * inv_rms;
            float a1 = __int_as_float((int)(((unsigned int)hi) << 16)) * inv_rms;

            f[2*i+0] = a0;
            f[2*i+1] = a1;
            local_max = fmaxf(local_max, fabsf(a0));
            local_max = fmaxf(local_max, fabsf(a1));
        }

        float other = __shfl_xor_sync(0xffffffff, local_max, 1);
        float amax  = fmaxf(local_max, other);

        int exp_unbiased = exp_unbiased_rceil_448(amax);
        unsigned char e8 = (amax > 0.0f) ? exp_to_e8m0_biased(exp_unbiased) : (unsigned char)0;
        float inv_scale  = (amax > 0.0f) ? inv_scale_from_exp(exp_unbiased) : 1.0f;

        if (half == 0) {
            int r_blk = r >> 7;
            int c_blk = c >> 2;
            int r_in  = r & 127;
            int c_in  = c & 3;

            int tile_idx = r_blk * (pad_cols >> 2) + c_blk;

            int new_row = r_in & 31;
            int new_col = ((r_in >> 5) << 2) + c_in;

            int swizzled_blk = (tile_idx << 9) + (new_row << 4) + new_col;
            s_u8[swizzled_blk] = e8;
        }

        unsigned int p0=0, p1=0, p2=0, p3=0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned char q = cvt_e4m3(f[i] * inv_scale);
            int sh = (i & 3) * 8;
            if (i < 4)       p0 |= ((unsigned int)q) << sh;
            else if (i < 8)  p1 |= ((unsigned int)q) << sh;
            else if (i < 12) p2 |= ((unsigned int)q) << sh;
            else             p3 |= ((unsigned int)q) << sh;
        }
        uint4 out; out.x=p0; out.y=p1; out.z=p2; out.w=p3;

        int global_q_idx = r * K + smem_idx;
        ((uint4*)(q_u8 + global_q_idx))[0] = out;
    }
}
}
"""


@dataclass
class _MXFP8QuantKernels:
    standard: cp.RawKernel
    rmsnorm_smem: cp.RawKernel

    @staticmethod
    def build() -> "_MXFP8QuantKernels":
        mod_std = cp.RawModule(code=_KERNEL_MXFP8_FUSED, options=("--std=c++17",))
        fn_std = mod_std.get_function("quant_mxfp8_fused_opt")

        mod_rms = cp.RawModule(code=_KERNEL_RMSNORM_MXFP8_SMEM, options=("--std=c++17",))
        fn_rms = mod_rms.get_function("rmsnorm_quant_mxfp8_smem_opt")
        fn_rms.max_dynamic_shared_size_bytes = 98304

        return _MXFP8QuantKernels(
            standard=fn_std,
            rmsnorm_smem=fn_rms,
        )


_KERNELS: Optional[_MXFP8QuantKernels] = None


def _get_kernels() -> _MXFP8QuantKernels:
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _MXFP8QuantKernels.build()
    return _KERNELS


def quant_mxfp8(
    x_bf16: torch.Tensor,
    *,
    apply_rmsnorm: bool = False,
    epsilon: float = 1e-6,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    x = x_bf16.contiguous()
    require(
        x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2,
        "quant_mxfp8 expects a CUDA BF16 2D tensor.",
    )

    R, K = x.shape
    require(K % 32 == 0, "K must be divisible by 32 for MXFP8 1x32 scaling.")

    num_elements = x.numel()
    cols = K // 32

    pad_R = ((R + 127) // 128) * 128
    pad_cols = ((cols + 3) // 4) * 4

    q_u8 = torch.empty((num_elements,), device=x.device, dtype=torch.uint8)
    s_u8 = torch.zeros((pad_R * pad_cols,), device=x.device, dtype=torch.uint8)

    if not apply_rmsnorm:
        threads = 256
        warps_per_block = threads // 32
        blocks_per_warp = 16

        num_blocks32 = num_elements // 32
        num_warps = (num_blocks32 + blocks_per_warp - 1) // blocks_per_warp
        grid = ((num_warps + warps_per_block - 1) // warps_per_block,)

        kern = _get_kernels().standard
        kern(
            grid,
            (threads,),
            (
                x.data_ptr(),
                q_u8.data_ptr(),
                s_u8.data_ptr(),
                num_elements,
                cols,
            ),
        )

        q = q_u8.view(torch.float8_e4m3fn).view(R, K)
        s = s_u8.view(torch.float8_e8m0fnu).view(pad_R, pad_cols)
        return q, s

    inv_rms_out = torch.empty((R,), device=x.device, dtype=torch.float32)

    threads = 256
    grid = (R,)
    smem_bytes = K * 2

    kern = _get_kernels().rmsnorm_smem
    kern(
        grid,
        (threads,),
        (
            x.data_ptr(),
            q_u8.data_ptr(),
            s_u8.data_ptr(),
            inv_rms_out.data_ptr(),
            R,
            K,
            pad_cols,
            cp.float32(epsilon),
        ),
        shared_mem=smem_bytes,
    )

    q = q_u8.view(torch.float8_e4m3fn).view(R, K)
    s = s_u8.view(torch.float8_e8m0fnu).view(pad_R, pad_cols)
    return q, s, inv_rms_out


__all__ = [
    "require",
    "quant_mxfp8",
]