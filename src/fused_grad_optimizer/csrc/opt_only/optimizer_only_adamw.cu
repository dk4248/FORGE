/*
 * Custom optimizer-only AdamW kernel.
 *
 * Applies the AdamW update in-place on W, M, V given a pre-computed bf16 grad.
 * Used as the second step after CUTLASS 2.x GEMM materializes grad_W.
 *
 *   m = beta1*m + (1-beta1)*g
 *   v = beta2*v + (1-beta2)*g^2
 *   w = w*(1 - lr*wd) - lr * (m/bc1) / (sqrt(v/bc2) + eps)
 *
 * Design:
 *   - 128-bit vectorized (ld.global.cs.v4.b32 / st.global.cs.v4.b32) — 8 bf16
 *     per lane per memory instruction.
 *   - Grid-stride loop so a fixed block count covers any tensor size.
 *   - Cache-streaming hints (.cs): W/M/V/grad are single-use per optimizer
 *     step, don't pollute L2.
 *   - All tensors share the same (V, H) row-major layout, so element index is
 *     the same across them.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

constexpr int NT        = 256;         // threads per block
constexpr int ELEMS_PER_THREAD = 8;    // 128 bits / 16 bits bf16
constexpr int TILE      = NT * ELEMS_PER_THREAD;   // 2048 elems per block


__device__ __forceinline__ uint4 ld_global_cs_v4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ void st_global_cs_v4(void* ptr, uint4 v) {
    asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}


__global__ void __launch_bounds__(NT)
optimizer_only_adamw_kernel(
    const __nv_bfloat16* __restrict__ G,
    __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ M,
    __nv_bfloat16* __restrict__ VB,
    int64_t total_elems,
    float lr, float beta1, float beta2, float eps, float wd,
    float inv_bc1, float inv_bc2
) {
    const float one_minus_lr_wd = 1.0f - lr * wd;
    const float one_minus_b1    = 1.0f - beta1;
    const float one_minus_b2    = 1.0f - beta2;

    const int64_t n_vec       = total_elems / ELEMS_PER_THREAD;
    const int64_t tail_start  = n_vec * ELEMS_PER_THREAD;
    const int64_t grid_stride = (int64_t)blockDim.x * gridDim.x;

    // ── Vectorized body: 8 bf16 per thread per iter ──
    for (int64_t vec = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         vec < n_vec; vec += grid_stride)
    {
        int64_t idx = vec * ELEMS_PER_THREAD;

        uint4 g_raw = ld_global_cs_v4(&G [idx]);
        uint4 w_raw = ld_global_cs_v4(&W [idx]);
        uint4 m_raw = ld_global_cs_v4(&M [idx]);
        uint4 v_raw = ld_global_cs_v4(&VB[idx]);

        auto* gb = reinterpret_cast<__nv_bfloat16*>(&g_raw);
        auto* wb = reinterpret_cast<__nv_bfloat16*>(&w_raw);
        auto* mb = reinterpret_cast<__nv_bfloat16*>(&m_raw);
        auto* vb = reinterpret_cast<__nv_bfloat16*>(&v_raw);

        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            float g  = __bfloat162float(gb[e]);
            float wf = __bfloat162float(wb[e]);
            float mf = __bfloat162float(mb[e]);
            float vf = __bfloat162float(vb[e]);

            mf = beta1 * mf + one_minus_b1 * g;
            vf = beta2 * vf + one_minus_b2 * g * g;

            float mhat = mf * inv_bc1;
            float vhat = vf * inv_bc2;
            wf = wf * one_minus_lr_wd
               - lr * mhat / (sqrtf(vhat) + eps);

            wb[e] = __float2bfloat16(wf);
            mb[e] = __float2bfloat16(mf);
            vb[e] = __float2bfloat16(vf);
        }

        st_global_cs_v4(&W [idx], w_raw);
        st_global_cs_v4(&M [idx], m_raw);
        st_global_cs_v4(&VB[idx], v_raw);
    }

    // ── Scalar tail (only when total_elems % 8 != 0 — never for LLaMA) ──
    if (blockIdx.x == 0) {
        for (int64_t i = tail_start + threadIdx.x; i < total_elems; i += blockDim.x) {
            float g  = __bfloat162float(G [i]);
            float wf = __bfloat162float(W [i]);
            float mf = __bfloat162float(M [i]);
            float vf = __bfloat162float(VB[i]);

            mf = beta1 * mf + one_minus_b1 * g;
            vf = beta2 * vf + one_minus_b2 * g * g;
            float mhat = mf * inv_bc1;
            float vhat = vf * inv_bc2;
            wf = wf * one_minus_lr_wd - lr * mhat / (sqrtf(vhat) + eps);

            W [i] = __float2bfloat16(wf);
            M [i] = __float2bfloat16(mf);
            VB[i] = __float2bfloat16(vf);
        }
    }
}


void optimizer_only_adamw_cuda(
    torch::Tensor grad,       // (V, H) bf16, read only
    torch::Tensor weight,     // (V, H) bf16, in-place
    torch::Tensor m,          // (V, H) bf16, in-place
    torch::Tensor v,          // (V, H) bf16, in-place
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2
) {
    TORCH_CHECK(grad.is_cuda()   && grad.scalar_type()   == torch::kBFloat16 && grad.is_contiguous(),
                "grad must be contiguous bf16 CUDA");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kBFloat16 && weight.is_contiguous(),
                "weight must be contiguous bf16 CUDA");
    TORCH_CHECK(m.is_cuda()      && m.scalar_type()      == torch::kBFloat16 && m.is_contiguous(),
                "m must be contiguous bf16 CUDA");
    TORCH_CHECK(v.is_cuda()      && v.scalar_type()      == torch::kBFloat16 && v.is_contiguous(),
                "v must be contiguous bf16 CUDA");

    const int64_t total_elems = weight.numel();
    TORCH_CHECK(grad.numel() == total_elems, "grad shape mismatch");
    TORCH_CHECK(m.numel()    == total_elems, "m shape mismatch");
    TORCH_CHECK(v.numel()    == total_elems, "v shape mismatch");

    // Launch one wave per SM × ~4 tiles of work, bounded by tensor size.
    int dev; cudaGetDevice(&dev);
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);

    int64_t n_vec = total_elems / ELEMS_PER_THREAD;
    int64_t blocks_needed = (n_vec + NT - 1) / NT;
    int64_t cap = (int64_t)nsm * 8;                // at most 8 waves
    int blocks = (int)std::min(blocks_needed, cap);
    if (blocks < 1) blocks = 1;

    float inv_bc1 = 1.0f / bc1;
    float inv_bc2 = 1.0f / bc2;

    optimizer_only_adamw_kernel<<<blocks, NT>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(m.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
        total_elems,
        lr, beta1, beta2, eps, wd,
        inv_bc1, inv_bc2
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optimizer_only_adamw_cuda", &optimizer_only_adamw_cuda,
          "Vectorized 128-bit AdamW optimizer-only step (in-place W/M/V)");
}
