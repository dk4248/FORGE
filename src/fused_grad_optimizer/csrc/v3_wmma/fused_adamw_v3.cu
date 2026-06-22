/*
 * Fused grad_W + AdamW — fully-pipelined WMMA kernel for sm_120.
 *
 * Design:
 *   GO is (BT, V) row-major → V is contiguous.
 *   A in SMEM stored col-major: sa_col[k * BV_LD + m], M contiguous.
 *   cp.async copies GO[bt, v..v+7] → sa_col[bt*BV_LD + v..v+7].
 *   WMMA with col_major A reads directly from this layout.
 *
 * Pipeline: 2-stage cp.async for A + B, overlapping loads with WMMA compute.
 *
 * Optimizations:
 *   1. SMEM padding (+8 elements leading dim) to eliminate bank conflicts
 *      on wmma::load_matrix_sync. BV_LD=136 shifts consecutive K-rows to
 *      different 4-byte banks so 32 lanes never serialize.
 *   2. Vectorized 128-bit (uint4) loads/stores for W/M/V in the epilogue.
 *      Each lane handles 8 bf16 contiguously — 1 vector op instead of 8
 *      scalar ops per tensor.
 *   3. W/M/V loads issued before AdamW compute so HBM latency overlaps
 *      with the bf16→fp32 conversion and Adam math.
 *   4. 2-stage cp.async pipeline (BBT=64). One chunk's load is always in
 *      flight while the other is being consumed by the WMMA loop.
 *   5. Precomputed inv_bc1 / inv_bc2 / (1 - lr*wd).
 *
 * SMEM:  2 × (A: 64*136*2=17408 + B: 64*136*2=17408) ≈ 68 KB  (< 99 KB).
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>

using namespace nvcuda;

constexpr int BV  = 128;
constexpr int BH  = 128;
constexpr int BBT = 64;
constexpr int GSV = 32;
constexpr int NT  = 256;
constexpr int NSTAGES = 2;

constexpr int WM = 16, WN = 16, WK = 16;
constexpr int NV_TILES = BV / WM;          // 8
constexpr int NH_TILES = BH / WN;          // 8
constexpr int NK_TILES = BBT / WK;         // 4

// Padded leading dims: +8 bf16 = +16 bytes shifts each K-row by one bank group,
// eliminating bank conflicts during wmma::load_matrix_sync.
constexpr int BV_LD = BV + 8;              // 136 bf16 leading dim for A (col-major)
constexpr int BH_LD = BH + 8;              // 136 bf16 leading dim for B (row-major)

constexpr int SA = BBT * BV_LD;            // col-major A: sa[k * BV_LD + m]
constexpr int SB = BBT * BH_LD;            // row-major B: sb[k * BH_LD + n]
constexpr int SBUF = SA + SB;
constexpr int SMEM_PIPE_BYTES = NSTAGES * SBUF * (int)sizeof(__nv_bfloat16);

// Epilogue reuses SMEM for fp32 WMMA accumulator fragment store.
// 8 warps × 16×16 fp32 = 8 KB. Reuses pipeline SMEM after the matmul completes.
constexpr int EPI_SMEM_BYTES = NH_TILES * WM * WN * (int)sizeof(float);
constexpr int SMEM_BYTES = (SMEM_PIPE_BYTES > EPI_SMEM_BYTES)
                            ? SMEM_PIPE_BYTES : EPI_SMEM_BYTES;

// ── cp.async helpers ──────────────────────────────────────────────────────
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void cp_async_wait_0() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}
template<int N>
__device__ __forceinline__ void cp_async_wait_group_n() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// 16-byte cp.async: 8 × bf16 from global to SMEM in one transaction.
__device__ __forceinline__ void cp_async_16B(uint32_t smem_addr,
                                             const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr));
}


// ── Fully async load stage: cp.async for BOTH A and B ────────────────────
__device__ __forceinline__ void load_stage_full_async(
    __nv_bfloat16* __restrict__ sa_col,   // col-major: sa_col[k * BV_LD + m]
    __nv_bfloat16* __restrict__ sb,        // row-major: sb  [k * BH_LD + n]
    const __nv_bfloat16* __restrict__ GO,
    const __nv_bfloat16* __restrict__ INP,
    int tv, int th, int bt_start,
    int BT, int V, int H,
    int go_stride, int in_stride, int tid
) {
    constexpr int VEC = 8;  // 16 bytes / 2 bytes per bf16

    // ── A: cp.async GO[bt, v..v+7] → sa_col[bt*BV_LD + v..v+7] ──
    constexpr int A_VEC_PER_ROW = BV / VEC;                 // 16
    constexpr int A_TOTAL_VEC   = BBT * A_VEC_PER_ROW;      // 1024

    #pragma unroll 4
    for (int i = tid; i < A_TOTAL_VEC; i += NT) {
        int btl     = i / A_VEC_PER_ROW;
        int v_chunk = i % A_VEC_PER_ROW;
        int gbt     = bt_start + btl;
        int gv      = tv + v_chunk * VEC;

        __nv_bfloat16* dst = &sa_col[btl * BV_LD + v_chunk * VEC];
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

        if (gbt < BT && gv + VEC <= V) {
            cp_async_16B(smem_addr, &GO[gbt * go_stride + gv]);
        } else {
            #pragma unroll
            for (int e = 0; e < VEC; e++) {
                int gve = gv + e;
                dst[e] = (gbt < BT && gve < V) ? GO[gbt * go_stride + gve]
                                               : __float2bfloat16(0.0f);
            }
        }
    }

    // ── B: cp.async INP[bt, h..h+7] → sb[bt*BH_LD + h..h+7] ──
    constexpr int B_VEC_PER_ROW = BH / VEC;
    constexpr int B_TOTAL_VEC   = BBT * B_VEC_PER_ROW;

    #pragma unroll 4
    for (int i = tid; i < B_TOTAL_VEC; i += NT) {
        int btl     = i / B_VEC_PER_ROW;
        int h_chunk = i % B_VEC_PER_ROW;
        int gbt     = bt_start + btl;
        int gh      = th + h_chunk * VEC;

        __nv_bfloat16* dst = &sb[btl * BH_LD + h_chunk * VEC];
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));

        if (gbt < BT && gh + VEC <= H) {
            cp_async_16B(smem_addr, &INP[gbt * in_stride + gh]);
        } else {
            #pragma unroll
            for (int e = 0; e < VEC; e++) {
                int ghe = gh + e;
                dst[e] = (gbt < BT && ghe < H) ? INP[gbt * in_stride + ghe]
                                               : __float2bfloat16(0.0f);
            }
        }
    }
}


// ── Vectorized 128-bit global loads/stores (cache-streaming) ─────────────
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
fused_grad_adamw_v3_kernel(
    const __nv_bfloat16* __restrict__ GO,
    const __nv_bfloat16* __restrict__ INP,
    __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ MB,
    __nv_bfloat16* __restrict__ VB,
    int BT, int V, int H,
    int go_stride, int in_stride, int w_stride,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2,
    int num_sms
) {
    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lane = tid % 32;
    const int ntv  = (V + BV - 1) / BV;
    const int nth  = (H + BH - 1) / BH;
    const int total_tiles = ntv * nth;

    // Hoisted AdamW constants — same value for every element.
    const float one_minus_lr_wd = 1.0f - lr * wd;
    const float inv_bc1         = 1.0f / bc1;
    const float inv_bc2         = 1.0f / bc2;
    const float one_minus_b1    = 1.0f - beta1;
    const float one_minus_b2    = 1.0f - beta2;

    extern __shared__ __nv_bfloat16 smem[];
    auto sa_col = [&](int stage) -> __nv_bfloat16* { return smem + stage * SBUF; };
    auto sb     = [&](int stage) -> __nv_bfloat16* { return smem + stage * SBUF + SA; };

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[NV_TILES];

    for (int tile = blockIdx.x; tile < total_tiles; tile += num_sms) {
        // Grouped tile ordering for L2 reuse of GO/INP across V-tiles.
        int grp_n = GSV * nth;
        int gid   = tile / grp_n;
        int fpv   = gid * GSV;
        int gsz   = min(ntv - fpv, GSV);
        int pv    = fpv + ((tile % grp_n) % gsz);
        int ph    = (tile % grp_n) / gsz;
        int tv    = pv * BV;
        int th    = ph * BH;

        #pragma unroll
        for (int i = 0; i < NV_TILES; i++)
            wmma::fill_fragment(acc[i], 0.0f);

        int num_chunks = (BT + BBT - 1) / BBT;

        // ── Pipeline prologue: fill both stages with async loads ──
        load_stage_full_async(sa_col(0), sb(0), GO, INP, tv, th, 0,
                              BT, V, H, go_stride, in_stride, tid);
        cp_async_commit();

        if (num_chunks > 1) {
            load_stage_full_async(sa_col(1), sb(1), GO, INP, tv, th, BBT,
                                  BT, V, H, go_stride, in_stride, tid);
            cp_async_commit();
        }

        // ── Main loop ──
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int stage = chunk % NSTAGES;

            // Wait for this chunk's data to be resident in SMEM.
            if (chunk == 0) {
                if (num_chunks > 1) cp_async_wait_group_n<1>();
                else                 cp_async_wait_0();
            } else {
                cp_async_wait_group_n<1>();
            }
            __syncthreads();

            // Prefetch chunk `next` into the slot we'll reuse.
            int next = chunk + NSTAGES;
            if (next < num_chunks) {
                load_stage_full_async(
                    sa_col(next % NSTAGES), sb(next % NSTAGES),
                    GO, INP, tv, th, next * BBT,
                    BT, V, H, go_stride, in_stride, tid);
                cp_async_commit();
            }

            __nv_bfloat16* sa_cur = sa_col(stage);
            __nv_bfloat16* sb_cur = sb(stage);

            // ── WMMA matmul over BBT for the current chunk ──
            if (wid < NH_TILES) {
                #pragma unroll
                for (int kt = 0; kt < NK_TILES; kt++) {
                    wmma::fragment<wmma::matrix_b, WM, WN, WK,
                                   __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(bf,
                        sb_cur + kt * WK * BH_LD + wid * WN, BH_LD);

                    #pragma unroll
                    for (int vt = 0; vt < NV_TILES; vt++) {
                        wmma::fragment<wmma::matrix_a, WM, WN, WK,
                                       __nv_bfloat16, wmma::col_major> af;
                        wmma::load_matrix_sync(af,
                            sa_cur + kt * WK * BV_LD + vt * WM, BV_LD);

                        wmma::mma_sync(acc[vt], af, bf, acc[vt]);
                    }
                }
            }
            __syncthreads();
        }

        cp_async_wait_0();
        __syncthreads();

        // ── AdamW epilogue (128-bit vectorized) ──
        // Per lane layout: row = lane/2, col = (lane%2) * 8 → 8 consecutive
        // bf16 elements per lane. 32 lanes × 8 = 256 = one 16×16 WMMA tile.
        if (wid < NH_TILES) {
            float* acc_smem_warp = reinterpret_cast<float*>(smem)
                                 + wid * WM * WN;

            const int r       = lane >> 1;
            const int c       = (lane & 1) << 3;                // 0 or 8
            const int base_gh = th + wid * WN + c;

            // Fast path: H is a multiple of 8 AND this lane's 8-element chunk
            // is fully in-bounds. True for every tile of LLaMA MLP & lm_head.
            const bool h_in_bounds = (base_gh + 8 <= H);

            #pragma unroll
            for (int vt = 0; vt < NV_TILES; vt++) {
                wmma::store_matrix_sync(acc_smem_warp, acc[vt], WN,
                                        wmma::mem_row_major);
                __syncwarp();

                int gv  = tv + vt * WM + r;
                int idx = gv * w_stride + base_gh;

                // Read 8 fp32 grads from SMEM (two 128-bit loads).
                float4 g0 = *reinterpret_cast<float4*>(
                    &acc_smem_warp[r * WN + c]);
                float4 g1 = *reinterpret_cast<float4*>(
                    &acc_smem_warp[r * WN + c + 4]);
                float gv_arr[8] = { g0.x, g0.y, g0.z, g0.w,
                                     g1.x, g1.y, g1.z, g1.w };

                if (gv < V && h_in_bounds) {
                    // Issue all three global loads up-front so their HBM
                    // round-trip overlaps with the bf16→fp32 conversion and
                    // subsequent AdamW math below.
                    uint4 w_raw = ld_global_cs_v4(&W [idx]);
                    uint4 m_raw = ld_global_cs_v4(&MB[idx]);
                    uint4 v_raw = ld_global_cs_v4(&VB[idx]);

                    __nv_bfloat16* wb = reinterpret_cast<__nv_bfloat16*>(&w_raw);
                    __nv_bfloat16* mb = reinterpret_cast<__nv_bfloat16*>(&m_raw);
                    __nv_bfloat16* vb = reinterpret_cast<__nv_bfloat16*>(&v_raw);

                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        float g  = gv_arr[e];
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
                    st_global_cs_v4(&MB[idx], m_raw);
                    st_global_cs_v4(&VB[idx], v_raw);
                } else if (gv < V) {
                    // Boundary fallback: element-wise for the tail.
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        int gh = base_gh + e;
                        if (gh >= H) break;
                        int idx_e = gv * w_stride + gh;
                        float g  = gv_arr[e];
                        float wf = __bfloat162float(W [idx_e]);
                        float mf = __bfloat162float(MB[idx_e]);
                        float vf = __bfloat162float(VB[idx_e]);

                        mf = beta1 * mf + one_minus_b1 * g;
                        vf = beta2 * vf + one_minus_b2 * g * g;

                        float mhat = mf * inv_bc1;
                        float vhat = vf * inv_bc2;
                        wf = wf * one_minus_lr_wd
                           - lr * mhat / (sqrtf(vhat) + eps);

                        W [idx_e] = __float2bfloat16(wf);
                        MB[idx_e] = __float2bfloat16(mf);
                        VB[idx_e] = __float2bfloat16(vf);
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }
}


void fused_grad_adamw_v3(
    torch::Tensor grad_output, torch::Tensor input,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2
) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.scalar_type() == torch::kBFloat16);
    int BT = grad_output.size(0), V = grad_output.size(1), H = input.size(1);
    int dev; cudaGetDevice(&dev);
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);

    cudaFuncSetAttribute(fused_grad_adamw_v3_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    // Prefer SMEM over L1 carveout: we use 68 KB of dynamic SMEM, worth it.
    cudaFuncSetAttribute(fused_grad_adamw_v3_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    fused_grad_adamw_v3_kernel<<<nsm, NT, SMEM_BYTES>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(m.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(v.data_ptr()),
        BT, V, H,
        (int)grad_output.stride(0), (int)input.stride(0), (int)weight.stride(0),
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2, nsm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_grad_adamw_v3", &fused_grad_adamw_v3,
          "Fused grad_W + AdamW v3 (padded SMEM + cp.async pipeline + "
          "128-bit vectorized epilogue + WMMA)");
}
