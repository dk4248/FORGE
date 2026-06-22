/*
 * Fused grad_W + AdamW — CUDA kernel (final version).
 *
 * Same algorithm as the Triton _fused_grad_adamw_persistent kernel:
 *   - Persistent grid (1 CTA per SM)
 *   - Tile-by-tile: grad_tile = GO.T_tile @ INP_tile, then AdamW(W, m, v, grad)
 *   - Gradient never materialized in HBM
 *   - Grouped tile ordering for L2 reuse
 *   - Streaming cache hints (ld.cs/st.cs) for W/m/v
 *
 * Matmul uses WMMA bf16 tensor cores.
 * NOTE: On Blackwell sm_120, WMMA does NOT use the native tcgen05 tensor cores
 * (those require CUTLASS 3.x collective builders or Triton). WMMA falls back
 * to a slower path. This kernel exists to show the fused algorithm in CUDA
 * and compare against Triton's tcgen05-based implementation.
 *
 * SMEM: double-buffered (BBT×BV + BBT×BH) × 2 = 64 KB
 * grad_output loaded transposed into SMEM for row_major WMMA.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
// No cp.async pipeline — __pipeline_memcpy_async requires min 4 bytes,
// but bf16 is 2 bytes. Using synchronous loads + __syncthreads() instead.
#include <mma.h>
#include <math.h>

using namespace nvcuda;

// Tile config (same as Triton persistent kernel)
constexpr int BV  = 128;   // tile rows (V dimension)
constexpr int BH  = 128;   // tile cols (H dimension)
constexpr int BBT = 64;    // inner loop batch chunk
constexpr int GSV = 32;    // GROUP_SIZE_V for L2 reuse
constexpr int NT  = 256;   // 8 warps × 32 lanes

// WMMA dimensions
constexpr int WM = 16, WN = 16, WK = 16;
constexpr int NV_TILES = BV / WM;    // 8
constexpr int NH_TILES = BH / WN;    // 8
constexpr int NK_TILES = BBT / WK;   // 4

// SMEM: buf_a=(BV,BBT) + buf_b=(BBT,BH), double-buffered
constexpr int SA = BV * BBT;         // 8192 elements
constexpr int SB = BBT * BH;         // 8192 elements
constexpr int SBUF = SA + SB;        // 16384 elements per buffer
constexpr int SMEM_BYTES = 2 * SBUF * sizeof(__nv_bfloat16);  // 64 KB


__global__ void __launch_bounds__(NT)
fused_grad_adamw_cuda_kernel(
    const __nv_bfloat16* __restrict__ GO,   // (BT, V) grad_output
    const __nv_bfloat16* __restrict__ INP,  // (BT, H) input
    __nv_bfloat16* __restrict__ W,          // (V, H)  weight
    __nv_bfloat16* __restrict__ MB,         // (V, H)  first moment
    __nv_bfloat16* __restrict__ VB,         // (V, H)  second moment
    int BT, int V, int H,
    int go_stride,    // V (grad_output row stride)
    int in_stride,    // H (input row stride)
    int w_stride,     // H (weight row stride)
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2,
    int num_sms
) {
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int ntv = (V + BV - 1) / BV;
    const int nth = (H + BH - 1) / BH;
    const int total_tiles = ntv * nth;

    extern __shared__ __nv_bfloat16 smem[];
    // Buffer pointers
    __nv_bfloat16* sa0 = smem;                  // buf 0: A (BV, BBT) transposed grad_output
    __nv_bfloat16* sb0 = smem + SA;             // buf 0: B (BBT, BH) input
    __nv_bfloat16* sa1 = smem + SBUF;           // buf 1: A
    __nv_bfloat16* sb1 = smem + SBUF + SA;      // buf 1: B

    // Per-warp accumulators: each warp handles 1 H-column × all 8 V-rows
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[NV_TILES];

    // ── Persistent tile loop ──
    for (int tile = blockIdx.x; tile < total_tiles; tile += num_sms) {
        // Grouped tile decomposition for L2 reuse
        int grp_n = GSV * nth;
        int gid = tile / grp_n;
        int fpv = gid * GSV;
        int gsz = min(ntv - fpv, GSV);
        int pv = fpv + ((tile % grp_n) % gsz);
        int ph = (tile % grp_n) / gsz;
        int tv = pv * BV;
        int th = ph * BH;

        // Zero accumulators
        #pragma unroll
        for (int i = 0; i < NV_TILES; i++)
            wmma::fill_fragment(acc[i], 0.0f);

        // ── Double-buffered BT inner loop with cp.async ──
        // Load first chunk into buf 0
        // A: transpose grad_output into (BV, BBT)
        //   sa[v_local * BBT + bt_local] = GO[(0 + bt_local) * go_stride + (tv + v_local)]
        {
            __nv_bfloat16* sa = sa0;
            __nv_bfloat16* sb = sb0;
            int n_a = BV * BBT;
            for (int i = tid; i < n_a; i += NT) {
                int vl = i / BBT, btl = i % BBT;
                int gbt = btl, gv = tv + vl;
                if (gbt < BT && gv < V)
                    sa[i] = GO[gbt * go_stride + gv];
                else
                    sa[i] = __float2bfloat16(0.0f);
            }
            int n_b = BBT * BH;
            for (int i = tid; i < n_b; i += NT) {
                int btl = i / BH, hl = i % BH;
                int gbt = btl, gh = th + hl;
                if (gbt < BT && gh < H)
                    sb[i] = INP[gbt * in_stride + gh];
                else
                    sb[i] = __float2bfloat16(0.0f);
            }
        }
        // (sync-based, no async pipeline)

        for (int bt = 0; bt < BT; bt += BBT) {
            int next_bt = bt + BBT;
            int cur = (bt / BBT) & 1;

            // Start loading next chunk into other buffer
            if (next_bt < BT) {
                __nv_bfloat16* sa_next = (cur == 0) ? sa1 : sa0;
                __nv_bfloat16* sb_next = (cur == 0) ? sb1 : sb0;
                int n_a = BV * BBT;
                for (int i = tid; i < n_a; i += NT) {
                    int vl = i / BBT, btl = i % BBT;
                    int gbt = next_bt + btl, gv = tv + vl;
                    if (gbt < BT && gv < V)
                        sa_next[i] = GO[gbt * go_stride + gv];
                    else
                        sa_next[i] = __float2bfloat16(0.0f);
                }
                int n_b = BBT * BH;
                for (int i = tid; i < n_b; i += NT) {
                    int btl = i / BH, hl = i % BH;
                    int gbt = next_bt + btl, gh = th + hl;
                    if (gbt < BT && gh < H)
                        sb_next[i] = INP[gbt * in_stride + gh];
                    else
                        sb_next[i] = __float2bfloat16(0.0f);
                }
                // (sync-based, no async pipeline)
            }

            __syncthreads();

            __nv_bfloat16* sa_cur = (cur == 0) ? sa0 : sa1;
            __nv_bfloat16* sb_cur = (cur == 0) ? sb0 : sb1;

            // WMMA matmul: A(BV,BBT) row_major @ B(BBT,BH) row_major
            // Each warp handles H-column wid, all V-rows
            if (wid < NH_TILES) {
                #pragma unroll
                for (int kt = 0; kt < NK_TILES; kt++) {
                    wmma::fragment<wmma::matrix_b, WM, WN, WK,
                                   __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(bf,
                        sb_cur + kt * WK * BH + wid * WN, BH);

                    #pragma unroll
                    for (int vt = 0; vt < NV_TILES; vt++) {
                        wmma::fragment<wmma::matrix_a, WM, WN, WK,
                                       __nv_bfloat16, wmma::row_major> af;
                        wmma::load_matrix_sync(af,
                            sa_cur + vt * WM * BBT + kt * WK, BBT);
                        wmma::mma_sync(acc[vt], af, bf, acc[vt]);
                    }
                }
            }
            __syncthreads();
        }

        // ── AdamW epilogue: store acc → SMEM, read per-element, update W/m/v ──
        if (wid < NH_TILES) {
            // Reuse SMEM for accumulator staging
            float* acc_smem = reinterpret_cast<float*>(smem);
            // Each warp gets 256 floats (16×16) at offset wid*256

            #pragma unroll
            for (int vt = 0; vt < NV_TILES; vt++) {
                wmma::store_matrix_sync(
                    acc_smem + wid * WM * WN,
                    acc[vt], WN, wmma::mem_row_major);
                __syncwarp();

                int base_v = tv + vt * WM;
                int base_h = th + wid * WN;

                // Each lane processes 256/32 = 8 elements of the 16×16 tile
                for (int e = lane; e < WM * WN; e += 32) {
                    int r = e / WN;
                    int c = e % WN;
                    int gv = base_v + r;
                    int gh = base_h + c;

                    if (gv < V && gh < H) {
                        float g = acc_smem[wid * WM * WN + r * WN + c];
                        int idx = gv * w_stride + gh;

                        // Streaming load (ld.cs = cache streaming = evict first)
                        unsigned short wr, mr, vr;
                        asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(wr) : "l"(&W[idx]));
                        asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(mr) : "l"(&MB[idx]));
                        asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(vr) : "l"(&VB[idx]));

                        float wf = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&wr));
                        float mf = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&mr));
                        float vf = __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&vr));

                        mf = beta1 * mf + (1.0f - beta1) * g;
                        vf = beta2 * vf + (1.0f - beta2) * g * g;
                        float mh = mf / bc1;
                        float vh = vf / bc2;
                        wf = wf * (1.0f - lr * wd) - lr * mh / (sqrtf(vh) + eps);

                        __nv_bfloat16 wo = __float2bfloat16(wf);
                        __nv_bfloat16 mo = __float2bfloat16(mf);
                        __nv_bfloat16 vo = __float2bfloat16(vf);
                        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(&W[idx]),
                            "h"(*reinterpret_cast<unsigned short*>(&wo)));
                        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(&MB[idx]),
                            "h"(*reinterpret_cast<unsigned short*>(&mo)));
                        asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(&VB[idx]),
                            "h"(*reinterpret_cast<unsigned short*>(&vo)));
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }
}


// ---------------------------------------------------------------------------
// Python wrapper
// ---------------------------------------------------------------------------
void fused_grad_adamw_cutlass(
    torch::Tensor grad_output, torch::Tensor input,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2
) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.scalar_type() == torch::kBFloat16);
    int BT = grad_output.size(0), V = grad_output.size(1), H = input.size(1);
    int dev; cudaGetDevice(&dev);
    int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);

    cudaFuncSetAttribute(fused_grad_adamw_cuda_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

    fused_grad_adamw_cuda_kernel<<<nsm, NT, SMEM_BYTES>>>(
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
    m.def("fused_grad_adamw_cutlass", &fused_grad_adamw_cutlass,
          "Fused grad_W + AdamW CUDA kernel (WMMA + cp.async + streaming cache)");
}
