/*
 * Blackwell-native bf16 GEMM for grad_W = GO^T @ INP on sm_100a (B200).
 *
 * Direct port of csrc/hopper_evt/fused_adamw_hopper.cu but targeting the
 * Blackwell collective builder. Uses CUTLASS 4.x's CollectiveBuilder with
 * ArchTag = Sm100, which with KernelScheduleAuto selects:
 *   * tcgen05.mma  (UMMA — the only bf16 tensor-core MMA on Blackwell)
 *   * Tensor Memory (TMEM) for the fp32 accumulator. NOTE: this is
 *     hardware-enforced on sm_100 — there is no register-file accumulator
 *     path like Hopper WGMMA. Each CTA's per-tile fp32 accum
 *     (TileShape_M × TileShape_N elements) lives in TMEM across the
 *     entire K-loop and is only moved to RF at the start of the epilogue.
 *   * TMA loads (A/B) and TMA stores (D)
 *   * PersistentTileSchedulerSm100  (default `void` → builder picks it)
 *   * Producer/consumer warp-specialization. The mainloop schedule
 *     (KernelTmaWarpSpecialized1SmSm100 / 2SmSm100) splits warps into
 *     a TMA-load producer group and a tcgen05.mma consumer group with
 *     a TmaUmma async pipeline between them — this is the structural
 *     WS that CUTLASS expresses directly, distinct from Triton's
 *     `tl.range(warp_specialize=True)` that hits #8932 on Triton 3.6.0.
 *     CollectiveOp lowering bypasses Triton entirely, so #8932 doesn't
 *     apply here.
 *
 * Why a separate file rather than reusing csrc/hopper_evt/...:
 *   The H200 file fixes ArchTag = Sm90, which the Sm90 collective builder
 *   pins to WGMMA (Hopper-only) and a different SMEM/SHEM layout. On
 *   Blackwell that builder either fails to compile or produces a Hopper-
 *   only cubin. The sm_100 builder lives in a separate include path and
 *   needs its own template instantiation.
 *
 * This is the "step 1 fast GEMM" path — same trade-off as the H200 file:
 * we materialise grad_W in a bf16 temp buffer and then run the existing
 * optimizer_only_adamw CUDA kernel for the AdamW step. EVT-fusing the
 * AdamW math into the epilogue (the "true EVT" path) is left for a
 * follow-up; getting a clean Sm100 dense-bf16 GEMM into the benchmark
 * first is more useful right now.
 *
 * Shape mapping (GEMM D = A @ B), identical to H200 file:
 *   A = GO^T   (M=V, K=BT)  ColumnMajor    (V contiguous, BT strided)
 *   B = INP    (K=BT, N=H)  RowMajor       (H contiguous, BT strided)
 *   D = grad_W (M=V, N=H)   RowMajor       (H contiguous)
 *
 * Two tile variants, mirroring the most useful H200 ones:
 *   128×256×64  cluster 1×1×1   — 1SM mode, wide-N, best for down_proj
 *                                  and Q/K/V/O. Per-CTA tile = 128×256.
 *   256×128×64  cluster 2×1×1   — 2SM mode, wide-M. Per-CTA tile = 128×128;
 *                                  the 2-CTA cluster issues a 256×128 atom
 *                                  via tcgen05.mma's 2-CTA mode. Best for
 *                                  lm_head and gate/up_proj.
 *
 * For the Sm100 builder, TileShape is the *MMA atom shape*, not the per-CTA
 * tile. The per-CTA tile is TileShape / AtomThrShape, where AtomThrShape is
 * (2,1,1) for 2SM and (1,1,1) for 1SM. KernelScheduleAuto picks 2SM iff
 * ClusterShape M % 2 == 0.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;


using ElementA           = cutlass::bfloat16_t;
using LayoutA            = cutlass::layout::ColumnMajor;   // A = GO^T : M-major
using ElementB           = cutlass::bfloat16_t;
using LayoutB            = cutlass::layout::RowMajor;      // B = INP  : N-major
using ElementC           = cutlass::bfloat16_t;
using LayoutC            = cutlass::layout::RowMajor;
using ElementD           = cutlass::bfloat16_t;
using LayoutD            = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute     = float;

// 16-byte alignment = 8 bf16 elements = required for TMA.
constexpr int AlignmentA = 16 / sizeof(ElementA);  // 8
constexpr int AlignmentB = 16 / sizeof(ElementB);  // 8
constexpr int AlignmentC = 16 / sizeof(ElementC);  // 8
constexpr int AlignmentD = 16 / sizeof(ElementD);  // 8


// Scheduler choice. void → PersistentTileSchedulerSm100 (default).
// cutlass::gemm::StreamKScheduler enables stream-K decomposition: instead of
// statically assigning whole output tiles to SMs, work is split along K so
// every SM stays busy even when the tile count doesn't divide evenly into
// 148 SMs. Helps shapes like Q/K/V/O (4096×4096 / 256×256 = 256 tiles, 1.7
// waves on 148 SMs → terrible last-wave fill) where the persistent
// scheduler leaves SMs idle.
struct PersistentSched { using type = void; };
struct StreamKSched    { using type = cutlass::gemm::StreamKScheduler; };

// ── Template parametric in TileShape / ClusterShape / Scheduler ──────────
template <class TileShape_, class ClusterShape_, class SchedTag = PersistentSched>
struct BlackwellGemm {
    using TileShape    = TileShape_;
    using ClusterShape = ClusterShape_;
    using SchedulerTag = typename SchedTag::type;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // For Sm100, void → PersistentTileSchedulerSm100; StreamKScheduler →
    // stream-K decomposition (split K-dim work across SMs to fill tail waves).
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,          // (M, N, K, L)
        CollectiveMainloop,
        CollectiveEpilogue,
        SchedulerTag
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    static void launch(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
        TORCH_CHECK(go.is_cuda()     && go.scalar_type()     == torch::kBFloat16 && go.is_contiguous());
        TORCH_CHECK(inp.is_cuda()    && inp.scalar_type()    == torch::kBFloat16 && inp.is_contiguous());
        TORCH_CHECK(grad_w.is_cuda() && grad_w.scalar_type() == torch::kBFloat16 && grad_w.is_contiguous());

        const int BT = (int)go.size(0);
        const int V  = (int)go.size(1);
        const int H  = (int)inp.size(1);
        TORCH_CHECK(inp.size(0) == BT);
        TORCH_CHECK(grad_w.size(0) == V && grad_w.size(1) == H);

        const int M = V, N = H, K = BT, L = 1;

        auto* pA = reinterpret_cast<ElementA const*>(go.data_ptr());
        auto* pB = reinterpret_cast<ElementB const*>(inp.data_ptr());
        auto* pD = reinterpret_cast<ElementD*>(grad_w.data_ptr());

        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, L});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, L});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, L});
        StrideC stride_C = stride_D;

        typename Gemm::Arguments args {
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, L},
            {pA, stride_A, pB, stride_B},
            {
                {1.0f, 0.0f},                  // alpha, beta
                nullptr, stride_C,
                pD,      stride_D
            }
        };

        Gemm gemm;

        auto can = gemm.can_implement(args);
        TORCH_CHECK(can == cutlass::Status::kSuccess,
                    "Blackwell GEMM cannot implement problem (status ",
                    cutlassGetStatusString(can), ")");

        size_t workspace_size = Gemm::get_workspace_size(args);
        auto workspace = torch::empty(
            {(int64_t)workspace_size},
            torch::dtype(torch::kUInt8).device(go.device()));

        auto init = gemm.initialize(args, workspace.data_ptr());
        TORCH_CHECK(init == cutlass::Status::kSuccess,
                    "Blackwell GEMM init failed (status ",
                    cutlassGetStatusString(init), ")");

        auto run = gemm.run();
        TORCH_CHECK(run == cutlass::Status::kSuccess,
                    "Blackwell GEMM run failed (status ",
                    cutlassGetStatusString(run), ")");
    }
};


// ── Tile variants ────────────────────────────────────────────────────────
// Cluster<M,1,1>: tcgen05.mma atom = TileShape, per-CTA tile = TileShape / (M,1,1).
// KernelScheduleAuto picks 2SM mode iff cluster M % 2 == 0.
//
// In every variant the fp32 accumulator lives in TMEM (per CTA, hardware-fixed
// on sm_100). Atom size is what one tcgen05.mma instruction issues; bigger atom
// = fewer instructions = better arithmetic intensity, capped by SMEM/TMEM.
//
// Tile choice on B200 (148 SMs, 192 GB HBM3e, 256 KiB TMEM/SM):
//
//   small  128×128 1SM   per-CTA 128×128, atom 128×128.   For Q/K/V/O
//                                                          (M=N=4096) where
//                                                          big tiles starve
//                                                          the last wave.
//   med    128×256 1SM   per-CTA 128×256, atom 128×256.   Original "wide-N"
//                                                          tile, kept as
//                                                          medium fallback.
//   med    256×128 2SM   per-CTA 128×128, atom 256×128.   Original "wide-M",
//                                                          kept for moderate
//                                                          aspect ratios.
//   big    256×256 2SM   per-CTA 128×256, atom 256×256.   B200's max bf16
//                                                          atom — best for
//                                                          lm_head /
//                                                          gate_proj /
//                                                          down_proj where
//                                                          M*N >> 148*tile.
//   huge   256×256 4×1   per-CTA  64×256, atom 256×256.   4-CTA cluster
//                                                          multicasts B
//                                                          across 4 CTAs;
//                                                          best when N is
//                                                          small vs cluster
//                                                          size. Use only
//                                                          when M >> N.
using Cluster1x1 = Shape<_1,_1,_1>;
using Cluster2x1 = Shape<_2,_1,_1>;
using Cluster4x1 = Shape<_4,_1,_1>;

using BlackwellGemm_128x128_1sm = BlackwellGemm<Shape<_128,_128,_64>, Cluster1x1>;
using BlackwellGemm_128x256_1sm = BlackwellGemm<Shape<_128,_256,_64>, Cluster1x1>;
using BlackwellGemm_256x128_2sm = BlackwellGemm<Shape<_256,_128,_64>, Cluster2x1>;
using BlackwellGemm_256x256_2sm = BlackwellGemm<Shape<_256,_256,_64>, Cluster2x1>;
using BlackwellGemm_256x256_4cl = BlackwellGemm<Shape<_256,_256,_64>, Cluster4x1>;

// Stream-K variants. Useful for shapes where the persistent scheduler's
// static tile assignment leaves a half-empty last wave on the SMs.
using BlackwellGemm_128x128_1sm_streamk = BlackwellGemm<Shape<_128,_128,_64>, Cluster1x1, StreamKSched>;
using BlackwellGemm_256x256_2sm_streamk = BlackwellGemm<Shape<_256,_256,_64>, Cluster2x1, StreamKSched>;

// T1.3 attempt — K=128 atom on 256x256 2SM. Compiled cleanly here (no-EVT
// epilogue is light enough for ≥2 mainloop stages) but **regressed the
// no-EVT row by ~50 TF/s** in benchmarks. The bigger atom doubles per-stage
// A/B SMEM, so auto-carveout drops from 4-5 stages (K=64) to ~2 stages
// (K=128). Halving iteration count didn't compensate for the lost
// producer/consumer pipelining. Removed; keep K=64 as the default atom.


// ── Python-facing entry points ───────────────────────────────────────────
void blackwell_grad_w_bf16_128x128_1sm(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_128x128_1sm::launch(go, inp, grad_w);
}
void blackwell_grad_w_bf16_128x256_1sm(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_128x256_1sm::launch(go, inp, grad_w);
}
void blackwell_grad_w_bf16_256x128_2sm(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_256x128_2sm::launch(go, inp, grad_w);
}
void blackwell_grad_w_bf16_256x256_2sm(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_256x256_2sm::launch(go, inp, grad_w);
}
void blackwell_grad_w_bf16_256x256_4cl(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_256x256_4cl::launch(go, inp, grad_w);
}

// Stream-K variants
void blackwell_grad_w_bf16_128x128_1sm_streamk(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_128x128_1sm_streamk::launch(go, inp, grad_w);
}
void blackwell_grad_w_bf16_256x256_2sm_streamk(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_256x256_2sm_streamk::launch(go, inp, grad_w);
}

// Backwards-compat aliases for callers using the previous names.
void blackwell_grad_w_bf16_128x256(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_128x256_1sm::launch(go, inp, grad_w);
}
void blackwell_grad_w_bf16_256x128(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    BlackwellGemm_256x128_2sm::launch(go, inp, grad_w);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blackwell_grad_w_bf16_128x128_1sm", &blackwell_grad_w_bf16_128x128_1sm,
          "Blackwell bf16 grad_W (atom 128x128x64, cluster 1x1, 1SM)");
    m.def("blackwell_grad_w_bf16_128x256_1sm", &blackwell_grad_w_bf16_128x256_1sm,
          "Blackwell bf16 grad_W (atom 128x256x64, cluster 1x1, 1SM)");
    m.def("blackwell_grad_w_bf16_256x128_2sm", &blackwell_grad_w_bf16_256x128_2sm,
          "Blackwell bf16 grad_W (atom 256x128x64, cluster 2x1, 2SM)");
    m.def("blackwell_grad_w_bf16_256x256_2sm", &blackwell_grad_w_bf16_256x256_2sm,
          "Blackwell bf16 grad_W (atom 256x256x64, cluster 2x1, 2SM) — max-atom for big shapes");
    m.def("blackwell_grad_w_bf16_256x256_4cl", &blackwell_grad_w_bf16_256x256_4cl,
          "Blackwell bf16 grad_W (atom 256x256x64, cluster 4x1) — TMA-multicast across 4 CTAs");
    // Stream-K variants
    m.def("blackwell_grad_w_bf16_128x128_1sm_streamk", &blackwell_grad_w_bf16_128x128_1sm_streamk,
          "Stream-K: atom 128x128x64, cluster 1x1, 1SM");
    m.def("blackwell_grad_w_bf16_256x256_2sm_streamk", &blackwell_grad_w_bf16_256x256_2sm_streamk,
          "Stream-K: atom 256x256x64, cluster 2x1, 2SM");
    // Aliases for the previous public names.
    m.def("blackwell_grad_w_bf16_128x256", &blackwell_grad_w_bf16_128x256, "alias of 128x256_1sm");
    m.def("blackwell_grad_w_bf16_256x128", &blackwell_grad_w_bf16_256x128, "alias of 256x128_2sm");
}
