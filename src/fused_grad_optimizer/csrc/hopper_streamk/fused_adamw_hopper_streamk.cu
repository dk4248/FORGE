/*
 * Hopper-native bf16 GEMM for grad_W = GO^T @ INP on sm_90a (H200),
 * Stream-K variant.
 *
 * Differs from csrc/hopper_evt/fused_adamw_hopper.cu only in that the tile
 * scheduler is `cutlass::gemm::StreamKScheduler` instead of
 * `cutlass::gemm::PersistentScheduler`.
 *
 * Why Stream-K for our workload
 * ------------------------------
 * At SEQ_LEN=512, K=BT=512 is tiny.  With TILE_K=64 the GEMM has only 8
 * K-iterations per CTA — not enough to fill Hopper's 3-4-stage TMA+WGMMA
 * pipeline.  The persistent scheduler maps one CTA per output-tile and
 * processes K serially inside the CTA, so every CTA pays ~50% pipeline
 * fill/drain overhead.
 *
 * Stream-K splits the K reduction across multiple CTAs: e.g. a (V, H)
 * output tile can be split into 4 K-slices processed by 4 different CTAs,
 * which each compute a partial accumulator and reduce at the end.  This
 * effectively *quadruples* parallelism when K is short, hiding the
 * pipeline-fill cost and reaching closer to peak throughput.  cuBLAS
 * automatically uses split-K for small K; this is its direct analogue.
 *
 * At long K (SEQ_LEN=4096 etc.) Stream-K gracefully degrades to behave
 * like the persistent scheduler — so it's also a fine default for long
 * sequences, not a short-K-only specialisation.
 */

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/packed_stride.hpp"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

constexpr int AlignmentA = 16 / sizeof(ElementA);  // 8
constexpr int AlignmentB = 16 / sizeof(ElementB);  // 8
constexpr int AlignmentC = 16 / sizeof(ElementC);  // 8
constexpr int AlignmentD = 16 / sizeof(ElementD);  // 8


// ── Template parametric in TileShape / ClusterShape ──────────────────────
template <class TileShape_, class ClusterShape_>
struct HopperGemmStreamK {
    using TileShape    = TileShape_;
    using ClusterShape = ClusterShape_;

    // Stream-K requires a scheduler-compatible kernel schedule.  CUTLASS 3.5.1
    // supports Stream-K with KernelTmaWarpSpecializedCooperative ONLY (ping-pong
    // variant static-asserts against it).  So we pin the schedule explicitly
    // here instead of using KernelScheduleAuto (which picks ping-pong for
    // some tile sizes).
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::TmaWarpSpecializedCooperative
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::CollectiveOp;

    // *** The key change vs hopper_evt: StreamKScheduler instead of Persistent ***
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::StreamKScheduler
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
                    "Hopper Stream-K GEMM cannot implement problem (status ",
                    cutlassGetStatusString(can), ")");

        size_t workspace_size = Gemm::get_workspace_size(args);
        auto workspace = torch::empty(
            {(int64_t)workspace_size},
            torch::dtype(torch::kUInt8).device(go.device()));

        auto init = gemm.initialize(args, workspace.data_ptr());
        TORCH_CHECK(init == cutlass::Status::kSuccess,
                    "Hopper Stream-K GEMM init failed (status ",
                    cutlassGetStatusString(init), ")");

        auto run = gemm.run();
        TORCH_CHECK(run == cutlass::Status::kSuccess,
                    "Hopper Stream-K GEMM run failed (status ",
                    cutlassGetStatusString(run), ")");
    }
};


// ── Tile variants ────────────────────────────────────────────────────────
// Stream-K needs enough K-slices to be useful.  For K=512, it can split
// into up to 4-8 partitions depending on tile size.  Cluster<1,1,1>: stream-K
// with clusters >1 is unsupported in CUTLASS 3.5.1 for some configs.
using Cluster1x1 = Shape<_1,_1,_1>;

// KernelTmaWarpSpecializedCooperative requires CTA tile M >= 128, so no 64x128.
using HopperSK_128x128 = HopperGemmStreamK<Shape<_128,_128,_64>, Cluster1x1>;
using HopperSK_128x256 = HopperGemmStreamK<Shape<_128,_256,_64>, Cluster1x1>;
using HopperSK_256x128 = HopperGemmStreamK<Shape<_256,_128,_64>, Cluster1x1>;


// ── Python-facing entry points ───────────────────────────────────────────
void hopper_streamk_grad_w_bf16_128x128(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    HopperSK_128x128::launch(go, inp, grad_w);
}
void hopper_streamk_grad_w_bf16_128x256(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    HopperSK_128x256::launch(go, inp, grad_w);
}
void hopper_streamk_grad_w_bf16_256x128(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    HopperSK_256x128::launch(go, inp, grad_w);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hopper_streamk_grad_w_bf16_128x128", &hopper_streamk_grad_w_bf16_128x128,
          "Hopper WGMMA+TMA+Stream-K bf16 grad_W (128x128x64)");
    m.def("hopper_streamk_grad_w_bf16_128x256", &hopper_streamk_grad_w_bf16_128x256,
          "Hopper WGMMA+TMA+Stream-K bf16 grad_W (128x256x64)");
    m.def("hopper_streamk_grad_w_bf16_256x128", &hopper_streamk_grad_w_bf16_256x128,
          "Hopper WGMMA+TMA+Stream-K bf16 grad_W (256x128x64)");
}
