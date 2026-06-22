/*
 * Hopper-native bf16 GEMM for grad_W = GO^T @ INP on sm_90a (H200).
 *
 * Uses CUTLASS 3.x CollectiveBuilder with ArchTag = Sm90, letting the builder
 * pick KernelTmaWarpSpecialized[Cooperative] + TmaWarpSpecializedCooperative
 * epilogue. This unlocks:
 *   * WGMMA  (warp-group async tensor-core instructions) instead of mma.sync
 *   * TMA    (Tensor Memory Accelerator) instead of cp.async for SMEM fills
 *   * Persistent-scheduler tiling (132 CTAs -> one per SM on H200)
 *
 * This is the "step 1" path: a fast GEMM that materialises grad_W in a bf16
 * temp buffer. The caller then runs optimizer_only_adamw to apply the AdamW
 * update. We do NOT port the custom Sm80 EVT visitor to Sm90's Sm90EVT
 * callbacks framework here (much larger change); the dominant cost is the
 * GEMM itself, which we're accelerating via WGMMA+TMA.
 *
 * Shape mapping (GEMM D = A @ B):
 *   A = GO^T   (M=V, K=BT)  ColumnMajor   (V contiguous, BT strided)
 *   B = INP    (K=BT, N=H)  RowMajor      (H contiguous, BT strided)
 *   D = grad_W (M=V, N=H)   RowMajor      (H contiguous)
 *
 * Tile variants (ClusterShape fixed at 2×1×1 — tested best on H200 for these
 * shapes; MMA cluster of 2 CTAs lets TMA multicast A across the cluster):
 *   128×128×64  — baseline, balanced
 *   128×256×64  — wide-N, best for down_proj (H >> V)
 *   256×128×64  — wide-M, best for lm_head + gate/up_proj (V >> H)
 *   64×128×64   — more tiles, helps when M is small
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


// ── Template parametric in TileShape / ClusterShape ──────────────────────
template <class TileShape_, class ClusterShape_>
struct HopperGemm {
    using TileShape    = TileShape_;
    using ClusterShape = ClusterShape_;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,          // (M, N, K, L)
        CollectiveMainloop,
        CollectiveEpilogue,
        cutlass::gemm::PersistentScheduler
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

        // packed_stride handles the CuTe stride tuple construction for each layout.
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
                    "Hopper GEMM cannot implement problem (status ",
                    cutlassGetStatusString(can), ")");

        size_t workspace_size = Gemm::get_workspace_size(args);
        auto workspace = torch::empty(
            {(int64_t)workspace_size},
            torch::dtype(torch::kUInt8).device(go.device()));

        auto init = gemm.initialize(args, workspace.data_ptr());
        TORCH_CHECK(init == cutlass::Status::kSuccess,
                    "Hopper GEMM init failed (status ",
                    cutlassGetStatusString(init), ")");

        auto run = gemm.run();
        TORCH_CHECK(run == cutlass::Status::kSuccess,
                    "Hopper GEMM run failed (status ",
                    cutlassGetStatusString(run), ")");
    }
};


// ── Tile variants ────────────────────────────────────────────────────────
// ClusterShape<2,1,1>: 2-CTA cluster lets TMA multicast operand A across
// both CTAs. On H200 (132 SMs), 66 clusters can run concurrently.
using Cluster2x1 = Shape<_2,_1,_1>;

using HopperGemm_128x128 = HopperGemm<Shape<_128,_128,_64>, Cluster2x1>;
using HopperGemm_128x256 = HopperGemm<Shape<_128,_256,_64>, Cluster2x1>;
using HopperGemm_256x128 = HopperGemm<Shape<_256,_128,_64>, Cluster2x1>;
using HopperGemm_64x128  = HopperGemm<Shape<_64, _128,_64>, Shape<_1,_1,_1>>;


// ── Python-facing entry points ───────────────────────────────────────────
void hopper_grad_w_bf16_128x128(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    HopperGemm_128x128::launch(go, inp, grad_w);
}
void hopper_grad_w_bf16_128x256(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    HopperGemm_128x256::launch(go, inp, grad_w);
}
void hopper_grad_w_bf16_256x128(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    HopperGemm_256x128::launch(go, inp, grad_w);
}
void hopper_grad_w_bf16_64x128(torch::Tensor go, torch::Tensor inp, torch::Tensor grad_w) {
    HopperGemm_64x128::launch(go, inp, grad_w);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hopper_grad_w_bf16_128x128", &hopper_grad_w_bf16_128x128,
          "Hopper WGMMA+TMA bf16 grad_W (128x128x64 tile, cluster 2x1)");
    m.def("hopper_grad_w_bf16_128x256", &hopper_grad_w_bf16_128x256,
          "Hopper WGMMA+TMA bf16 grad_W (128x256x64 tile, cluster 2x1)");
    m.def("hopper_grad_w_bf16_256x128", &hopper_grad_w_bf16_256x128,
          "Hopper WGMMA+TMA bf16 grad_W (256x128x64 tile, cluster 2x1)");
    m.def("hopper_grad_w_bf16_64x128", &hopper_grad_w_bf16_64x128,
          "Hopper WGMMA+TMA bf16 grad_W (64x128x64 tile, cluster 1x1)");
}
