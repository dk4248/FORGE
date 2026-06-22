/*
 * CUTLASS 2.x bf16 GEMM for grad_W = GO^T @ INP on sm_120.
 *
 * Why 2.x rather than 3.x:
 *   The 3.x CollectiveBuilder for sm_120 currently only supports F8/F6/F4
 *   MMA and requires TN layout (K-major for both operands). Our memory has
 *   GO with V contiguous and INP with H contiguous, giving NN layout for
 *   GEMM (M-major A, N-major B), which 3.x on sm_120 rejects.
 *
 *   CUTLASS 2.x `cutlass::gemm::device::Gemm` supports arbitrary layouts
 *   and bf16 on sm_80+. sm_120 is tensor-core-compatible with sm_80 (mma.sync
 *   + cp.async path), so a kernel compiled with Sm80 tag + gencode sm_120
 *   runs natively on the device.
 *
 * Shape mapping (GEMM D = A @ B):
 *   A = GO^T   (M=V, K=BT)  ColumnMajor   (V contiguous, BT strided)
 *   B = INP    (K=BT, N=H)  RowMajor      (H contiguous, BT strided)
 *   D = grad_W (M=V, N=H)   RowMajor      (H contiguous)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/device_memory.h"


using ElementA           = cutlass::bfloat16_t;
using LayoutA            = cutlass::layout::ColumnMajor;
using ElementB           = cutlass::bfloat16_t;
using LayoutB            = cutlass::layout::RowMajor;
using ElementC           = cutlass::bfloat16_t;
using LayoutC            = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementComputeEpilogue = float;

// Per-tile access granularity (8 bf16 = 128-bit vectorized memory ops).
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

// Tile geometry: matches our WMMA kernel's BV×BH×BBT for a fair comparison.
// 128×128×32 threadblock, 64×64×32 warp, 16×8×16 MMA (Ampere bf16 tensor core
// instruction shape — runs natively on sm_120 as mma.sync).
using ThreadblockShape   = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape          = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape   = cutlass::gemm::GemmShape<16, 8, 16>;

// Standard linear-combination epilogue: D = alpha * accum + beta * C.
// We pass alpha=1.0, beta=0.0, and ignore C (pass null).
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    AlignmentC,                   // vector length per thread epilogue step
    ElementAccumulator,           // accumulator type (fp32)
    ElementComputeEpilogue        // alpha/beta type (fp32)
>;

// 3 cp.async stages — hides most HBM latency for this tile size.
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,                                          // backward-compatible on sm_120
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB
>;


// ── Python-facing entry point ────────────────────────────────────────────
void cutlass3x_grad_w_bf16(
    torch::Tensor go,        // (BT, V) bf16 row-major
    torch::Tensor inp,       // (BT, H) bf16 row-major
    torch::Tensor grad_w     // (V,  H) bf16 row-major — output
) {
    TORCH_CHECK(go.is_cuda()     && go.scalar_type()     == torch::kBFloat16 && go.is_contiguous(),
                "go must be a contiguous bf16 CUDA tensor");
    TORCH_CHECK(inp.is_cuda()    && inp.scalar_type()    == torch::kBFloat16 && inp.is_contiguous(),
                "inp must be a contiguous bf16 CUDA tensor");
    TORCH_CHECK(grad_w.is_cuda() && grad_w.scalar_type() == torch::kBFloat16 && grad_w.is_contiguous(),
                "grad_w must be a contiguous bf16 CUDA tensor");

    const int BT = (int)go.size(0);
    const int V  = (int)go.size(1);
    const int H  = (int)inp.size(1);
    TORCH_CHECK(inp.size(0) == BT,      "GO and INP must share BT");
    TORCH_CHECK(grad_w.size(0) == V && grad_w.size(1) == H, "grad_w shape mismatch");

    const int M = V, N = H, K = BT;

    // Leading dims (row-stride in elements):
    //   A col-major (V, BT) → ldA = V (stride between consecutive BT rows)
    //   B row-major (BT, H) → ldB = H
    //   C row-major (V,  H) → ldC = H
    const int lda = V;
    const int ldb = H;
    const int ldc = H;

    auto* pA = reinterpret_cast<ElementA const*>(go.data_ptr());
    auto* pB = reinterpret_cast<ElementB const*>(inp.data_ptr());
    auto* pD = reinterpret_cast<ElementC*>(grad_w.data_ptr());

    typename Gemm::Arguments args {
        {M, N, K},                                             // problem size
        { pA,           lda },                                 // A ref
        { pB,           ldb },                                 // B ref
        { pD,           ldc },                                 // C (unused, beta=0 — reuse D)
        { pD,           ldc },                                 // D
        { ElementComputeEpilogue(1.0f),
          ElementComputeEpilogue(0.0f) }                       // alpha, beta
    };

    Gemm gemm;

    auto can = gemm.can_implement(args);
    TORCH_CHECK(can == cutlass::Status::kSuccess,
                "CUTLASS GEMM not implementable (status ",
                cutlassGetStatusString(can), ")");

    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty(
        {(int64_t)workspace_size},
        torch::dtype(torch::kUInt8).device(go.device()));

    auto init = gemm.initialize(args, workspace.data_ptr());
    TORCH_CHECK(init == cutlass::Status::kSuccess,
                "CUTLASS GEMM init failed (status ",
                cutlassGetStatusString(init), ")");

    auto run = gemm.run();
    TORCH_CHECK(run == cutlass::Status::kSuccess,
                "CUTLASS GEMM run failed (status ",
                cutlassGetStatusString(run), ")");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass3x_grad_w_bf16", &cutlass3x_grad_w_bf16,
          "CUTLASS 2.x bf16 GEMM: grad_W = GO^T @ INP (sm_80 tag, runs on sm_120)");
}