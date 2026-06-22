/*
 * Fused grad_W + AdamW via CUTLASS 2.x Epilogue Visitor Tree (EVT).
 *
 * The GEMM mainloop is the same well-tuned bf16 tensor-op GEMM from the
 * unfused CUTLASS path. The difference is the epilogue: instead of
 * `LinearCombination` (which writes alpha*accum to D), we use
 * `EpilogueVisitorAdamW` which:
 *   - reads W from C slot
 *   - reads M, V from global via explicit 128-bit loads at the thread's
 *     global (row, col) position (derived from iterator_D_.thread_start())
 *   - computes AdamW update in fp32
 *   - writes M, V directly to global (128-bit stores)
 *   - writes updated W through the D-slot iterator
 *
 * Three tile-shape variants are exposed:
 *   - 128×128×32  (baseline; good for square layers like q/k/v/o_proj)
 *   - 128×256×32  (wide-N; better for down_proj where H >> V)
 *   - 256×128×32  (wide-M; better for lm_head and gate/up_proj where V >> H)
 *
 * Layouts (all three variants):
 *   A = GO^T (V × BT)  ColumnMajor
 *   B = INP  (BT × H)  RowMajor (CUTLASS 2.x (K, N) view with N contiguous)
 *   C = D = W (V × H)  RowMajor
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "gemm_with_epilogue_visitor.h"
#include "epilogue_visitor_adamw.h"


using ElementA           = cutlass::bfloat16_t;
using LayoutA            = cutlass::layout::ColumnMajor;
using ElementB           = cutlass::bfloat16_t;
using LayoutB            = cutlass::layout::RowMajor;
using ElementC           = cutlass::bfloat16_t;
using LayoutC            = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute     = float;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

using OperatorClass  = cutlass::arch::OpClassTensorOp;
using ArchTag        = cutlass::arch::Sm80;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;


// ── EVT kernel parametric in ThreadblockShape / WarpShape / Stages ───────
template <typename ThreadblockShape_, typename WarpShape_, int Stages_>
struct EVTKernel {
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape        = WarpShape_;

    using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
        ElementC, AlignmentC, ElementAccumulator, ElementCompute>;

    using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementC, LayoutC,
        ElementAccumulator,
        OperatorClass, ArchTag,
        ThreadblockShape, WarpShape, InstructionShape,
        EpilogueFunctorOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Stages_,
        true,
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OperatorClass, ArchTag,
            ElementA, ElementB, ElementC, ElementCompute>::Operator,
        cutlass::gemm::SharedMemoryClearOption::kNone
    >::GemmKernel;

    using EpilogueVisitor = cutlass::epilogue::threadblock::EpilogueVisitorAdamW<
        ThreadblockShape,
        DefaultGemmKernel::kThreadCount,
        typename DefaultGemmKernel::Epilogue::OutputTileIterator,
        ElementAccumulator,
        EpilogueFunctorOp>;

    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        EpilogueVisitor,
        typename DefaultGemmKernel::Epilogue>::Epilogue;

    using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
        typename DefaultGemmKernel::Mma,
        Epilogue,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle>;

    using TensorRefA = cutlass::TensorRef<ElementA, LayoutA>;
    using TensorRefB = cutlass::TensorRef<ElementB, LayoutB>;
    using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

    static void launch(
        torch::Tensor go, torch::Tensor inp,
        torch::Tensor weight, torch::Tensor m, torch::Tensor v,
        float lr, float beta1, float beta2, float eps, float weight_decay,
        float bc1, float bc2)
    {
        TORCH_CHECK(go.is_cuda()     && go.scalar_type()     == torch::kBFloat16 && go.is_contiguous());
        TORCH_CHECK(inp.is_cuda()    && inp.scalar_type()    == torch::kBFloat16 && inp.is_contiguous());
        TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kBFloat16 && weight.is_contiguous());
        TORCH_CHECK(m.is_cuda()      && m.scalar_type()      == torch::kBFloat16 && m.is_contiguous());
        TORCH_CHECK(v.is_cuda()      && v.scalar_type()      == torch::kBFloat16 && v.is_contiguous());

        const int BT = (int)go.size(0);
        const int V  = (int)go.size(1);
        const int H  = (int)inp.size(1);
        const int M  = V, N = H, K = BT;

        auto* pA = reinterpret_cast<ElementA*>(go.data_ptr());
        auto* pB = reinterpret_cast<ElementB*>(inp.data_ptr());
        auto* pW = reinterpret_cast<ElementC*>(weight.data_ptr());
        auto* pM = reinterpret_cast<ElementC*>(m.data_ptr());
        auto* pV = reinterpret_cast<ElementC*>(v.data_ptr());
        const int ldc = H;

        cutlass::gemm::GemmCoord problem({M, N, K});

        typename EpilogueFunctorOp::Params alpha_beta{
            ElementCompute(1.0f), ElementCompute(0.0f)};
        typename EpilogueVisitor::Arguments visitor_args{
            alpha_beta,
            pM, pV, (int64_t)ldc,
            lr, beta1, beta2, eps, weight_decay, bc1, bc2};

        typename GemmKernel::Arguments gemm_args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem, /*batch_count=*/1,
            TensorRefA(pA, LayoutA::packed({M, K})),
            TensorRefB(pB, LayoutB::packed({K, N})),
            TensorRefC(pW, LayoutC::packed({M, N})),
            TensorRefC(pW, LayoutC::packed({M, N})),
            nullptr, nullptr,
            0, 0,
            visitor_args);

        typename GemmKernel::Params params(gemm_args);

        dim3 grid = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle()
                        .get_grid_shape(params.grid_tiled_shape);
        dim3 block(GemmKernel::kThreadCount, 1, 1);
        int smem = (int)sizeof(typename GemmKernel::SharedStorage);

        if (smem >= (48 << 10)) {
            cudaFuncSetAttribute(
                cutlass::Kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }

        cutlass::Kernel<GemmKernel><<<grid, block, smem>>>(params);

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
                    "EVT fused kernel launch failed: ", cudaGetErrorString(err));
    }
};


// ── Tile-shape variants ─────────────────────────────────────────────────
// ThreadblockShape, WarpShape, Stages
// Note: WarpShape is chosen so ThreadblockShape is evenly covered by a 2×N
// grid of warps (each 64×64×32). 8-warp CTAs for the larger shapes.

using EVT_128x128 = EVTKernel<
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape< 64,  64, 32>,
    /*Stages=*/3>;

using EVT_128x256 = EVTKernel<
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape< 64,  64, 32>,
    /*Stages=*/3>;

using EVT_256x128 = EVTKernel<
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape< 64,  64, 32>,
    /*Stages=*/3>;


// ── Python-facing entry points ──────────────────────────────────────────
void cutlass_fused_adamw_evt(
    torch::Tensor go, torch::Tensor inp,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2) {
    EVT_128x128::launch(go, inp, weight, m, v,
                        lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}

void cutlass_fused_adamw_evt_128x256(
    torch::Tensor go, torch::Tensor inp,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2) {
    EVT_128x256::launch(go, inp, weight, m, v,
                        lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}

void cutlass_fused_adamw_evt_256x128(
    torch::Tensor go, torch::Tensor inp,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2) {
    EVT_256x128::launch(go, inp, weight, m, v,
                        lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_fused_adamw_evt",         &cutlass_fused_adamw_evt,
          "EVT fused grad_W + AdamW (128×128×32, Stages=3)");
    m.def("cutlass_fused_adamw_evt_128x256", &cutlass_fused_adamw_evt_128x256,
          "EVT fused grad_W + AdamW (128×256×32, Stages=3 — wide-N)");
    m.def("cutlass_fused_adamw_evt_256x128", &cutlass_fused_adamw_evt_256x128,
          "EVT fused grad_W + AdamW (256×128×32, Stages=3 — wide-M)");
}
