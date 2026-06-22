/*
 * Fused grad_W + AdamW — CUTLASS 3.x kernel for Hopper/Blackwell.
 *
 * Replaces the WMMA-based kernel with CUTLASS 3.x collective builders:
 *   - TMA (Tensor Memory Accelerator) for all global memory transfers
 *   - WGMMA (Warp Group Matrix Multiply-Accumulate) for native tensor cores
 *   - Warp specialization: producer warps load data, consumer warps compute
 *   - Persistent scheduling: one CTA per SM, looping over tiles
 *   - Software pipelining: overlaps next tile's loads with current compute
 *
 * The AdamW optimizer update is fused into the GEMM epilogue via CUTLASS's
 * Epilogue Visitor Tree (EVT):
 *   - AuxLoad(W, m, v): load optimizer state from global memory
 *   - Compute: m_new = β1·m + (1-β1)·grad, v_new = β2·v + (1-β2)·grad²
 *   - Compute: w_new = w·(1-lr·wd) - lr·m̂/(√v̂ + ε)
 *   - AuxStore(m_new, v_new): write updated state, D output = w_new
 *
 * GEMM: grad(V,H) = GO^T(V,BT) @ INP(BT,H)  →  M=V, K=BT, N=H
 *
 * Requires: CUDA 12.0+, CUTLASS 4.x, sm_90a+ (Hopper/Blackwell)
 */

// For sm_120 (Blackwell workstation): enable SM100 features (tcgen05, TMA, TMEM)
// sm_120 has the same tcgen05 tensor cores as sm_100, but compute_120a
// doesn't define __CUDA_ARCH_FEAT_SM100_ALL. Force-enable the needed macros.
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 1000
// Enable SM100A features: tcgen05 MMA + TMEM + TMA
#ifndef CUTLASS_ARCH_MMA_SM100A_ENABLED
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif
#ifndef CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED
#define CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_ENABLED 1
#endif
#endif
#endif

#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"

using namespace cute;

// ============================================================================
// Custom AdamW Functors for EVT
// ============================================================================
// These operate on Array<float, N> fragments inside the epilogue visitor.
// Sm90Compute converts inputs to ElementCompute (float), calls the functor,
// and converts the output to ElementOutput.

// Arguments structs defined OUTSIDE templates so that
// ComputeFn<float>::Arguments == ComputeFn<Array<float,N>>::Arguments
// (CUTLASS Sm90Compute deduces params type from ComputeFn<ElementCompute>::Arguments
//  but calls operator() on ComputeFn<Array<ElementCompute,N>>)

struct MomentUpdate1Args {
    float beta1;
};

struct MomentUpdate2Args {
    float beta2;
};

struct AdamWWeightUpdateArgs {
    float lr;
    float wd;
    float bc1;
    float bc2;
    float eps;
};

// First moment update: m_new = beta1 * m + (1 - beta1) * grad
template <class T>
struct MomentUpdate1 {
    using Arguments = MomentUpdate1Args;

    CUTLASS_HOST_DEVICE
    T operator()(T const& m, T const& grad, Arguments const& args) const {
        T result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < T::kElements; ++i) {
            result[i] = args.beta1 * float(m[i]) + (1.0f - args.beta1) * float(grad[i]);
        }
        return result;
    }
};

// Second moment update: v_new = beta2 * v + (1 - beta2) * grad^2
template <class T>
struct MomentUpdate2 {
    using Arguments = MomentUpdate2Args;

    CUTLASS_HOST_DEVICE
    T operator()(T const& v, T const& grad, Arguments const& args) const {
        T result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < T::kElements; ++i) {
            float g = float(grad[i]);
            result[i] = args.beta2 * float(v[i]) + (1.0f - args.beta2) * g * g;
        }
        return result;
    }
};

// AdamW weight update: w_new = w*(1 - lr*wd) - lr * (m_new/bc1) / (sqrt(v_new/bc2) + eps)
template <class T>
struct AdamWWeightUpdate {
    using Arguments = AdamWWeightUpdateArgs;

    CUTLASS_HOST_DEVICE
    T operator()(T const& w, T const& m_new, T const& v_new, Arguments const& args) const {
        T result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < T::kElements; ++i) {
            float mh = float(m_new[i]) / args.bc1;
            float vh = float(v_new[i]) / args.bc2;
            result[i] = float(w[i]) * (1.0f - args.lr * args.wd)
                       - args.lr * mh / (sqrtf(vh) + args.eps);
        }
        return result;
    }
};

// ============================================================================
// CUTLASS 3.x GEMM + AdamW Epilogue Kernel Type
// ============================================================================

using ElementBF16 = cutlass::bfloat16_t;
using ElementFP32 = float;

constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
constexpr int AlignBF16 = 128 / cutlass::sizeof_bits_v<ElementBF16>;  // = 8 elements

// --- Auxiliary Load (Stages=0: direct global→register, no shared memory) ---
// Used to load W, m, v during the epilogue. Stages=0 avoids SMEM pressure
// since these are single-use loads (not pipelined across the K loop).
using AuxLoad = cutlass::epilogue::fusion::Sm90AuxLoad<
    0,                              // Stages (0 = direct G2R)
    void,                           // EpilogueTile (unused for Stages=0)
    ElementBF16,                    // Element type
    cutlass::layout::RowMajor,      // Memory layout (stride type derived from this)
    void,                           // SmemLayoutAtom (unused)
    void,                           // CopyOpS2R (unused)
    AlignBF16,                      // Alignment (8 bf16 elements = 16 bytes)
    true                            // EnableNullptr
>;

// --- Auxiliary Store (Stages=0: direct register→global, no shared memory) ---
// Used to store updated m, v during the epilogue.
using AuxStore = cutlass::epilogue::fusion::Sm90AuxStore<
    0,                              // Stages (0 = direct R2G)
    void,                           // EpilogueTile (unused)
    ElementBF16,                    // Element type
    RoundStyle,                     // Float rounding for fp32→bf16 conversion
    cutlass::layout::RowMajor,      // Memory layout
    void,                           // SmemLayoutAtom (unused)
    void,                           // CopyOpR2S (unused)
    AlignBF16,                      // Alignment
    true                            // EnableNullptr
>;

// --- Epilogue Visitor Tree (EVT) for fused AdamW ---
//
// Tree structure:
//   D(w_new) = AdamWWeightUpdate(
//       AuxLoad(W),
//       AuxStore(m_new) ← MomentUpdate1(AuxLoad(m), AccFetch(grad)),
//       AuxStore(v_new) ← MomentUpdate2(AuxLoad(v), AccFetch(grad))
//   )
//
// Data flow:
//   1. AccFetch returns the GEMM accumulator (gradient tile, fp32)
//   2. AuxLoad reads current W, m, v from global memory (bf16→fp32)
//   3. MomentUpdate1/2 compute new m, v in fp32
//   4. AuxStore writes m_new, v_new to global memory (fp32→bf16) and passes through
//   5. AdamWWeightUpdate computes w_new from (w, m_new, v_new)
//   6. w_new is stored as D output via the epilogue's TMA store pipeline
//
using AdamWEVT = cutlass::epilogue::fusion::Sm90EVT<
    // Root node: ternary compute → w_new (stored as D output)
    cutlass::epilogue::fusion::Sm90Compute<AdamWWeightUpdate, ElementBF16, ElementFP32, RoundStyle>,
    // Child 0: Load current weight W
    AuxLoad,
    // Child 1: m branch — compute m_new, store as side-effect, pass through
    cutlass::epilogue::fusion::Sm90EVT<
        AuxStore,   // Stores m_new to global memory, returns m_new (passthrough)
        cutlass::epilogue::fusion::Sm90EVT<
            // Binary compute: m_new = beta1*m + (1-beta1)*grad
            cutlass::epilogue::fusion::Sm90Compute<MomentUpdate1, ElementFP32, ElementFP32, RoundStyle>,
            AuxLoad,                                        // Load current m
            cutlass::epilogue::fusion::Sm90AccFetch         // grad (accumulator)
        >
    >,
    // Child 2: v branch — compute v_new, store as side-effect, pass through
    cutlass::epilogue::fusion::Sm90EVT<
        AuxStore,   // Stores v_new to global memory, returns v_new (passthrough)
        cutlass::epilogue::fusion::Sm90EVT<
            // Binary compute: v_new = beta2*v + (1-beta2)*grad^2
            cutlass::epilogue::fusion::Sm90Compute<MomentUpdate2, ElementFP32, ElementFP32, RoundStyle>,
            AuxLoad,                                        // Load current v
            cutlass::epilogue::fusion::Sm90AccFetch         // grad (accumulator)
        >
    >
>;

// --- GEMM configuration ---
// Problem: grad(V,H) = GO^T(V,BT) @ INP(BT,H)
//   A = GO viewed as (V, BT) column-major  (GO is (BT,V) row-major in memory)
//   B = INP as (BT, H) row-major
//   D = W as (V, H) row-major (output: updated weight)
using LayoutA = cutlass::layout::ColumnMajor;   // A(M=V, K=BT): GO is (BT,V) row-major → A(V,BT) col-major
using LayoutB = cutlass::layout::ColumnMajor;   // B(N=H, K=BT): INP is (BT,H) row-major → B^T(H,BT) col-major
using LayoutC = cutlass::layout::RowMajor;      // C: placeholder (not used)
using LayoutD = cutlass::layout::RowMajor;      // D(V, H): weight output

// Use SM100 (Blackwell) collective builders with tcgen05 tensor cores.
// B200 workstation (sm_120) has tcgen05 + 99 KB SMEM.
// 64x64x64 tiles keep SMEM under the limit.
using TileShape = Shape<_128, _128, _64>;       // CTA tile (tcgen05 supports 128x128)
using ClusterShape = Shape<_1, _1, _1>;         // 1x1x1 cluster

using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementFP32, ElementFP32,                     // Accumulator, Compute element
    ElementBF16, LayoutC, AlignBF16,              // C (unused placeholder)
    ElementBF16, LayoutD, AlignBF16,              // D (weight output)
    EpilogueSchedule,
    AdamWEVT                                       // Custom epilogue fusion
>::CollectiveOp;

// B200 workstation (sm_120) has only 99 KB max SMEM per block.
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementBF16, LayoutA, AlignBF16,              // A: grad_output (column-major view)
    ElementBF16, LayoutB, AlignBF16,              // B: input (column-major view)
    ElementFP32,                                   // Accumulator (fp32)
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCount<2>,      // Explicit 2 stages (fits 99 KB SMEM)
    MainloopSchedule
>::CollectiveOp;

// Full kernel: GemmUniversal with default tile scheduler
// (PersistentScheduler requires special grid launch; try void for basic launch first)
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,          // ProblemShape: (M, N, K, L)
    CollectiveMainloop,
    CollectiveEpilogue
    // void = default tile scheduler (1 CTA per output tile)
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ============================================================================
// Stride types
// ============================================================================

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using StrideAux = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;

// ============================================================================
// Python wrapper
// ============================================================================

void fused_grad_adamw_cutlass_v2(
    torch::Tensor grad_output,    // (BT, V) bf16
    torch::Tensor input,          // (BT, H) bf16
    torch::Tensor weight,         // (V, H) bf16 — updated in-place
    torch::Tensor m,              // (V, H) bf16 — first moment, updated in-place
    torch::Tensor v,              // (V, H) bf16 — second moment, updated in-place
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2
) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.scalar_type() == torch::kBFloat16,
                "grad_output must be CUDA bf16");
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16,
                "input must be CUDA bf16");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kBFloat16,
                "weight must be CUDA bf16");
    TORCH_CHECK(grad_output.is_contiguous() && input.is_contiguous(),
                "grad_output and input must be contiguous");
    TORCH_CHECK(weight.is_contiguous() && m.is_contiguous() && v.is_contiguous(),
                "weight, m, v must be contiguous");

    int BT_dim = grad_output.size(0);
    int V_dim  = grad_output.size(1);
    int H_dim  = input.size(1);

    // GEMM problem: M=V, N=H, K=BT, L=1 (no batching)
    auto problem_size = cute::make_shape(V_dim, H_dim, BT_dim, 1);

    // Compute CuTe strides from shapes
    auto stride_A   = cutlass::make_cute_packed_stride(StrideA{},   cute::make_shape(V_dim, BT_dim, 1));
    auto stride_B   = cutlass::make_cute_packed_stride(StrideB{},   cute::make_shape(H_dim, BT_dim, 1));
    auto stride_D   = cutlass::make_cute_packed_stride(StrideD{},   cute::make_shape(V_dim, H_dim, 1));
    auto stride_aux = cutlass::make_cute_packed_stride(StrideAux{}, cute::make_shape(V_dim, H_dim, 1));

    // Hardware info for persistent scheduling
    cutlass::KernelHardwareInfo hw_info;
    int device_id = grad_output.device().index();
    if (device_id < 0) device_id = 0;
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    // Raw pointers
    auto* go_ptr = reinterpret_cast<ElementBF16 const*>(grad_output.data_ptr());
    auto* inp_ptr = reinterpret_cast<ElementBF16 const*>(input.data_ptr());
    auto* w_ptr = reinterpret_cast<ElementBF16*>(weight.data_ptr());
    auto* m_ptr = reinterpret_cast<ElementBF16*>(m.data_ptr());
    auto* v_ptr = reinterpret_cast<ElementBF16*>(v.data_ptr());

    // Build CUTLASS arguments
    // The EVT arguments follow the tree structure:
    //   {child0_args, child1_args, ..., node_args}
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        // ── Mainloop arguments ──
        {go_ptr, stride_A, inp_ptr, stride_B},
        // ── Epilogue arguments ──
        {
            // Thread args (EVT tree, recursive structure):
            {
                // Child 0: AuxLoad(W) args
                {w_ptr, ElementBF16(0), stride_aux},

                // Child 1: m branch = EVT<AuxStore, EVT<Compute, AuxLoad, AccFetch>>
                {
                    // Child 0 of m branch: EVT<Compute(m_update), AuxLoad(m), AccFetch>
                    {
                        {m_ptr, ElementBF16(0), stride_aux},   // AuxLoad(m) args
                        {},                                     // AccFetch args (empty)
                        {beta1}                                 // MomentUpdate1::Arguments
                    },
                    // Node of m branch: AuxStore(m) args
                    {m_ptr, stride_aux}
                },

                // Child 2: v branch = EVT<AuxStore, EVT<Compute, AuxLoad, AccFetch>>
                {
                    // Child 0 of v branch: EVT<Compute(v_update), AuxLoad(v), AccFetch>
                    {
                        {v_ptr, ElementBF16(0), stride_aux},   // AuxLoad(v) args
                        {},                                     // AccFetch args (empty)
                        {beta2}                                 // MomentUpdate2::Arguments
                    },
                    // Node of v branch: AuxStore(v) args
                    {v_ptr, stride_aux}
                },

                // Root node: AdamWWeightUpdate::Arguments
                {lr, weight_decay, bias_correction1, bias_correction2, eps}
            },
            // C source pointer (unused — no Sm90SrcFetch in our EVT)
            // Pass a valid pointer anyway: CUTLASS may create TMA descriptors for C
            // even when the EVT doesn't fetch from it.
            w_ptr,
            stride_D,   // C stride (same layout as D)
            // D output pointer (updated weight)
            w_ptr,
            stride_D
        },
        hw_info
    };

    // Instantiate and run
    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    auto workspace_tensor = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(weight.device()));

    cutlass::Status status;

    status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS kernel cannot implement this problem: ",
                cutlass::cutlassGetStatusString(status));

    // Clear any prior CUDA errors
    cudaGetLastError();

    status = gemm_op.initialize(arguments, workspace_tensor.data_ptr());
    if (status != cutlass::Status::kSuccess) {
        cudaError_t cuda_err = cudaGetLastError();
        TORCH_CHECK(false,
                    "CUTLASS kernel initialization failed: ",
                    cutlass::cutlassGetStatusString(status),
                    " | CUDA error: ", cudaGetErrorString(cuda_err));
    }

    // Debug: print SMEM requirement
    int smem_size = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
    printf("CUTLASS kernel SMEM size: %d bytes (%d KB)\n", smem_size, smem_size / 1024);
    printf("Max configurable SMEM: 101376 bytes (99 KB)\n");
    printf("SMEM fits: %s\n", smem_size <= 101376 ? "YES" : "NO - TOO LARGE");
    fflush(stdout);

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        cudaError_t cuda_err = cudaGetLastError();
        TORCH_CHECK(false,
                    "CUTLASS kernel launch failed: ",
                    cutlass::cutlassGetStatusString(status),
                    " | CUDA: ", cudaGetErrorString(cuda_err),
                    " | SMEM: ", smem_size, " bytes");
    }
}

// ============================================================================
// Simple GEMM test (no EVT, standard LinearCombination epilogue)
// Used to verify basic CUTLASS SM90 GEMM works on this hardware
// ============================================================================
using SimpleEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementFP32, ElementFP32,
    ElementBF16, LayoutD, AlignBF16,
    ElementBF16, LayoutD, AlignBF16,
    EpilogueSchedule
>::CollectiveOp;

using SimpleMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementBF16, LayoutA, AlignBF16,
    ElementBF16, LayoutB, AlignBF16,
    ElementFP32,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCount<2>,
    MainloopSchedule
>::CollectiveOp;

using SimpleGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    SimpleMainloop,
    SimpleEpilogue,
    cutlass::gemm::PersistentScheduler
>;

using SimpleGemm = cutlass::gemm::device::GemmUniversalAdapter<SimpleGemmKernel>;

void test_simple_gemm(
    torch::Tensor grad_output,    // (BT, V) bf16
    torch::Tensor input,          // (BT, H) bf16
    torch::Tensor output          // (V, H) bf16 — output
) {
    int BT_dim = grad_output.size(0);
    int V_dim  = grad_output.size(1);
    int H_dim  = input.size(1);

    auto problem_size = cute::make_shape(V_dim, H_dim, BT_dim, 1);

    using SStrideA = typename SimpleGemm::GemmKernel::StrideA;
    using SStrideB = typename SimpleGemm::GemmKernel::StrideB;
    using SStrideD = typename SimpleGemm::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(SStrideA{}, cute::make_shape(V_dim, BT_dim, 1));
    auto stride_B = cutlass::make_cute_packed_stride(SStrideB{}, cute::make_shape(H_dim, BT_dim, 1));
    auto stride_D = cutlass::make_cute_packed_stride(SStrideD{}, cute::make_shape(V_dim, H_dim, 1));

    cutlass::KernelHardwareInfo hw_info;
    int device_id = grad_output.device().index();
    if (device_id < 0) device_id = 0;
    hw_info.device_id = device_id;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);

    auto* go_ptr = reinterpret_cast<ElementBF16 const*>(grad_output.data_ptr());
    auto* inp_ptr = reinterpret_cast<ElementBF16 const*>(input.data_ptr());
    auto* out_ptr = reinterpret_cast<ElementBF16*>(output.data_ptr());

    typename SimpleGemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {go_ptr, stride_A, inp_ptr, stride_B},
        {{1.0f, 0.0f}, out_ptr, stride_D, out_ptr, stride_D},
        hw_info
    };

    SimpleGemm gemm_op;
    size_t workspace_size = SimpleGemm::get_workspace_size(arguments);
    auto workspace_tensor = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(output.device()));

    cudaGetLastError();

    auto status = gemm_op.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "Simple GEMM can_implement failed: ", cutlass::cutlassGetStatusString(status));

    status = gemm_op.initialize(arguments, workspace_tensor.data_ptr());
    if (status != cutlass::Status::kSuccess) {
        cudaError_t cuda_err = cudaGetLastError();
        TORCH_CHECK(false, "Simple GEMM initialize failed: ",
                    cutlass::cutlassGetStatusString(status),
                    " | CUDA: ", cudaGetErrorString(cuda_err));
    }

    status = gemm_op.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "Simple GEMM run failed: ", cutlass::cutlassGetStatusString(status));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m_mod) {
    m_mod.def("fused_grad_adamw_cutlass", &fused_grad_adamw_cutlass_v2,
              "Fused grad_W + AdamW (CUTLASS 3.x: TMA + WGMMA + warp-specialized + persistent)");
    m_mod.def("test_simple_gemm", &test_simple_gemm,
              "Simple GEMM test (CUTLASS 3.x SM90, no EVT)");
}
