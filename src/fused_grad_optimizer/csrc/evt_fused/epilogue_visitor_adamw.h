/*
 * Custom EpilogueVisitor that fuses the AdamW update into the GEMM.
 *
 * Semantics per visit():
 *   grad = accumulator_fragment                (fp32, from GEMM mainloop)
 *   W    = fragment_C_[frag_idx]               (bf16, loaded in begin_step)
 *   M, V = loaded from global at thread_offset_ (bf16, via uint4)
 *
 *   M_new = beta1 * M + (1-beta1) * grad
 *   V_new = beta2 * V + (1-beta2) * grad^2
 *   W_new = W*(1 - lr*wd) - lr * (M_new/bc1) / (sqrt(V_new/bc2) + eps)
 *
 *   fragment_D_[frag_idx] = W_new             (stored via iterator_D_)
 *   M[offset], V[offset]  = M_new, V_new      (stored via uint4 direct write)
 *
 * Constraints:
 *   - kElementsPerAccess must equal 8 (we access M/V as uint4 = 16 bytes = 8 bf16).
 *   - M and V strides must match the weight stride (same (V, H) row-major layout).
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/memory.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/fast_math.h"


namespace cutlass {
namespace epilogue {
namespace threadblock {

template <
    typename ThreadblockShape_,
    int ThreadCount,
    typename OutputTileIterator_,
    typename ElementAccumulator_,
    typename ElementwiseFunctor_      // cutlass::epilogue::thread::LinearCombination (alpha/beta)
>
class EpilogueVisitorAdamW {
public:

    using ThreadblockShape   = ThreadblockShape_;
    static int const kThreadCount = ThreadCount;

    using OutputTileIterator = OutputTileIterator_;
    using ElementwiseFunctor = ElementwiseFunctor_;

    static int const kIterations         = OutputTileIterator::kIterations;
    static int const kElementsPerAccess  = OutputTileIterator::kElementsPerAccess;

    using ElementOutput      = typename OutputTileIterator::Element;   // bf16
    using LayoutOutput       = cutlass::layout::RowMajor;
    using ElementAccumulator = ElementAccumulator_;                    // float

    // Dummy typedefs — GemmWithEpilogueVisitor template pulls these from the
    // visitor; they're not actually used in the AdamW path.
    using ElementNorm = float;
    using ElementSum  = float;

    using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
    using OutputVector        = Array<ElementOutput,      kElementsPerAccess>;

    // 128-bit access type (must match kElementsPerAccess == 8 bf16).
    static_assert(kElementsPerAccess == 8,
                  "EpilogueVisitorAdamW requires kElementsPerAccess == 8 (128-bit).");

    /// Argument structure (host-side)
    struct Arguments {
        typename ElementwiseFunctor::Params elementwise;  // alpha=1, beta=0 always here

        ElementOutput* ptr_M;
        ElementOutput* ptr_V;
        int64_t        ldm;          // row stride of W/M/V (elements) — same for all three

        float lr;
        float beta1;
        float beta2;
        float eps;
        float weight_decay;
        float bc1;                   // 1 - beta1^step
        float bc2;                   // 1 - beta2^step

        Arguments()
        : ptr_M(nullptr), ptr_V(nullptr), ldm(0),
          lr(0), beta1(0), beta2(0), eps(0), weight_decay(0), bc1(1), bc2(1) {}

        Arguments(typename ElementwiseFunctor::Params elementwise_,
                  ElementOutput* M_, ElementOutput* V_, int64_t ldm_,
                  float lr_, float b1_, float b2_, float eps_, float wd_,
                  float bc1_, float bc2_)
        : elementwise(elementwise_),
          ptr_M(M_), ptr_V(V_), ldm(ldm_),
          lr(lr_), beta1(b1_), beta2(b2_), eps(eps_), weight_decay(wd_),
          bc1(bc1_), bc2(bc2_) {}
    };

    /// Params (device-side, trivially constructible from Arguments)
    struct Params {
        typename ElementwiseFunctor::Params elementwise;
        ElementOutput* ptr_M;
        ElementOutput* ptr_V;
        int64_t        ldm;
        float lr, beta1, beta2, eps, wd;
        float inv_bc1, inv_bc2;
        float one_minus_b1, one_minus_b2, one_minus_lr_wd;

        CUTLASS_HOST_DEVICE
        Params() {}

        CUTLASS_HOST_DEVICE
        Params(Arguments const& a)
        : elementwise(a.elementwise),
          ptr_M(a.ptr_M), ptr_V(a.ptr_V), ldm(a.ldm),
          lr(a.lr), beta1(a.beta1), beta2(a.beta2),
          eps(a.eps), wd(a.weight_decay),
          inv_bc1(1.0f / a.bc1), inv_bc2(1.0f / a.bc2),
          one_minus_b1(1.0f - a.beta1), one_minus_b2(1.0f - a.beta2),
          one_minus_lr_wd(1.0f - a.lr * a.weight_decay) {}
    };

    /// Shared storage (none required)
    struct SharedStorage {};

private:

    Params const&                           params_;
    SharedStorage&                          shared_storage_;
    MatrixCoord                             extent_;
    ElementwiseFunctor                      elementwise_;

    OutputTileIterator                      iterator_W_;   // C-slot: reads W
    OutputTileIterator                      iterator_Wo_;  // D-slot: writes W
    typename OutputTileIterator::Fragment   fragment_W_;
    typename OutputTileIterator::Fragment   fragment_Wo_;

    MatrixCoord                             thread_offset_;

public:

    // Constructor signature matches what GemmWithEpilogueVisitor expects
    // (it passes ptr_Max / ptr_Sum / column_offset we just ignore).
    CUTLASS_DEVICE
    EpilogueVisitorAdamW(
        Params const& params,
        SharedStorage& shared_storage,
        cutlass::MatrixCoord const& problem_size,
        int thread_idx, int warp_idx, int lane_idx,
        typename OutputTileIterator::Params params_C,
        typename OutputTileIterator::Params params_D,
        typename OutputTileIterator::Element* ptr_C,
        typename OutputTileIterator::Element* ptr_D,
        ElementNorm* /*ptr_Max*/ = nullptr,
        ElementSum*  /*ptr_Sum*/ = nullptr,
        cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0, 0),
        int /*column_offset*/ = 0,
        cutlass::MatrixCoord const& /*problem_size_real*/ = cutlass::MatrixCoord(0, 0)
    )
    : params_(params),
      shared_storage_(shared_storage),
      extent_(problem_size),
      elementwise_(params.elementwise),
      iterator_W_ (params_C, ptr_C, problem_size, thread_idx, threadblock_offset),
      iterator_Wo_(params_D, ptr_D, problem_size, thread_idx, threadblock_offset)
    {}

    CUTLASS_DEVICE
    void set_k_partition(int /*split_k_index*/, int /*split_k_slices*/) {}

    CUTLASS_DEVICE
    void set_batch_index(int /*batch_idx*/) {}

    CUTLASS_DEVICE
    void begin_epilogue() {}

    /// Called at the start of one step: load W fragment.
    CUTLASS_DEVICE
    void begin_step(int /*step_idx*/) {
        fragment_W_.clear();
        fragment_Wo_.clear();
        iterator_W_.load(fragment_W_);
        ++iterator_W_;
    }

    CUTLASS_DEVICE
    void begin_row(int /*row_idx*/) {}

    /// Per-fragment: apply AdamW, R/W M and V directly from/to global.
    CUTLASS_DEVICE
    void visit(int iter_idx, int row_idx, int column_idx, int frag_idx,
               AccumulatorFragment const& grad)
    {
        // Current thread's global coordinates for this fragment.
        thread_offset_ =
            iterator_Wo_.thread_start() +
            OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

        bool row_guard = (thread_offset_.row()    < extent_.row());
        bool col_guard = (thread_offset_.column() + kElementsPerAccess <= extent_.column());

        // W already in registers from iterator_W_.
        OutputVector& src_W = reinterpret_cast<OutputVector*>(&fragment_W_)[frag_idx];
        OutputVector& dst_W = reinterpret_cast<OutputVector*>(&fragment_Wo_)[frag_idx];

        // Compute M/V global element offset (same layout as W).
        int64_t offset = (int64_t)thread_offset_.row() * params_.ldm
                       + thread_offset_.column();

        // 128-bit loads for M and V (bf16 × 8 = uint4). Out-of-bounds: skip.
        uint4 m_raw = {0, 0, 0, 0};
        uint4 v_raw = {0, 0, 0, 0};
        if (row_guard && col_guard) {
            asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(m_raw.x), "=r"(m_raw.y), "=r"(m_raw.z), "=r"(m_raw.w)
                : "l"(params_.ptr_M + offset));
            asm volatile("ld.global.cs.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                : "=r"(v_raw.x), "=r"(v_raw.y), "=r"(v_raw.z), "=r"(v_raw.w)
                : "l"(params_.ptr_V + offset));
        }

        auto* mb = reinterpret_cast<ElementOutput*>(&m_raw);
        auto* vb = reinterpret_cast<ElementOutput*>(&v_raw);

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kElementsPerAccess; e++) {
            float g  = static_cast<float>(grad[e]);
            float wf = static_cast<float>(src_W[e]);
            float mf = static_cast<float>(mb[e]);
            float vf = static_cast<float>(vb[e]);

            mf = params_.beta1 * mf + params_.one_minus_b1 * g;
            vf = params_.beta2 * vf + params_.one_minus_b2 * g * g;

            float mhat = mf * params_.inv_bc1;
            float vhat = vf * params_.inv_bc2;
            wf = wf * params_.one_minus_lr_wd
               - params_.lr * mhat / (sqrtf(vhat) + params_.eps);

            mb[e] = static_cast<ElementOutput>(mf);
            vb[e] = static_cast<ElementOutput>(vf);
            dst_W[e] = static_cast<ElementOutput>(wf);
        }

        if (row_guard && col_guard) {
            asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n"
                :: "l"(params_.ptr_M + offset),
                   "r"(m_raw.x), "r"(m_raw.y), "r"(m_raw.z), "r"(m_raw.w));
            asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};\n"
                :: "l"(params_.ptr_V + offset),
                   "r"(v_raw.x), "r"(v_raw.y), "r"(v_raw.z), "r"(v_raw.w));
        }
    }

    CUTLASS_DEVICE
    void end_row(int /*row_idx*/) {}

    /// End of step: flush W output via iterator_D_.
    CUTLASS_DEVICE
    void end_step(int /*step_idx*/) {
        iterator_Wo_.store(fragment_Wo_);
        ++iterator_Wo_;
    }

    CUTLASS_DEVICE
    void end_epilogue() {}
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
