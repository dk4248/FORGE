/*
 * Sm90 EVT-fused grad_W + AdamW (H200).
 *
 * True fused kernel: grad_W is never written to HBM. The CUTLASS 3.x Hopper
 * collective mainloop (WGMMA + TMA) computes the fp32 accumulator tile-by-
 * tile; the epilogue then applies the entire AdamW update and writes the new
 * W back out. m and v are streamed through via Sm90AuxLoad / Sm90AuxStore.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * Design note — why we do NOT define a custom FusionOperation tag here.
 * ─────────────────────────────────────────────────────────────────────────
 * CUTLASS 3.5.1's `CallbacksBuilder` only ships specializations for the XOR
 * case (exactly one of aux-in OR aux-out). Supplying a custom specialization
 * for the both-case is error-prone: the builder composes the aux SmemLayout
 * + CopyOps and forwards them into a user-supplied FusionCallbacks; if the
 * packing of TMA descriptors in the resulting Params struct doesn't exactly
 * match what CUTLASS expects, launches fault with "misaligned address"
 * because CUtensorMap is alignas(64).
 *
 * CUTLASS exposes a clean escape hatch for this: if the type passed to
 * CollectiveBuilder as `FusionOpOrCallbacks` is NOT a FusionOperation
 * subclass, `CallbacksBuilder` selects a "passthrough" specialization
 * (collective_builder.hpp:97) that forwards the type verbatim. The
 * CollectiveEpilogue then treats it as a ready-to-use visitor tree, and
 * CUTLASS's own Sm90VisitorImplBase<...> (which has correct Params layout
 * for 1, 2, 3, 4 ops + a tuple fallback for 5+) handles descriptor packing.
 *
 * That's what this file does. We build the tree with Sm90EVT<...> directly,
 * using `EpilogueDescriptor` / `AuxLoadDescriptor` / `AuxStoreDescriptor`
 * helpers to resolve SmemLayoutAtom / CopyOpR2S / CopyOpS2R the same way
 * the CUTLASS unit tests do (see sm90_evt_operations.hpp and
 * sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_cooperative_aux_store.cu).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * Arithmetic (standard AdamW with eps outside the sqrt):
 *     g     = acc                              (fp32, GEMM accumulator)
 *     m'    = β1·m + (1-β1)·g
 *     v'    = β2·v + (1-β2)·g²
 *     m_hat = m' / (1 - β1^t)
 *     v_hat = v' / (1 - β2^t)
 *     denom = sqrt(v_hat) + eps
 *     w'    = (1 - lr·wd) · w + (-lr) · m_hat / denom
 *
 * Writes:
 *     D     = new W        (main output)
 *     aux_m = m'           (Sm90AuxStore, writes TMA)
 *     aux_v = v'           (Sm90AuxStore, writes TMA)
 *
 * Layouts:
 *   A = GO^T   (M=V, K=BT)  ColumnMajor   (V contiguous, BT strided)
 *   B = INP    (K=BT, N=H)  RowMajor      (H contiguous, BT strided)
 *   C = D = W  (V × H)      RowMajor      (H contiguous)
 *   Aux m, v   (V × H)      RowMajor      (same as W)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/packed_stride.hpp"

using namespace cute;


// ══════════════════════════════════════════════════════════════════════════
//                   Fat elementwise functors for AdamW
// ══════════════════════════════════════════════════════════════════════════
// Three fat functors replace the previous 11-node tree of small ops. Each
// functor's operator() does one chunk of AdamW inline with no nested EVT
// calls, and carries its scalar hyperparameters through an `Arguments`
// member (Sm90Compute will SFINAE-detect it and forward `Arguments` as the
// last positional arg in operator()).
//
// Why three instead of one: the EVT visit() contract returns a single
// fragment per node, so we need one node per output channel —
//    m_new (stored via AuxStore_M, returned as fp32 for chaining)
//    v_new (stored via AuxStore_V, returned as fp32 for chaining)
//    w_new (returned as bf16, becomes the main D output)
// But each of these three is a flat, side-effect-free inline function body,
// which gives ptxas a solid shot at inlining end-to-end and leaving the
// WGMMA mainloop pipeline intact (no function-boundary `wgmma.wait_group`).

namespace adamw_fn {

// ── m_new = one_m_b1 * g + beta1 * m_load ──────────────────────────────
template <class T>
struct m_update {
  struct Arguments { float one_m_b1; float beta1; };

  CUTLASS_HOST_DEVICE T
  operator()(T const& g, T const& m_load, Arguments const& args) const {
    return T(float(g) * args.one_m_b1 + float(m_load) * args.beta1);
  }
};

template <class T, int N, bool R>
struct m_update<cutlass::Array<T, N, R>> {
  using Arguments = typename m_update<T>::Arguments;

  CUTLASS_HOST_DEVICE cutlass::Array<T, N, R>
  operator()(cutlass::Array<T, N, R> const& g,
             cutlass::Array<T, N, R> const& m_load,
             Arguments const& args) const {
    cutlass::Array<T, N, R> r;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      r[i] = T(float(g[i]) * args.one_m_b1 + float(m_load[i]) * args.beta1);
    }
    return r;
  }
};

// ── v_new = one_m_b2 * g² + beta2 * v_load ─────────────────────────────
template <class T>
struct v_update {
  struct Arguments { float one_m_b2; float beta2; };

  CUTLASS_HOST_DEVICE T
  operator()(T const& g, T const& v_load, Arguments const& args) const {
    float gf = float(g);
    return T(gf * gf * args.one_m_b2 + float(v_load) * args.beta2);
  }
};

template <class T, int N, bool R>
struct v_update<cutlass::Array<T, N, R>> {
  using Arguments = typename v_update<T>::Arguments;

  CUTLASS_HOST_DEVICE cutlass::Array<T, N, R>
  operator()(cutlass::Array<T, N, R> const& g,
             cutlass::Array<T, N, R> const& v_load,
             Arguments const& args) const {
    cutlass::Array<T, N, R> r;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      float gf = float(g[i]);
      r[i] = T(gf * gf * args.one_m_b2 + float(v_load[i]) * args.beta2);
    }
    return r;
  }
};

// ── w_new = wd_scale*w_src + neg_lr_inv_bc1 * m_new
//                            / (sqrt_inv_bc2*sqrt(v_new) + eps) ──────────
template <class T>
struct w_update {
  struct Arguments {
    float wd_scale;
    float neg_lr_inv_bc1;
    float sqrt_inv_bc2;
    float eps;
  };

  CUTLASS_HOST_DEVICE T
  operator()(T const& w_src, T const& m_new, T const& v_new,
             Arguments const& args) const {
    float denom  = args.sqrt_inv_bc2 * ::sqrtf(float(v_new)) + args.eps;
    float update = args.neg_lr_inv_bc1 * float(m_new) / denom;
    return T(args.wd_scale * float(w_src) + update);
  }
};

template <class T, int N, bool R>
struct w_update<cutlass::Array<T, N, R>> {
  using Arguments = typename w_update<T>::Arguments;

  CUTLASS_HOST_DEVICE cutlass::Array<T, N, R>
  operator()(cutlass::Array<T, N, R> const& w_src,
             cutlass::Array<T, N, R> const& m_new,
             cutlass::Array<T, N, R> const& v_new,
             Arguments const& args) const {
    cutlass::Array<T, N, R> r;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      float denom  = args.sqrt_inv_bc2 * ::sqrtf(float(v_new[i])) + args.eps;
      float update = args.neg_lr_inv_bc1 * float(m_new[i]) / denom;
      r[i] = T(args.wd_scale * float(w_src[i]) + update);
    }
    return r;
  }
};

} // namespace adamw_fn


// ══════════════════════════════════════════════════════════════════════════
//                     Problem type choices (bf16, fp32 acc)
// ══════════════════════════════════════════════════════════════════════════
using ElementA           = cutlass::bfloat16_t;
using LayoutA            = cutlass::layout::ColumnMajor;   // GO^T
using ElementB           = cutlass::bfloat16_t;
using LayoutB            = cutlass::layout::RowMajor;      // INP
using ElementC           = cutlass::bfloat16_t;
using LayoutC            = cutlass::layout::RowMajor;      // W source
using ElementD           = cutlass::bfloat16_t;
using LayoutD            = cutlass::layout::RowMajor;      // W output
using ElementAccumulator = float;
using ElementCompute     = float;
using ElementAux         = cutlass::bfloat16_t;            // m, v dtype
using LayoutAux          = cutlass::layout::RowMajor;      // same as W

constexpr int AlignmentA   = 16 / sizeof(ElementA);        // 8 bf16 -> 16B
constexpr int AlignmentB   = 16 / sizeof(ElementB);
constexpr int AlignmentC   = 16 / sizeof(ElementC);
constexpr int AlignmentD   = 16 / sizeof(ElementD);
constexpr int AlignmentAux = 16 / sizeof(ElementAux);


// ══════════════════════════════════════════════════════════════════════════
//                   Kernel factory, parameterised by tile shape
// ══════════════════════════════════════════════════════════════════════════
template <class TileShape_,
          class ClusterShape_,
          class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto>
struct HopperEVTAdamW {
  using TileShape    = TileShape_;
  using ClusterShape = ClusterShape_;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  // With CUTLASS 3.6.0, the Sm90AuxLoad / Sm90AuxStore <Stages=0, ...>
  // specialisations take a direct global→register (register→global) path,
  // allocating zero smem and using zero TMA descriptors. That skips the
  // LDSM/STSM atoms whose SMEM-alignment requirements were the source of
  // the "Misaligned shared or local address" fault on CUTLASS 3.5.1 when
  // multiple aux TMA nodes coexisted. It also frees the smem we had been
  // giving back to aux pipelines, so the cooperative schedule (tile_m=128,
  // higher throughput on large-M shapes like lm_head) is viable again.
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  // EpilogueTile is per-variant template parameter. On 128×128 we manually
  // pick (128, 64) — it halves epilogue iterations per tile (4 → 2) and the
  // mainloop A/B still fits 4 SMEM stages. On 128×256 and 256×128 the
  // mainloop A/B footprint is larger (~48KB/stage), so enlarging EpilogueTile
  // collapses mainloop to 2 stages and net-regresses: those variants stay on
  // EpilogueTileAuto (which picks (128, 32)).
  using EpilogueTileType = EpilogueTile_;

  // CUTLASS-provided helpers that resolve Stages / EpilogueTile /
  // SmemLayoutAtom / CopyOpR2S / CopyOpS2R consistently with the
  // CollectiveBuilder's own choices. Using these keeps the aux nodes in
  // lockstep with the main C/D TMA smem layout.
  using EpiDesc = cutlass::epilogue::collective::detail::EpilogueDescriptor<
      TileShape, EpilogueTileType, ElementD, ElementD, EpilogueSchedule>;

  using AuxLoadDesc = cutlass::epilogue::collective::detail::AuxLoadDescriptor<
      EpiDesc, LayoutAux, ElementAux>;
  using AuxStoreDesc = cutlass::epilogue::collective::detail::AuxStoreDescriptor<
      EpiDesc, LayoutAux, ElementAux>;

  // Force aux pipeline Stages=1 (minimum TMA-pipelined depth). CUTLASS
  // auto picks StagesC=4 and StagesD=2 which eats ~96 KB of SMEM for the
  // four aux pipelines and leaves the mainloop WGMMA pipeline with only
  // 2 SMEM stages (measured: MainloopSm90TmaGmmaWarpSpecialized<2, ...>).
  // Bottom line: aux was stealing ~4 mainloop stages. Dropping aux to
  // Stages=1 frees ~72 KB back to mainloop, which should promote it to
  // Stages=4-5 — the regime where WGMMA hits >50% of peak (proven by
  // the Hopper GEMM fallback kernel which runs at Stages=4-7 and hits
  // 55% of peak on the same shape).
  static constexpr int AuxStages = 1;

  using AuxLoad = cutlass::epilogue::fusion::Sm90AuxLoad<
      AuxStages, typename EpiDesc::EpilogueTile,
      ElementAux,
      typename AuxLoadDesc::Stride,
      typename AuxLoadDesc::SmemLayoutAtom,
      typename AuxLoadDesc::CopyOpS2R,
      AlignmentAux>;

  using AuxStore = cutlass::epilogue::fusion::Sm90AuxStore<
      AuxStages, typename EpiDesc::EpilogueTile,
      ElementAux, RoundStyle,
      typename AuxStoreDesc::Stride,
      typename AuxStoreDesc::SmemLayoutAtom,
      typename AuxStoreDesc::CopyOpR2S,
      AlignmentAux>;

  // ─── Collapsed EVT tree (3 fat compute nodes, no ScalarBroadcast) ─────
  //
  //   m_new_stored = STORE_M( m_update(acc, m_load) )   [fp32 chain]
  //   v_new_stored = STORE_V( v_update(acc, v_load) )   [fp32 chain]
  //   w_new        = w_update(w_src, m_new_stored, v_new_stored)   [ElementD OUTPUT]
  //
  // Scalars (β1, β2, one_m_b1, one_m_b2, wd_scale, neg_lr_inv_bc1,
  // sqrt_inv_bc2, eps) live in each functor's Arguments — no separate
  // ScalarBroadcast nodes. Every compute functor is one flat inline body,
  // maximising the compiler's chance of keeping WGMMA pipelined across
  // the consumer mainloop/epilogue boundary (ptxas warning C7510).
  using Sm90SrcFetchT = cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>;
  using Sm90AccFetchT = cutlass::epilogue::fusion::Sm90AccFetch;

  // M_compute: (acc, m_load) → m_new [fp32]
  using MComputeNode = cutlass::epilogue::fusion::Sm90Compute<
      adamw_fn::m_update, ElementCompute, ElementCompute, RoundStyle>;
  using M_New_Compute = cutlass::epilogue::fusion::Sm90EVT<
      MComputeNode, Sm90AccFetchT, AuxLoad>;
  using M_New_Stored = cutlass::epilogue::fusion::Sm90EVT<AuxStore, M_New_Compute>;

  // V_compute: (acc, v_load) → v_new [fp32]
  using VComputeNode = cutlass::epilogue::fusion::Sm90Compute<
      adamw_fn::v_update, ElementCompute, ElementCompute, RoundStyle>;
  using V_New_Compute = cutlass::epilogue::fusion::Sm90EVT<
      VComputeNode, Sm90AccFetchT, AuxLoad>;
  using V_New_Stored = cutlass::epilogue::fusion::Sm90EVT<AuxStore, V_New_Compute>;

  // W_compute: (w_src, m_new, v_new) → w_new [ElementD = bf16]
  using WComputeNode = cutlass::epilogue::fusion::Sm90Compute<
      adamw_fn::w_update, ElementD, ElementCompute, RoundStyle>;
  using FusionCallbacks = cutlass::epilogue::fusion::Sm90EVT<
      WComputeNode, Sm90SrcFetchT, M_New_Stored, V_New_Stored>;

  // ─── Collective epilogue: hand the EVT tree to the builder directly. ──
  // Because FusionCallbacks (Sm90EVT<...>) is NOT a FusionOperation, the
  // CallbacksBuilder passthrough in collective_builder.hpp:97 is selected
  // and the tree is forwarded verbatim.
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      FusionCallbacks
  >::CollectiveOp;

  // Cooperative mainloop: two consumer warpgroups cooperate on one
  // tile_m=128 WGMMA tile. With aux nodes now zero-SMEM, the full smem
  // budget is available for mainloop pipelining (~4 stages on H200).
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

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA   = typename Gemm::GemmKernel::StrideA;
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using StrideD   = typename Gemm::GemmKernel::StrideD;
  using StrideAux = typename AuxLoadDesc::Stride;

  // ─── Arguments constructor (collapsed tree) ───────────────────────────
  //
  // Sm90EVT<Op, Children...>::Arguments layout is { child_args..., op_args }.
  //   AccFetch / SrcFetch leaves: op_args = {}
  //   AuxLoad  : op_args = { ptr_aux, null_default, dAux }
  //   AuxStore : op_args = { ptr_aux, dAux }
  //   Sm90Compute<Fn> : op_args = Fn::Arguments (our functor's scalar struct)
  //
  // Tree (matches type aliases above):
  //   FusionCallbacks = W_compute( SrcFetch, M_New_Stored, V_New_Stored )
  //   M_New_Stored    = AuxStore_M( M_compute( AccFetch, AuxLoad_M ) )
  //   V_New_Stored    = AuxStore_V( V_compute( AccFetch, AuxLoad_V ) )
  static typename FusionCallbacks::Arguments
  make_fusion_args(
      ElementAux* ptr_m, ElementAux* ptr_v,
      StrideAux dM, StrideAux dV,
      float wd_scale,        // 1 - lr·wd
      float neg_lr_inv_bc1,  // -lr / (1 - β1^t)
      float sqrt_inv_bc2,    // sqrt(1 / (1 - β2^t))
      float eps,             // AdamW eps outside sqrt
      float one_m_b1,        // 1 - β1
      float one_m_b2,        // 1 - β2
      float beta1,           // β1
      float beta2)           // β2
  {
    // --- M chain: STORE_M( M_compute(acc, m_load) ) ---
    typename M_New_Compute::Arguments m_compute_args{
        {},                                // AccFetch (no args)
        { ptr_m, ElementAux(0), dM },      // AuxLoad_M
        { one_m_b1, beta1 }                // M_compute op_args
    };
    typename M_New_Stored::Arguments m_stored_args{
        m_compute_args,
        { ptr_m, dM }                      // AuxStore_M op_args
    };

    // --- V chain: STORE_V( V_compute(acc, v_load) ) ---
    typename V_New_Compute::Arguments v_compute_args{
        {},                                // AccFetch
        { ptr_v, ElementAux(0), dV },      // AuxLoad_V
        { one_m_b2, beta2 }                // V_compute op_args
    };
    typename V_New_Stored::Arguments v_stored_args{
        v_compute_args,
        { ptr_v, dV }                      // AuxStore_V op_args
    };

    // --- W root: W_compute( SrcFetch, M_stored, V_stored ) ---
    typename FusionCallbacks::Arguments root_args{
        {},                                          // Sm90SrcFetch
        m_stored_args,
        v_stored_args,
        { wd_scale, neg_lr_inv_bc1, sqrt_inv_bc2, eps } // W_compute op_args
    };
    return root_args;
  }

  // ─── Launch entry point ───────────────────────────────────────────────
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
    TORCH_CHECK(inp.size(0) == BT);
    TORCH_CHECK(weight.size(0) == V && weight.size(1) == H);
    TORCH_CHECK(m.size(0) == V && m.size(1) == H);
    TORCH_CHECK(v.size(0) == V && v.size(1) == H);

    const int M = V, N = H, K = BT, L = 1;

    auto* pA = reinterpret_cast<ElementA const*>(go.data_ptr());
    auto* pB = reinterpret_cast<ElementB const*>(inp.data_ptr());
    auto* pW = reinterpret_cast<ElementC*>(weight.data_ptr());
    auto* pM = reinterpret_cast<ElementAux*>(m.data_ptr());
    auto* pV = reinterpret_cast<ElementAux*>(v.data_ptr());

    StrideA   strideA   = cutlass::make_cute_packed_stride(StrideA{},   {M, K, L});
    StrideB   strideB   = cutlass::make_cute_packed_stride(StrideB{},   {N, K, L});
    StrideC   strideC   = cutlass::make_cute_packed_stride(StrideC{},   {M, N, L});
    StrideD   strideD   = strideC;
    StrideAux strideAux = cutlass::make_cute_packed_stride(StrideAux{}, {M, N, L});

    // Precompute AdamW scalars on the host.
    const float wd_scale       = 1.0f - lr * weight_decay;
    const float inv_bc1        = 1.0f / bc1;
    const float inv_bc2        = 1.0f / bc2;
    const float neg_lr_inv_bc1 = -lr * inv_bc1;
    const float sqrt_inv_bc2   = ::sqrtf(inv_bc2);
    const float one_m_b1       = 1.0f - beta1;
    const float one_m_b2       = 1.0f - beta2;

    auto fusion_args = make_fusion_args(
        pM, pV, strideAux, strideAux,
        wd_scale, neg_lr_inv_bc1, sqrt_inv_bc2, eps,
        one_m_b1, one_m_b2, beta1, beta2);

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        { pA, strideA, pB, strideB },
        {
            fusion_args,
            pW, strideC,    // C = current W
            pW, strideD     // D = new W   (in-place)
        }
    };

    // Grouped tile ordering for L2 reuse — matches Triton GROUP_SIZE_V=8.
    // max_swizzle_size=8 walks 8 consecutive tiles along the minor axis
    // before advancing the major one. raster_order is left at Heuristic so
    // CUTLASS picks the minor axis per-shape: lm_head/gate/up (tiles_m ≫
    // tiles_n) get M-minor → 8-tile V groups; down_proj (tiles_n > tiles_m)
    // gets N-minor → 8-tile H groups. Either way, the smaller-dim operand
    // (INP for lm_head, GO for down_proj) stays warm in L2 across 8 tiles.
    args.scheduler.max_swizzle_size = 8;

    Gemm gemm;

    auto can = gemm.can_implement(args);
    TORCH_CHECK(can == cutlass::Status::kSuccess,
                "Sm90 EVT AdamW GEMM cannot implement problem (status ",
                cutlassGetStatusString(can), ")");

    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty(
        {(int64_t)workspace_size},
        torch::dtype(torch::kUInt8).device(go.device()));

    auto init = gemm.initialize(args, workspace.data_ptr());
    TORCH_CHECK(init == cutlass::Status::kSuccess,
                "Sm90 EVT AdamW init failed (", cutlassGetStatusString(init), ")");

    auto run = gemm.run();
    TORCH_CHECK(run == cutlass::Status::kSuccess,
                "Sm90 EVT AdamW run failed (", cutlassGetStatusString(run), ")");
  }
};


// ══════════════════════════════════════════════════════════════════════════
//                             Tile variants
// ══════════════════════════════════════════════════════════════════════════
// Cooperative schedule: tile_m=128. Cluster<2,1,1> enables TMA multicast of
// operand B (INP) across the 2 CTAs that share the same N-tile. Safe on
// CUTLASS 4.4+ (the SMEM alignment bug that plagued 3.5.1 multi-aux trees
// is fixed upstream).
using Cluster2x1 = Shape<_2,_1,_1>;

// All three variants use EpilogueTileAuto. With 4 aux nodes (m, v load +
// m, v store) at CUTLASS's auto-selected Stages=StagesC (typically 2), the
// epilogue SharedStorage is ~64-96 KB depending on EpilogueTile. An earlier
// revision overrode 128x128 to EpilogueTile<128,64> to halve epilogue
// iterations, but that doubled aux-SMEM per stage and — combined with
// Stages>=2 aux — starves the mainloop of SMEM (StageCountAutoCarveout
// lands at 0 and the Sm90 GMMA specialization static_asserts Stages>=2).
// Stick to Auto so CUTLASS balances the budget.
using EVT_AdamW_128x128 = HopperEVTAdamW<Shape<_128,_128,_64>, Cluster2x1>;
using EVT_AdamW_128x256 = HopperEVTAdamW<Shape<_128,_256,_64>, Cluster2x1>;
using EVT_AdamW_256x128 = HopperEVTAdamW<Shape<_256,_128,_64>, Cluster2x1>;


// ══════════════════════════════════════════════════════════════════════════
//                         Python-facing entry points
// ══════════════════════════════════════════════════════════════════════════
void hopper_evt_adamw_128x128(
    torch::Tensor go, torch::Tensor inp,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2)
{
  EVT_AdamW_128x128::launch(go, inp, weight, m, v,
                            lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}

void hopper_evt_adamw_128x256(
    torch::Tensor go, torch::Tensor inp,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2)
{
  EVT_AdamW_128x256::launch(go, inp, weight, m, v,
                            lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}

void hopper_evt_adamw_256x128(
    torch::Tensor go, torch::Tensor inp,
    torch::Tensor weight, torch::Tensor m, torch::Tensor v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bc1, float bc2)
{
  EVT_AdamW_256x128::launch(go, inp, weight, m, v,
                            lr, beta1, beta2, eps, weight_decay, bc1, bc2);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("hopper_evt_adamw_128x128", &hopper_evt_adamw_128x128,
          "Sm90 EVT-fused grad_W + AdamW (128x128x64, cluster 1x1)");
  mod.def("hopper_evt_adamw_128x256", &hopper_evt_adamw_128x256,
          "Sm90 EVT-fused grad_W + AdamW (128x256x64, cluster 1x1)");
  mod.def("hopper_evt_adamw_256x128", &hopper_evt_adamw_256x128,
          "Sm90 EVT-fused grad_W + AdamW (256x128x64, cluster 1x1)");
}
