/*
 * Sm100 EVT-fused grad_W + AdamW (B200).
 *
 * True fused kernel: grad_W is never written to HBM. The CUTLASS 4.x
 * Blackwell collective mainloop (tcgen05.mma + TMA + TMEM accumulator)
 * computes the fp32 accumulator tile-by-tile; the Sm100 epilogue then
 * applies the entire AdamW update and writes the new W back out. m and v
 * are streamed through via Sm90AuxLoad / Sm90AuxStore. (Despite the Sm90
 * prefix these node classes are arch-generic; CUTLASS 4.x reuses them
 * inside the Sm100 callbacks builder.)
 *
 * ─────────────────────────────────────────────────────────────────────────
 * Why this is a memory-bandwidth win on B200
 * ─────────────────────────────────────────────────────────────────────────
 * Without fusion:
 *   GEMM:   write grad_W (V·H·2 B) → HBM
 *   AdamW:  read grad_W, read W,m,v, write W',m',v' (8·V·H bytes)
 *   Total per layer: 10·V·H bytes of HBM traffic.
 * With fusion (this file):
 *   Single kernel: read W,m,v, write W',m',v' (6·V·H bytes).
 * 40% reduction in HBM traffic. For Llama-3.1-8B that's ~32 GB saved per
 * backward step at SeqLen=4096.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * Key Sm90 → Sm100 differences vs csrc/hopper_evt_adamw/fused_adamw_evt_sm90.cu
 * ─────────────────────────────────────────────────────────────────────────
 *   * ArchTag: Sm90 → Sm100
 *   * Mainloop schedule: KernelTmaWarpSpecializedCooperative →
 *     KernelTmaWarpSpecialized1SmSm100 / 2SmSm100 (auto-picked from cluster).
 *   * Epilogue schedule: TmaWarpSpecializedCooperative →
 *     TmaWarpSpecialized1Sm / TmaWarpSpecialized2Sm.
 *   * Descriptors: Sm90 EpilogueDescriptor → Sm100EpilogueDescriptor (more
 *     template params); Sm90 AuxLoad/StoreDescriptor → Sm100 versions.
 *   * TileShape semantics: on Sm100, TileShape is the *MMA atom* shape
 *     (per-CTA tile = TileShape / AtomThrShape). Cluster<2,1,1> → 2SM mode,
 *     atom is TileShape, per-CTA is TileShape/2 in M.
 *   * Accumulator: now in TMEM (was in RF on Sm90). The epilogue does a
 *     TMEM→RF load before our compute functors run, then RF→SMEM→TMA out.
 *
 * Arithmetic identical to H200 (standard AdamW with eps outside the sqrt).
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
// (Identical to the H200 EVT file — these are arch-agnostic.)
// ══════════════════════════════════════════════════════════════════════════
namespace adamw_fn {

template <class T> struct m_update {
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

template <class T> struct v_update {
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

template <class T> struct w_update {
  struct Arguments {
    float wd_scale; float neg_lr_inv_bc1; float sqrt_inv_bc2; float eps;
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
//                         Problem type choices
// ══════════════════════════════════════════════════════════════════════════
using ElementA           = cutlass::bfloat16_t;
using LayoutA            = cutlass::layout::ColumnMajor;   // GO^T
using ElementB           = cutlass::bfloat16_t;
using LayoutB            = cutlass::layout::RowMajor;      // INP
using ElementC           = cutlass::bfloat16_t;
using LayoutC            = cutlass::layout::RowMajor;      // W in
using ElementD           = cutlass::bfloat16_t;
using LayoutD            = cutlass::layout::RowMajor;      // W out
using ElementAccumulator = float;
using ElementCompute     = float;
using ElementAux         = cutlass::bfloat16_t;            // m, v dtype
using LayoutAux          = cutlass::layout::RowMajor;

constexpr int AlignmentA   = 16 / sizeof(ElementA);
constexpr int AlignmentB   = 16 / sizeof(ElementB);
constexpr int AlignmentC   = 16 / sizeof(ElementC);
constexpr int AlignmentD   = 16 / sizeof(ElementD);
constexpr int AlignmentAux = 16 / sizeof(ElementAux);


// ══════════════════════════════════════════════════════════════════════════
//                Kernel factory, parameterised by tile shape
// ══════════════════════════════════════════════════════════════════════════
//
// EpilogueScheduleSelector picks the right Sm100 epilogue schedule + mainloop
// schedule from the cluster shape. cluster M%2==0 → 2SM mode (cluster<2,1,1>),
// otherwise 1SM mode (cluster<1,1,1>).
template <class ClusterShape>
struct Sm100ScheduleFor {
    static constexpr bool Is2Sm = (cute::size<0>(ClusterShape{}) % 2 == 0);
    using Epilogue = cute::conditional_t<Is2Sm,
        cutlass::epilogue::TmaWarpSpecialized2Sm,
        cutlass::epilogue::TmaWarpSpecialized1Sm>;
    using Mainloop = cute::conditional_t<Is2Sm,
        cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
};

// Scheduler tag — mirrors the no-EVT BlackwellGemm tag pattern so we can
// instantiate persistent and StreamK variants from the same template.
// SchedTag::type is the value passed as the 4th template parameter of
// cutlass::gemm::kernel::GemmUniversal: void → PersistentTileSchedulerSm100
// (default), cutlass::gemm::StreamKScheduler → StreamK decomposition.
struct PersistentSched { using type = void; };
struct StreamKSched    { using type = cutlass::gemm::StreamKScheduler; };


template <class TileShape_, class ClusterShape_,
          class SchedTag = PersistentSched, int AuxStages_ = 0,
          class EpilogueTile_ = cutlass::epilogue::collective::EpilogueTileAuto>
struct BlackwellEVTAdamW {
  using TileShape    = TileShape_;
  using ClusterShape = ClusterShape_;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  using EpilogueSchedule = typename Sm100ScheduleFor<ClusterShape>::Epilogue;
  using MainloopSchedule = typename Sm100ScheduleFor<ClusterShape>::Mainloop;
  using EpilogueTileType = EpilogueTile_;

  // Sm100 descriptor — exposes EpilogueTile, StagesC, StagesD, ElementAccumulator,
  // AccLoadOp etc. for the AuxLoad/AuxStore descriptors below.
  using EpiDesc = cutlass::epilogue::collective::detail::Sm100EpilogueDescriptor<
      cutlass::arch::OpClassTensorOp,
      TileShape,
      EpilogueTileType,
      ElementAccumulator,
      ElementC,
      ElementD,
      EpilogueSchedule,
      cutlass::detail::TagToStrideC_t<LayoutC>,
      cutlass::detail::TagToStrideC_t<LayoutD>,
      /*IsPerColScaleSupported=*/false,
      /*IsBlockScaleSupported=*/false>;

  using AuxLoadDesc  = cutlass::epilogue::collective::detail::Sm100AuxLoadDescriptor<
      EpiDesc, LayoutAux, ElementAux>;
  using AuxStoreDesc = cutlass::epilogue::collective::detail::Sm100AuxStoreDescriptor<
      EpiDesc, LayoutAux, ElementAux>;

  // AuxStages = 0 — direct GMEM↔register path, zero SMEM, zero TMA descriptors.
  //
  // Why 0 (not 1+): with AuxStages>=1, Sm90AuxLoad<Stages, EpilogueTile, bf16, ...>
  // allocates a SMEM staging region; both M and V aux nodes here have the SAME
  // template type, so CUTLASS's SMEM allocator collapses them into one shared
  // region. M's TMA load fills the SMEM; V's TMA load then OVERWRITES the same
  // SMEM before the W-compute reads M's value. Step 1 was bit-exact only because
  // m_load = v_load = 0 (multiplied by 0 in the AdamW recurrence — corruption
  // invisible). Step 2+ produced wrong m_new, v_new with similar magnitude but
  // wrong values, diverging training loss. See benchmarks/ideation/_diagnose_*.
  //
  // The Stages=0 specialisations of Sm90AuxLoad/Sm90AuxStore (sm90_visitor_*.hpp)
  // have empty SharedStorage{} and read/write GMEM↔register directly, eliminating
  // the cross-aux SMEM aliasing. The H200 file documents the same fix from
  // CUTLASS 3.5.1 days; on Sm100 we hit the same class of bug because the
  // CollectiveBuilder for TmaWarpSpecialized{1,2}Sm pools aux SMEM the same way.
  // Performance cost is one extra register-pressure-equivalent of GMEM latency
  // per tile; fused_grad_adamw_evt is already HBM-bound for m,v reads, so it's
  // a noise-level tradeoff at this shape.
  //
  // T1.2 (AuxStages=2) only fits in SMEM at smaller mainloop tiles (128x128,
  // 256x128); the larger 256x256 tile statically asserts because its per-stage
  // A/B SMEM doesn't leave the mainloop carveout ≥2 stages when aux is doubled.
  static constexpr int AuxStages = AuxStages_;

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

  // ─── EVT tree (3 fat compute nodes — same shape as H200 file) ─────────
  //
  //   m_new_stored = STORE_M( m_update(acc, m_load) )         [fp32 chain]
  //   v_new_stored = STORE_V( v_update(acc, v_load) )         [fp32 chain]
  //   w_new        = w_update(w_src, m_new_stored, v_new_stored)  [bf16 = D out]
  //
  using SrcFetchT = cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>;
  using AccFetchT = cutlass::epilogue::fusion::Sm90AccFetch;

  using MComputeNode = cutlass::epilogue::fusion::Sm90Compute<
      adamw_fn::m_update, ElementCompute, ElementCompute, RoundStyle>;
  using M_New_Compute = cutlass::epilogue::fusion::Sm90EVT<
      MComputeNode, AccFetchT, AuxLoad>;
  using M_New_Stored  = cutlass::epilogue::fusion::Sm90EVT<AuxStore, M_New_Compute>;

  using VComputeNode = cutlass::epilogue::fusion::Sm90Compute<
      adamw_fn::v_update, ElementCompute, ElementCompute, RoundStyle>;
  using V_New_Compute = cutlass::epilogue::fusion::Sm90EVT<
      VComputeNode, AccFetchT, AuxLoad>;
  using V_New_Stored  = cutlass::epilogue::fusion::Sm90EVT<AuxStore, V_New_Compute>;

  using WComputeNode = cutlass::epilogue::fusion::Sm90Compute<
      adamw_fn::w_update, ElementD, ElementCompute, RoundStyle>;
  using FusionCallbacks = cutlass::epilogue::fusion::Sm90EVT<
      WComputeNode, SrcFetchT, M_New_Stored, V_New_Stored>;

  // ─── Collective epilogue: hand the EVT tree directly to CollectiveBuilder.
  // Because FusionCallbacks is NOT a FusionOperation subclass, the
  // CallbacksBuilder passthrough is selected and the tree is forwarded
  // verbatim into the Sm100 collective epilogue (same escape hatch as on H200).
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape,
      EpilogueTileType,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule,
      FusionCallbacks
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      typename SchedTag::type>;               // void → PersistentTileSchedulerSm100;
                                              // StreamKScheduler → StreamK decomposition

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA   = typename Gemm::GemmKernel::StrideA;
  using StrideB   = typename Gemm::GemmKernel::StrideB;
  using StrideC   = typename Gemm::GemmKernel::StrideC;
  using StrideD   = typename Gemm::GemmKernel::StrideD;
  using StrideAux = typename AuxLoadDesc::Stride;

  // ─── Arguments constructor for the EVT tree ───────────────────────────
  static typename FusionCallbacks::Arguments
  make_fusion_args(
      ElementAux* ptr_m, ElementAux* ptr_v,
      StrideAux dM, StrideAux dV,
      float wd_scale, float neg_lr_inv_bc1, float sqrt_inv_bc2, float eps,
      float one_m_b1, float one_m_b2, float beta1, float beta2)
  {
    typename M_New_Compute::Arguments m_compute_args{
        {},                                // AccFetch
        { ptr_m, ElementAux(0), dM },      // AuxLoad_M
        { one_m_b1, beta1 }                // M_compute op_args
    };
    typename M_New_Stored::Arguments m_stored_args{
        m_compute_args,
        { ptr_m, dM }                      // AuxStore_M op_args
    };

    typename V_New_Compute::Arguments v_compute_args{
        {},                                // AccFetch
        { ptr_v, ElementAux(0), dV },      // AuxLoad_V
        { one_m_b2, beta2 }                // V_compute op_args
    };
    typename V_New_Stored::Arguments v_stored_args{
        v_compute_args,
        { ptr_v, dV }                      // AuxStore_V op_args
    };

    typename FusionCallbacks::Arguments root_args{
        {},                                          // SrcFetch (W in via C)
        m_stored_args,
        v_stored_args,
        { wd_scale, neg_lr_inv_bc1, sqrt_inv_bc2, eps }  // W_compute op_args
    };
    return root_args;
  }

  // ─── Launch ──────────────────────────────────────────────────────────
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

    // ITER16: scheduler args — set max_swizzle_size + raster_order to
    // maximize weight-row L2 reuse. PersistentTileSchedulerSm100 exposes
    // these args at kernel/sm100_tile_scheduler.hpp:97-99. Default is
    // Heuristic+swizzle=0. swizzle=8 + AlongM keeps 8 consecutive M-tiles
    // per SM, which maximises C-operand (weight) row reuse since W rows
    // are reused across the AdamW epilogue chain.
#ifdef CUTLASS_EVT_SCHED_SWIZZLE
    args.scheduler.max_swizzle_size = CUTLASS_EVT_SCHED_SWIZZLE;
#endif
#ifdef CUTLASS_EVT_SCHED_RASTER_M
    args.scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
#endif
#ifdef CUTLASS_EVT_SCHED_RASTER_N
    args.scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;
#endif

    Gemm gemm;
    auto can = gemm.can_implement(args);
    TORCH_CHECK(can == cutlass::Status::kSuccess,
                "Sm100 EVT AdamW GEMM cannot implement problem (status ",
                cutlassGetStatusString(can), ")");

    size_t workspace_size = Gemm::get_workspace_size(args);
    auto workspace = torch::empty(
        {(int64_t)workspace_size},
        torch::dtype(torch::kUInt8).device(go.device()));

    auto init = gemm.initialize(args, workspace.data_ptr());
    TORCH_CHECK(init == cutlass::Status::kSuccess,
                "Sm100 EVT AdamW init failed (", cutlassGetStatusString(init), ")");

    auto run = gemm.run();
    TORCH_CHECK(run == cutlass::Status::kSuccess,
                "Sm100 EVT AdamW run failed (", cutlassGetStatusString(run), ")");
  }
};


// ══════════════════════════════════════════════════════════════════════════
//                             Tile variants
// ══════════════════════════════════════════════════════════════════════════
// Mirrors the 5+2 menu from cutlass_b200_gemm/fused_adamw_cutlass_b200.cu so
// the EVT picker has the same shape coverage:
//   * 128x128_1sm        — small/Q,K,V,O (M=N=4096)
//   * 256x128_2sm        — wide-M moderate (M ≥ 2N)
//   * 256x256_2sm        — big shapes with clean wave division
//   * 128x128_1sm_streamk — small shapes that need tail-wave fill
//   * 256x256_2sm_streamk — big shapes with bad fractional waves
using Cluster1x1 = Shape<_1,_1,_1>;
using Cluster2x1 = Shape<_2,_1,_1>;
using Cluster2x2 = Shape<_2,_2,_1>;

using EVT_AdamW_128x128_1sm  = BlackwellEVTAdamW<Shape<_128,_128,_64>, Cluster1x1>;
// 256x128_2sm — T2.7 attempt: forced EpilogueTile<_64,_32> to shrink
// epilogue SMEM. Compiled but bench showed ~0 net delta vs auto-pick
// (drop matched the run-to-run noise visible in the unchanged no-EVT
// row). Reverted to EpilogueTileAuto. T2.5 (AuxStages=2) likewise
// regressed; the SMEM trade-off isn't paying out at this tile size.
using EVT_AdamW_256x128_2sm  = BlackwellEVTAdamW<Shape<_256,_128,_64>, Cluster2x1>;
using EVT_AdamW_256x256_2sm  = BlackwellEVTAdamW<Shape<_256,_256,_64>, Cluster2x1>;
// ITER4: 256x256_2sm with explicit larger EpilogueTile<_128,_64> to halve
// epilogue iterations per tile (reduces AdamW-in-epilogue overhead). The
// H200 file uses this trick on its 128x128 tile. Mirror it for B200's
// largest tile to amortize epilogue scheduler/compute over fewer launches.
using EVT_AdamW_256x256_2sm_etile128x64 = BlackwellEVTAdamW<
    Shape<_256,_256,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_128,_64>>;

// ITER5+: epilogue-tile bracket. Per-CTA tile in 2SM mode is M/2 × N.
// 256x256_2sm → per-CTA 128×256.
//   <_128,_128>  → 2 epilogue iters per mainloop tile (fewest for 128×256)
//   <_64,_128>   → 4 epilogue iters
//   <_64,_64>    → 8 epilogue iters (likely worst — heavy AdamW math fires more)
// 256x128_2sm → per-CTA 128×128.
//   <_128,_128>  → 1 epilogue iter (fewest)
//   <_128,_64>   → 2 epilogue iters
// Larger EpilogueTile = fewer AdamW invocations per mainloop tile = better
// for compute-heavy EVT, but eats SMEM, leaving fewer mainloop stages.
using EVT_AdamW_256x256_2sm_etile128x128 = BlackwellEVTAdamW<
    Shape<_256,_256,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_128,_128>>;
using EVT_AdamW_256x256_2sm_etile64x64 = BlackwellEVTAdamW<
    Shape<_256,_256,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_64,_64>>;
using EVT_AdamW_256x256_2sm_etile64x128 = BlackwellEVTAdamW<
    Shape<_256,_256,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_64,_128>>;
using EVT_AdamW_256x128_2sm_etile128x128 = BlackwellEVTAdamW<
    Shape<_256,_128,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_128,_128>>;
using EVT_AdamW_256x128_2sm_etile128x64 = BlackwellEVTAdamW<
    Shape<_256,_128,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_128,_64>>;

// StreamK variants — split K-dim work across SMs to fill tail waves on
// shapes where the persistent scheduler leaves SMs idle.
using EVT_AdamW_128x128_1sm_streamk = BlackwellEVTAdamW<Shape<_128,_128,_64>, Cluster1x1, StreamKSched>;
using EVT_AdamW_256x256_2sm_streamk = BlackwellEVTAdamW<Shape<_256,_256,_64>, Cluster2x1, StreamKSched>;

// ITER10: cluster<2,2,1> with B-multicast. Splits per-CTA tile across
// 2 N-dim CTAs that share B input via TMA multicast → halves B HBM
// traffic per stage. Per-CTA tile = TileShape / Cluster:
//   Shape<_256,_256,_64>, Cluster2x2  → per-CTA 128x128
//   Shape<_256,_128,_64>, Cluster2x2  → per-CTA 128x64 — too small
using EVT_AdamW_256x256_2sm_etile128x64_c2x2 = BlackwellEVTAdamW<
    Shape<_256,_256,_64>, Cluster2x2, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_128,_64>>;

// ITER18: Cluster2x2 + smaller EpilogueTile<64,64>. Combines iter8 (c2x2)
// with iter10 (etile64x64). Halves epilogue register pressure while
// keeping the c2x2 B-multicast benefit on mlp shapes.
using EVT_AdamW_256x256_2sm_etile64x64_c2x2 = BlackwellEVTAdamW<
    Shape<_256,_256,_64>, Cluster2x2, PersistentSched, /*AuxStages=*/0,
    /*EpilogueTile=*/Shape<_64,_64>>;

// ITER8 sandbox: AuxStages=1 with current CUTLASS tip. Re-test the journey
// doc's "Sm100 aux SMEM aliasing" claim. If correctness holds on current
// CUTLASS, AuxStages=1 (TMA-staged GMEM→SMEM aux load) is faster than
// AuxStages=0 (direct GMEM→register) per-tile because the load is pipelined.
using EVT_AdamW_256x256_2sm_etile128x64_aux1 = BlackwellEVTAdamW<
    Shape<_256,_256,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/1,
    /*EpilogueTile=*/Shape<_128,_64>>;
using EVT_AdamW_256x128_2sm_aux1 = BlackwellEVTAdamW<
    Shape<_256,_128,_64>, Cluster2x1, PersistentSched, /*AuxStages=*/1>;

// T1.3 attempt: K=128 atom on 256x256 2SM — failed to compile. The K=128
// atom doubles per-stage A/B SMEM (256·128·2 + 128·128·2 ≈ 96 KB/stage in
// 2SM mode); the EVT epilogue already eats ~64 KB for src+dst+m+v+compute
// state; per-CTA SMEM (~228 KiB) only has room for 1 mainloop stage after
// that, which fails CUTLASS's `Stages >= 2` static_assert. Same root cause
// as the AuxStages=2 attempt above. K=128 needs lighter-epilogue paths.


// ══════════════════════════════════════════════════════════════════════════
//                         Python-facing entry points
// ══════════════════════════════════════════════════════════════════════════
#define EVT_ADAMW_LAUNCHER(NAME, KIND)                                        \
  void blackwell_evt_adamw_##NAME(                                            \
      torch::Tensor go, torch::Tensor inp,                                    \
      torch::Tensor weight, torch::Tensor m, torch::Tensor v,                 \
      float lr, float beta1, float beta2, float eps, float weight_decay,      \
      float bc1, float bc2)                                                   \
  {                                                                           \
    EVT_AdamW_##KIND::launch(go, inp, weight, m, v,                           \
                             lr, beta1, beta2, eps, weight_decay, bc1, bc2);  \
  }

EVT_ADAMW_LAUNCHER(128x128_1sm,                 128x128_1sm)
EVT_ADAMW_LAUNCHER(256x128_2sm,                 256x128_2sm)
EVT_ADAMW_LAUNCHER(256x256_2sm,                 256x256_2sm)
EVT_ADAMW_LAUNCHER(256x256_2sm_etile128x64,     256x256_2sm_etile128x64)
EVT_ADAMW_LAUNCHER(256x256_2sm_etile128x128,    256x256_2sm_etile128x128)
EVT_ADAMW_LAUNCHER(256x256_2sm_etile64x64,      256x256_2sm_etile64x64)
EVT_ADAMW_LAUNCHER(256x256_2sm_etile64x128,     256x256_2sm_etile64x128)
EVT_ADAMW_LAUNCHER(256x128_2sm_etile128x128,    256x128_2sm_etile128x128)
EVT_ADAMW_LAUNCHER(256x128_2sm_etile128x64,     256x128_2sm_etile128x64)
EVT_ADAMW_LAUNCHER(128x128_1sm_streamk,         128x128_1sm_streamk)
EVT_ADAMW_LAUNCHER(256x256_2sm_streamk,         256x256_2sm_streamk)
EVT_ADAMW_LAUNCHER(256x256_2sm_etile128x64_aux1, 256x256_2sm_etile128x64_aux1)
EVT_ADAMW_LAUNCHER(256x128_2sm_aux1,            256x128_2sm_aux1)
EVT_ADAMW_LAUNCHER(256x256_2sm_etile128x64_c2x2, 256x256_2sm_etile128x64_c2x2)
EVT_ADAMW_LAUNCHER(256x256_2sm_etile64x64_c2x2,  256x256_2sm_etile64x64_c2x2)


PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("blackwell_evt_adamw_128x128_1sm", &blackwell_evt_adamw_128x128_1sm,
          "Sm100 EVT-fused grad_W + AdamW (atom 128x128x64, cluster 1x1, 1SM)");
  mod.def("blackwell_evt_adamw_256x128_2sm", &blackwell_evt_adamw_256x128_2sm,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x128x64, cluster 2x1, 2SM) — wide-M");
  mod.def("blackwell_evt_adamw_256x256_2sm", &blackwell_evt_adamw_256x256_2sm,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, cluster 2x1, 2SM) — max-atom");
  mod.def("blackwell_evt_adamw_256x256_2sm_etile128x64",
          &blackwell_evt_adamw_256x256_2sm_etile128x64,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, cluster 2x1, 2SM, EpilogueTile<128,64>)");
  mod.def("blackwell_evt_adamw_256x256_2sm_etile128x128",
          &blackwell_evt_adamw_256x256_2sm_etile128x128,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, cluster 2x1, 2SM, EpilogueTile<128,128>)");
  mod.def("blackwell_evt_adamw_256x256_2sm_etile64x64",
          &blackwell_evt_adamw_256x256_2sm_etile64x64,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, cluster 2x1, 2SM, EpilogueTile<64,64>)");
  mod.def("blackwell_evt_adamw_256x256_2sm_etile64x128",
          &blackwell_evt_adamw_256x256_2sm_etile64x128,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, cluster 2x1, 2SM, EpilogueTile<64,128>)");
  mod.def("blackwell_evt_adamw_256x128_2sm_etile128x128",
          &blackwell_evt_adamw_256x128_2sm_etile128x128,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x128x64, cluster 2x1, 2SM, EpilogueTile<128,128>)");
  mod.def("blackwell_evt_adamw_256x128_2sm_etile128x64",
          &blackwell_evt_adamw_256x128_2sm_etile128x64,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x128x64, cluster 2x1, 2SM, EpilogueTile<128,64>)");
  mod.def("blackwell_evt_adamw_128x128_1sm_streamk",
          &blackwell_evt_adamw_128x128_1sm_streamk,
          "Sm100 EVT-fused grad_W + AdamW (StreamK, atom 128x128x64, cluster 1x1, 1SM)");
  mod.def("blackwell_evt_adamw_256x256_2sm_streamk",
          &blackwell_evt_adamw_256x256_2sm_streamk,
          "Sm100 EVT-fused grad_W + AdamW (StreamK, atom 256x256x64, cluster 2x1, 2SM)");
  mod.def("blackwell_evt_adamw_256x256_2sm_etile128x64_aux1",
          &blackwell_evt_adamw_256x256_2sm_etile128x64_aux1,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, 2SM, EpilogueTile<128,64>, AuxStages=1 sandbox)");
  mod.def("blackwell_evt_adamw_256x128_2sm_aux1",
          &blackwell_evt_adamw_256x128_2sm_aux1,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x128x64, 2SM, AuxStages=1 sandbox)");
  mod.def("blackwell_evt_adamw_256x256_2sm_etile128x64_c2x2",
          &blackwell_evt_adamw_256x256_2sm_etile128x64_c2x2,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, cluster 2x2, 2SM N-multicast, EpilogueTile<128,64>)");
  mod.def("blackwell_evt_adamw_256x256_2sm_etile64x64_c2x2",
          &blackwell_evt_adamw_256x256_2sm_etile64x64_c2x2,
          "Sm100 EVT-fused grad_W + AdamW (atom 256x256x64, cluster 2x2, 2SM N-multicast, EpilogueTile<64,64>)");
}
