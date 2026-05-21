#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CP_ASYNC_ENABLED
#endif

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ __launch_bounds__(Spec::kThreadNum) void dynamic_mma(void *__restrict__ Cptr,
                                                                const void *__restrict__ Aptr,
                                                                const void *__restrict__ Bptr,
                                                                int M,
                                                                int N,
                                                                int K,
                                                                void *__restrict__ Outptr) {
  using namespace cute;

  using X = Underscore;
  using MMA_shape = typename Spec::MMA_shape;
  using OutType = typename Spec::OutType;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using SmemLayoutA = typename Spec::SmemLayoutA;
  using SmemLayoutB = typename Spec::SmemLayoutB;
  using SmemLayoutC = typename Spec::SmemLayoutC;
  using SmemLayoutO = typename Spec::SmemLayoutO;

  constexpr int kBlockM = Spec::kBlockM;
  constexpr int kBlockN = Spec::kBlockN;
  constexpr int kBlockK = Spec::kBlockK;
  constexpr int kShmSizeA = Spec::kShmSizeA;
  constexpr int kShmSizeB = Spec::kShmSizeB;

  extern __shared__ __align__(1024) uint8_t smem[];

  uint8_t *Aptr_smem = smem;
  uint8_t *Bptr_smem = smem + kShmSizeA;
  uint8_t *Cptr_smem = smem + kShmSizeA + kShmSizeB;
  uint8_t *Optr_smem = smem;

  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;

  Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr), make_shape(M, K), make_stride(K, Int<1>{})); // (M, K)
  Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr), make_shape(N, K), make_stride(K, Int<1>{})); // (N, K)
  Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr), make_shape(M, N), make_stride(N, Int<1>{})); // (M, N)
  Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(M, N), make_stride(N, Int<1>{}));    // (M, N)

  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(bidy, bidx, _);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, K_TILES)
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (BLK_N, BLK_K, K_TILES)
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)
  Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

  auto m_max_coord = M - size<0>(gA) * bidy;      // M - BLK_M * m_coord
  auto n_max_coord = N - size<0>(gB) * bidx;      // N - BLK_N * n_coord
  auto k_residue = K - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

  // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
  // This aligns the tensor with BLK_K for all but the 0th k_tile
  gA = domain_offset(make_coord(0, k_residue, 0), gA);
  gB = domain_offset(make_coord(0, k_residue, 0), gB);

  Tensor sA = make_tensor(make_smem_ptr((ComputeTypeA *)Aptr_smem), SmemLayoutA{}); // (BLK_M, BLK_K)
  Tensor sB = make_tensor(make_smem_ptr((ComputeTypeB *)Bptr_smem), SmemLayoutB{}); // (BLK_N, BLK_K)
  Tensor sC = make_tensor(make_smem_ptr((ComputeTypeC *)Cptr_smem), SmemLayoutC{}); // (BLK_M, BLK_N)
  Tensor sO = make_tensor(make_smem_ptr((OutType *)Optr_smem), SmemLayoutO{});      // (BLK_M, BLK_N)

  typename Spec::TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(gC);          // (MMA, MMA_M, MMA_N)

  typename Spec::TiledCopyA_G2S g2s_tiled_copy_a;
  ThrCopy g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
  Tensor tAgA_g2s = g2s_thr_copy_a.partition_S(gA); // (ACPY, ACPY_M, ACPY_K, K_TILES)
  Tensor tAsA_g2s = g2s_thr_copy_a.partition_D(sA); // (ACPY, ACPY_M, ACPY_K)

  typename Spec::TiledCopyB_G2S g2s_tiled_copy_b;
  ThrCopy g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
  Tensor tBgB_g2s = g2s_thr_copy_b.partition_S(gB); // (BCPY, BCPY_N, BCPY_K, K_TILES)
  Tensor tBsB_g2s = g2s_thr_copy_b.partition_D(sB); // (BCPY, BCPY_N, BCPY_K)

  typename Spec::TiledCopyC_G2S g2s_tiled_copy_c;
  ThrCopy g2s_thr_copy_c = g2s_tiled_copy_c.get_slice(tid);
  Tensor tCgC_g2s = g2s_thr_copy_c.partition_S(gC); // (CCPY, CCPY_M, CCPY_N)
  Tensor tCsC_g2s = g2s_thr_copy_c.partition_D(sC); // (CCPY, CCPY_M, CCPY_N)

  //
  // PREDICATES
  //

  // Allocate predicate tensors
  Tensor tApA_g2s =
      make_tensor<bool>(make_shape(size<1>(tAsA_g2s), size<2>(tAsA_g2s)), Stride<_1, _0>{}); // (ACPY_M, ACPY_K)
  Tensor tBpB_g2s =
      make_tensor<bool>(make_shape(size<1>(tBsB_g2s), size<2>(tBsB_g2s)), Stride<_1, _0>{}); // (BCPY_N, BCPY_K)
  Tensor tCpC_g2s = make_tensor<bool>(make_shape(size<1>(tCsC_g2s), size<2>(tCsC_g2s)),
                                      Stride<_1, Int<size<1>(tCsC_g2s)>>{}); // (CCPY_M, CCPY_N)

  // Construct identity layout
  Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (BLK_N,BLK_K) -> (blk_n,blk_k)
  Tensor cC = make_identity_tensor(make_shape(size<0>(sC), size<1>(sC))); // (BLK_M,BLK_N) -> (blk_m,blk_n)

  // Repeat the partitioning with identity layouts
  Tensor tAcA_g2s = g2s_thr_copy_a.partition_S(cA); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tBcB_g2s = g2s_thr_copy_b.partition_S(cB); // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
  Tensor tCcC_g2s = g2s_thr_copy_c.partition_S(cC); // (CCPY,CCPY_M,CCPY_N) -> (blk_m,blk_n)

// Set predicates for m bounds
#pragma unroll
  for (int m = 0; m < size<0>(tApA_g2s); ++m) {
    tApA_g2s(m, 0) = get<0>(tAcA_g2s(0, m, 0)) < m_max_coord; // blk_m coord < residue_m
  }
// Set predicates for n bounds
#pragma unroll
  for (int n = 0; n < size<0>(tBpB_g2s); ++n) {
    tBpB_g2s(n, 0) = get<0>(tBcB_g2s(0, n, 0)) < n_max_coord; // blk_n coord < residue_n
  }
// Set predicates for (m,n) bounds
#pragma unroll
  for (int m = 0; m < size<0>(tCpC_g2s); ++m) {
#pragma unroll
    for (int n = 0; n < size<1>(tCpC_g2s); ++n) {
      // blk_m coord < residue_m and blk_n coord < residue_n
      tCpC_g2s(m, n) = elem_less(tCcC_g2s(0, m, n), make_coord(m_max_coord, n_max_coord));
      // Equivalent to:
      // tCpC_g2s(m,n) = (get<0>(tCcC_g2s(0,m,n)) < m_max_coord) && (get<1>(tCcC_g2s(0,m,n)) < n_max_coord);
    }
  }

  //
  // END PREDICATES
  //

  typename Spec::TiledCopyA_S2R s2r_tiled_copy_a;
  ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
  Tensor tAsA_s2r = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K)
  Tensor tArA_s2r = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB_s2r = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_M, CPY_K)
  Tensor tBrB_s2r = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_M, CPY_K)

  typename Spec::TiledCopyC_S2R s2r_tiled_copy_c;
  ThrCopy s2r_thr_copy_c = s2r_tiled_copy_c.get_slice(tid);
  Tensor tCsC_s2r = s2r_thr_copy_c.partition_S(sC); // (CPY, CPY_M, CPY_K)
  Tensor tCrC_s2r = s2r_thr_copy_c.retile_D(tCrC);  // (CPY, CPY_M, CPY_K)

  //
  // MAINLOOP
  //

  if constexpr (!IsGemm) {
    // Clear the smem tiles to account for predicated off loads
    clear(tCsC_g2s);
    copy_if(g2s_tiled_copy_c, tCpC_g2s, tCgC_g2s, tCsC_g2s);
  }

  int NTilesK = ceil_div(K, kBlockK);

  for (int ik = 0; ik < NTilesK; ++ik) {
    // Clear the smem tiles to account for predicated off loads
    clear(tAsA_g2s);
    clear(tBsB_g2s);
    if (ik == 0) {
#pragma unroll
      for (int k = 0; k < size<2>(tAsA_g2s); ++k) {
        if (get<1>(tAcA_g2s(0, 0, k)) >= -k_residue) { // blk_k coord < residue_k (gA shifted)
          copy_if(g2s_tiled_copy_a, tApA_g2s(_, k), tAgA_g2s(_, _, k, ik), tAsA_g2s(_, _, k));
        }
      }

#pragma unroll
      for (int k = 0; k < size<2>(tBsB_g2s); ++k) {
        if (get<1>(tBcB_g2s(0, 0, k)) >= -k_residue) { // blk_k coord < residue_k (gB shifted)
          copy_if(g2s_tiled_copy_b, tBpB_g2s(_, k), tBgB_g2s(_, _, k, ik), tBsB_g2s(_, _, k));
        }
      }
    } else {
      copy_if(g2s_tiled_copy_a, tApA_g2s, tAgA_g2s(_, _, _, ik), tAsA_g2s);
      copy_if(g2s_tiled_copy_b, tBpB_g2s, tBgB_g2s(_, _, _, ik), tBsB_g2s);
    }

#if defined(CP_ASYNC_ENABLED)
    cp_async_fence();
    cp_async_wait<0>();
#endif
    __syncthreads();

    if (ik == 0) {
      if constexpr (IsGemm) {
        clear(tCrC); // Set the accumulators to zero
      } else {
        copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r);
      }
    }

    copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r);
    copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r);

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    __syncthreads();
  }

  if constexpr (!IsCvtPrecision) {
    typename Spec::TiledCopyC_R2S r2s_tiled_copy_c;
    ThrCopy r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);
    Tensor tCrC_r2s = r2s_thr_copy_c.retile_S(tCrC);  // (CPY, CPY_M, CPY_N)
    Tensor tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);

    __syncthreads();

    typename Spec::TiledCopyC_S2G s2g_tiled_copy_c;
    ThrCopy s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(tid);
    Tensor tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, CPY_M, CPY_N)
    Tensor tCgC_s2g = s2g_thr_copy_c.partition_D(gC); // (CPY, CPY_M, CPY_N)

    //
    // PREDICATES
    //

    Tensor tCpC_s2g = make_tensor<bool>(make_shape(size<1>(tCgC_s2g), size<2>(tCgC_s2g)),
                                        Stride<_1, Int<size<1>(tCgC_s2g)>>{}); // (CCPY_M, CCPY_N)
    Tensor tCcC_s2g = s2g_thr_copy_c.partition_S(cC);                          // (CCPY,CCPY_M,CCPY_N) -> (blk_m,blk_n)

#pragma unroll
    for (int m = 0; m < size<0>(tCpC_s2g); ++m) {
#pragma unroll
      for (int n = 0; n < size<1>(tCpC_s2g); ++n) {
        tCpC_s2g(m, n) = elem_less(tCcC_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
      }
    }

    //
    // END PREDICATES
    //

    copy_if(s2g_tiled_copy_c, tCpC_s2g, tCsC_s2g, tCgC_s2g);

  } else {

    auto t = make_tensor_like<OutType>(tCrC);
    copy(tCrC, t); // Convert precision

    typename Spec::TiledCopyO_R2S r2s_tiled_copy_o;
    ThrCopy r2s_thr_copy_o = r2s_tiled_copy_o.get_slice(tid);
    Tensor tOrC_r2s = r2s_thr_copy_o.retile_S(t);     // (CPY, CPY_M, CPY_N)
    Tensor tOsO_r2s = r2s_thr_copy_o.partition_D(sO); // (CPY, CPY_M, CPY_N)
    copy(r2s_tiled_copy_o, tOrC_r2s, tOsO_r2s);

    __syncthreads();

    typename Spec::TiledCopyO_S2G s2g_tiled_copy_o;
    ThrCopy s2g_thr_copy_o = s2g_tiled_copy_o.get_slice(tid);
    Tensor tOsO_s2g = s2g_thr_copy_o.partition_S(sO); // (CPY, CPY_M, CPY_N)
    Tensor tOgO_s2g = s2g_thr_copy_o.partition_D(gO); // (CPY, CPY_M, CPY_N)

    //
    // PREDICATES
    //

    Tensor tOpO_s2g = make_tensor<bool>(make_shape(size<1>(tOgO_s2g), size<2>(tOgO_s2g)),
                                        Stride<_1, Int<size<1>(tOgO_s2g)>>{}); // (OCPY_M, OCPY_N)
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tOcO_s2g = s2g_thr_copy_o.partition_S(cO);                          // (OCPY,OCPY_M,OCPY_N) -> (blk_m,blk_n)

#pragma unroll
    for (int m = 0; m < size<0>(tOpO_s2g); ++m) {
#pragma unroll
      for (int n = 0; n < size<1>(tOpO_s2g); ++n) {
        tOpO_s2g(m, n) = elem_less(tOcO_s2g(0, m, n), make_coord(m_max_coord, n_max_coord));
      }
    }

    //
    // END PREDICATES
    //

    copy_if(s2g_tiled_copy_o, tOpO_s2g, tOsO_s2g, tOgO_s2g);
  }
}

namespace spec {

using namespace cute;

template <typename OutType_,
          typename ComputeTypeA_,
          typename ComputeTypeB_,
          typename ComputeTypeC_,
          int kBlockM_ = 128,
          int kBlockN_ = 128,
          int kBlockK_ = 64>
struct KernelSpec {
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kBlockK = kBlockK_;

  using MMA_op = std::conditional_t<
      std::is_same_v<ComputeTypeA, cute::bfloat16_t> && std::is_same_v<ComputeTypeB, cute::bfloat16_t> &&
          std::is_same_v<ComputeTypeC, float>,
      SM80_16x8x16_F32BF16BF16F32_TN,
      std::conditional_t<
          std::is_same_v<ComputeTypeA, cute::half_t> && std::is_same_v<ComputeTypeB, cute::half_t> &&
              std::is_same_v<ComputeTypeC, cute::half_t>,
          SM80_16x8x16_F16F16F16F16_TN,
          std::conditional_t<std::is_same_v<ComputeTypeA, cute::half_t> &&
                                 std::is_same_v<ComputeTypeB, cute::half_t> && std::is_same_v<ComputeTypeC, float>,
                             SM80_16x8x16_F32F16F16F32_TN,
                             void>>>;

  static_assert(!std::is_same_v<MMA_op, void>, "Unsupported MMA op!");

  using MMA_traits = MMA_Traits<MMA_op>;
  using MMA_atom = MMA_Atom<MMA_traits>;
  using MMA_shape = typename MMA_traits::Shape_MNK;

  static constexpr int kMmaThrExpandM = 2;
  static constexpr int kMmaThrExpandN = 4;
  static constexpr int kMmaThrExpandK = 1;

  static constexpr int kMmaValExpandM = 1;
  static constexpr int kMmaValExpandN = 1;
  static constexpr int kMmaValExpandK = 2;

  static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
  static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
  static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});

  using MMAThrLayout =
      decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
  using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;

  using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

  // Why AutoVectorizingCopy faults under copy_if (CUDA error 716, "misaligned
  // address"):
  //
  //   `AutoVectorizingCopyWithAssumedAlignment<MaxBits>` inherits from
  //   `UniversalCopy<uint_bit_t<MaxBits>>`, so `AutoVectorizingCopy` (=
  //   `<128>`) is structurally a 128-bit atom applied to whatever element
  //   type the tensor has.
  //
  //   In `cute/algorithm/copy.hpp`, the `copy()` overload for AutoVec does a
  //   `recast<uint_bit_t<vec_bits>>` of src/dst BEFORE issuing the atom:
  //
  //       copy(AutoVec<N>, src, dst)
  //         -> recast<uintN>(src/dst)        // fp16 -> uint128_t view
  //         -> copy_if(true, src_v, dst_v)   // 1 atom call per iter
  //
  //   But `copy_if(Copy_Atom<AutoVec<N>, T>, pred, src, dst)` has NO matching
  //   recast specialization. It falls through to the generic Copy_Atom
  //   path that unrolls atom calls over the per-thread tile WITHOUT
  //   recasting. With val (1, 8) of fp16 we get 8 atom calls at stride
  //   1 fp16 (= 2 B), each invoking the 128-bit atom:
  //
  //       ld.global.nc.v2.u64 [base + 0]    aligned
  //       ld.global.nc.v2.u64 [base + 2]    misaligned -> fault
  //       ld.global.nc.v2.u64 [base + 4]    misaligned
  //       ...
  //       ld.global.nc.v2.u64 [base + 14]   misaligned
  //
  //   These loads also have NO @p in PTX: copy_if's `if (pred)` becomes one
  //   coarse `setp + @p bra` gating the whole block. NVCC faithfully lowers
  //   exactly what CUTLASS asked for; the broken stride is at the CUTLASS
  //   layer, not at NVCC.
  //
  // Tracked upstream: NVIDIA/cutlass#2354 ("Missing copy_if implementation
  // for AutoVectorizingCopyWithAssumedAlignment"). As of CUTLASS main the
  // bug is still present -- no copy_if(AutoVec<N>, ...) specialization has
  // been merged. The official workaround in
  // cutlass/examples/cute/tutorial/tiled_copy_if.cu is to bake the width
  // into the atom type explicitly:
  //
  //   using CopyOp = UniversalCopy<uint_byte_t<sizeof(T) * size(val_layout)>>;
  //
  // SM80_CP_ASYNC_CACHEGLOBAL<uint128_t> follows the same principle: its
  // atom is 16 B intrinsically, so val (1, 8) collapses to one atom call per
  // iter -> one aligned `cp.async.cg.shared.global [smem], [gmem], 16` per
  // thread. It also drives the cp_async_fence / cp_async_wait pipeline
  // (AutoVec is sync ld+st, the fences become no-ops with respect to data
  // motion).
  using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

  // Note: ldmatrix only support 16-bit data type (or below)
  using Copy_S2R_op_A = std::conditional_t<sizeof(ComputeTypeA) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  using Copy_S2R_op_B = std::conditional_t<sizeof(ComputeTypeB) == 2, SM75_U32x4_LDSM_N, AutoVectorizingCopy>;
  // The C / O accumulator fragment has no K val-expand, so each thread only
  // owns half the 32-bit packets of an A/B fragment. Use the x2 LDSM/STSM
  // variant for C-side copies; A/B keep x4 because VAL_EXPAND_K=2 gives them
  // enough vals/thread.
  using Copy_S2R_op_C = std::conditional_t<sizeof(ComputeTypeC) == 2, SM75_U32x2_LDSM_N, AutoVectorizingCopy>;

  using CopyA_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeA>;
  using CopyB_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeB>;
  using CopyC_G2S_atom = Copy_Atom<Copy_G2S_op, ComputeTypeC>;

  using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op_A, ComputeTypeA>;
  using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op_B, ComputeTypeB>;
  using CopyC_S2R_atom = Copy_Atom<Copy_S2R_op_C, ComputeTypeC>;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // R2S writes a C-shaped fragment, so match the C-side x2 variant chosen
  // above. SM90_U32x4_STSM_N would static_assert on too-few vals/thread.
  using Copy_R2S_op = SM90_U32x2_STSM_N;
#else
  using Copy_R2S_op = AutoVectorizingCopy;
#endif

  using Copy_S2G_op = UniversalCopy<cute::uint128_t>;

  using CopyC_R2S_atom = Copy_Atom<Copy_R2S_op, ComputeTypeC>;
  using CopyO_R2S_atom = Copy_Atom<Copy_R2S_op, OutType>;

  using CopyC_S2G_atom = Copy_Atom<Copy_S2G_op, ComputeTypeC>;
  using CopyO_S2G_atom = Copy_Atom<Copy_S2G_op, OutType>;

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kBlockK_Copy = cute::min(64, kBlockK) / 8;
  static constexpr int kBlockN_Copy = cute::min(64, kBlockN) / 8;

  using TiledCopyA_G2S =
      decltype(make_tiled_copy(CopyA_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}),
                                           make_stride(Int<kBlockK_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using TiledCopyB_G2S =
      decltype(make_tiled_copy(CopyB_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockK_Copy>{}, Int<kBlockK_Copy>{}),
                                           make_stride(Int<kBlockK_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using TiledCopyC_G2S =
      decltype(make_tiled_copy(CopyC_G2S_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}),
                                           make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_S2R_atom{}, TiledMMA{}));
  using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_S2R_atom{}, TiledMMA{}));
  using TiledCopyC_S2R = decltype(make_tiled_copy_C(CopyC_S2R_atom{}, TiledMMA{}));

  using TiledCopyC_R2S = decltype(make_tiled_copy_C(CopyC_R2S_atom{}, TiledMMA{}));
  using TiledCopyO_R2S = decltype(make_tiled_copy_C(CopyO_R2S_atom{}, TiledMMA{}));

  using TiledCopyC_S2G =
      decltype(make_tiled_copy(CopyC_S2G_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}),
                                           make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using TiledCopyO_S2G =
      decltype(make_tiled_copy(CopyO_S2G_atom{},
                               make_layout(make_shape(Int<kThreadNum / kBlockN_Copy>{}, Int<kBlockN_Copy>{}),
                                           make_stride(Int<kBlockN_Copy>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  using SmemLayoutAtomAB = decltype(composition(Swizzle<3, 3, 3>{},
                                                make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockK)>{}),
                                                            make_stride(Int<cute::min(64, kBlockK)>{}, Int<1>{}))));
  using SmemLayoutAtomC = decltype(composition(Swizzle<3, 3, 3>{},
                                               make_layout(make_shape(Int<8>{}, Int<cute::min(64, kBlockN)>{}),
                                                           make_stride(Int<cute::min(64, kBlockN)>{}, Int<1>{}))));
  using SmemLayoutA =
      decltype(tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<kBlockM>{}, Int<kBlockK>{}), Step<_1, _2>{}));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtomAB{}, make_shape(Int<kBlockN>{}, Int<kBlockK>{}), Step<_1, _2>{}));
  using SmemLayoutC =
      decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{}), Step<_1, _2>{}));
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{}), Step<_1, _2>{}));

  static constexpr int kShmSizeA = cosize_v<SmemLayoutA> * sizeof(ComputeTypeA);
  static constexpr int kShmSizeB = cosize_v<SmemLayoutB> * sizeof(ComputeTypeB);
  static constexpr int kShmSizeC = cosize_v<SmemLayoutC> * sizeof(ComputeTypeC);
  static constexpr int kShmSizeO = cosize_v<SmemLayoutO> * sizeof(OutType);

  static constexpr int kShmSize = cute::max(kShmSizeA + kShmSizeB + kShmSizeC, kShmSizeO);
};

} // namespace spec

#define CHECK_TORCH_TENSOR_DTYPE(T, DTYPE)                                                                            \
  do {                                                                                                                \
    if ((T).options().dtype() != (DTYPE)) {                                                                           \
      std::cerr << "Tensor dtype mismatch! Expected: " << (DTYPE) << ", but got: " << (T).options().dtype() << " at " \
                << __FILE__ << ":" << __LINE__ << std::endl;                                                          \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define CHECK_TORCH_TENSOR_SHAPE(T, M, N)                                                                             \
  do {                                                                                                                \
    auto actual_shape = (T).sizes();                                                                                  \
    if (actual_shape != torch::IntArrayRef({M, N})) {                                                                 \
      std::cerr << "Tensor shape mismatch! Expected: " << torch::IntArrayRef({M, N}) << ", but got: " << actual_shape \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                                                \
      std::exit(EXIT_FAILURE);                                                                                        \
    }                                                                                                                 \
  } while (0);

#define BOOL_SWITCH(COND, CONST_NAME, ...)                                                                            \
  [&] {                                                                                                               \
    if (COND) {                                                                                                       \
      constexpr static bool CONST_NAME = true;                                                                        \
      return __VA_ARGS__();                                                                                           \
    } else {                                                                                                          \
      constexpr static bool CONST_NAME = false;                                                                       \
      return __VA_ARGS__();                                                                                           \
    }                                                                                                                 \
  }()

template <typename T> constexpr torch::ScalarType to_torch_scalar_type() {
  if constexpr (std::is_same_v<T, cute::half_t>)
    return torch::kHalf;
  else if constexpr (std::is_same_v<T, cute::bfloat16_t>)
    return torch::kBFloat16;
  else if constexpr (std::is_same_v<T, float>)
    return torch::kFloat32;
  else if constexpr (std::is_same_v<T, cute::float_e4m3_t>)
    return torch::kFloat8_e4m3fn;
  else if constexpr (std::is_same_v<T, cute::float_e5m2_t>)
    return torch::kFloat8_e5m2;
  else
    throw std::runtime_error("Unsupported type!");
}

template <typename ComputeTypeC, typename OutType> constexpr bool needs_precision_conversion() {
  return !std::is_same_v<ComputeTypeC, OutType>;
}

template <int kBlockM,
          int kBlockN,
          int kBlockK,
          typename OutType,
          typename ComputeTypeA,
          typename ComputeTypeB,
          typename ComputeTypeC = OutType>
torch::Tensor run_dynamic_mma(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto M = a.size(0);
  auto N = b.size(0);
  auto K = a.size(1);

  auto torch_compute_type_a = to_torch_scalar_type<ComputeTypeA>();
  auto torch_compute_type_b = to_torch_scalar_type<ComputeTypeB>();
  auto torch_compute_type_c = to_torch_scalar_type<ComputeTypeC>();

  torch::Tensor c, out;
  bool is_gemm;

  if (!_c.has_value()) {
    auto options = torch::TensorOptions().dtype(torch_compute_type_c).device(torch::kCUDA);
    c = torch::empty({M, N}, options);
    is_gemm = true;
  } else {
    c = _c.value();
    is_gemm = false;
  }

  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type_a)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type_b)
  CHECK_TORCH_TENSOR_DTYPE(c, torch_compute_type_c)

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  constexpr bool IsCvtPrecision = needs_precision_conversion<ComputeTypeC, OutType>();

  if constexpr (IsCvtPrecision) {
    auto torch_compute_type_out = to_torch_scalar_type<OutType>();
    auto options = torch::TensorOptions().dtype(torch_compute_type_out).device(torch::kCUDA);
    out = torch::empty({M, N}, options);

    CHECK_TORCH_TENSOR_DTYPE(out, torch_compute_type_out)
    CHECK_TORCH_TENSOR_SHAPE(out, M, N)
  }

  using Spec = spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, kBlockM, kBlockN, kBlockK>;

  dim3 block = Spec::kThreadNum;
  dim3 grid(cute::ceil_div(N, Spec::kBlockN), cute::ceil_div(M, Spec::kBlockM));
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
         block.z, grid.x, grid.y, grid.z, shm_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();

  auto get_data_ptr = [](const torch::Tensor &tensor) -> void * {
    return tensor.defined() ? tensor.data_ptr() : nullptr;
  };
  void *out_ptr = get_data_ptr(out);

  // Kernel launch
  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    cudaEventRecord(start, stream);
    if (shm_size >= 48 * 1024) {
      cudaFuncSetAttribute(dynamic_mma<Spec, IsGemm, IsCvtPrecision>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           shm_size);
    }
    dynamic_mma<Spec, IsGemm, IsCvtPrecision>
        <<<grid, block, shm_size, stream>>>(c.data_ptr(), a.data_ptr(), b.data_ptr(), M, N, K, out_ptr);
    cudaEventRecord(stop, stream);
  });

  cudaDeviceSynchronize();

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) +
                             " (error code: " + std::to_string(error) + ")");
  }

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if constexpr (IsCvtPrecision)
    return out;
  else
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dynamic_mma_fp16_fp16_fp16_fp16", &(run_dynamic_mma<128, 128, 64, cute::half_t, cute::half_t, cute::half_t>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("dynamic_mma_fp16_fp16_fp16_fp32",
        &(run_dynamic_mma<128, 128, 64, cute::half_t, cute::half_t, cute::half_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
  m.def("dynamic_mma_bf16_bf16_bf16_fp32",
        &(run_dynamic_mma<128, 128, 64, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision half 16x8x8 MMA operation.");
}
