"""Block-grid GEMM with predicated cp.async + swizzled smem in CuTe DSL.

CuTe DSL counterpart of ``dynamic_mma.cu`` / ``dynamic_mma.py``. Extends
``07-swizzling`` along two axes:

  1. **Block grid**: the kernel now sweeps a 2-D block grid over
     ``ceil_div(M, kBlockM) x ceil_div(N, kBlockN)``, rather than a
     single (1, 1, 1) grid.
  2. **Dynamic K-tile mainloop**: per block, the kernel iterates over
     ``ceil_div(K, kBlockK)`` K-tiles, performing predicated G2S copies
     plus S2R + tensor-core GEMM each iteration.
  3. **Out-of-bounds predication**: ``M``, ``N`` and ``K`` are no longer
     assumed to be multiples of the block tile. The kernel builds
     identity tensors, partitions them like the operands, and uses the
     resulting (m, k) / (n, k) / (m, n) coordinates against the live
     ``M / N / K`` to predicate every G2S / S2G copy.

The swizzled smem path from ``07-swizzling`` is reused verbatim:
``Swizzle<3,3,3>`` composed with an ``(8 x min(64, kBlockK))`` atom,
then tiled to ``(kBlockM, kBlockK)`` / ``(kBlockN, kBlockK)``
ComposedLayouts handed to ``allocate_tensor``. No pipelining here, so
plain 2D smem layouts suffice.

Three dtype specializations are exercised, matching ``dynamic_mma.py``:

  * fp16 in, fp16 acc, fp16 out
  * fp16 in, fp32 acc, fp16 out  (precision-conversion path)
  * bf16 in, fp32 acc, bf16 out  (precision-conversion path)

Run with ``python cutedsl_dynamic_mma.py``.
"""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream


# Block tile (matches dynamic_mma.cu's default KernelSpec defaults)
BLK_M = 128
BLK_N = 128
BLK_K = 64

MMA_INST_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 4, 1)
VAL_EXPAND_MNK = (1, 1, 2)
MMA_TILE_MNK = (
    ATOM_LAYOUT_MNK[0] * VAL_EXPAND_MNK[0] * MMA_INST_MNK[0],  # 32
    ATOM_LAYOUT_MNK[1] * VAL_EXPAND_MNK[1] * MMA_INST_MNK[1],  # 32
    ATOM_LAYOUT_MNK[2] * VAL_EXPAND_MNK[2] * MMA_INST_MNK[2],  # 32
)
NUM_THREADS = (
    ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2] * 32  # 256
)


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def dynamic_mma_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    tiled_mma: cute.TiledMma,
    g2s_tiled_copy_a: cute.TiledCopy,
    g2s_tiled_copy_b: cute.TiledCopy,
    g2s_tiled_copy_c: cute.TiledCopy,
    s2r_tiled_copy_a: cute.TiledCopy,
    s2r_tiled_copy_b: cute.TiledCopy,
    s2r_tiled_copy_c: cute.TiledCopy,
    r2s_tiled_copy_o: cute.TiledCopy,
    s2g_tiled_copy_o: cute.TiledCopy,
    sA_layout: cute.ComposedLayout,
    sB_layout: cute.ComposedLayout,
    sC_layout: cute.ComposedLayout,
    sO_layout: cute.ComposedLayout,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    M, K = mA.shape
    N, _ = mB.shape

    # 3-D block tiler so gA / gB carry a K-tile iteration mode.
    tiler = (BLK_M, BLK_N, BLK_K)
    # gA: (BLK_M, BLK_K, k_tiles), gB: (BLK_N, BLK_K, k_tiles)
    gA = cute.local_tile(mA, tiler=tiler, coord=(bidy, bidx, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=tiler, coord=(bidy, bidx, None), proj=(None, 1, 1))
    # gC, gO: (BLK_M, BLK_N) -- single-tile in the (M, N) directions
    gC = cute.local_tile(mC, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))
    gO = cute.local_tile(mO, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))

    # Per-block residue clamps (bounds the active region in this block).
    m_max = M - BLK_M * bidy
    n_max = N - BLK_N * bidx
    # k_residue is non-positive: how far the first K-tile's origin sits
    # *before* the true K=0. Shifting gA/gB by k_residue along K makes
    # every full K-tile (except the first) trivially in-bounds.
    k_tiles = cute.size(gA, mode=[2])
    k_residue = K - BLK_K * k_tiles
    gA = cute.domain_offset((0, k_residue, 0), gA)
    gB = cute.domain_offset((0, k_residue, 0), gB)

    # ----- Smem allocation (swizzled A/B, plain C, swizzled O) -----
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, byte_alignment=16)
    sC = smem.allocate_tensor(mC.element_type, sC_layout, byte_alignment=16)
    sO = smem.allocate_tensor(out_dtype, sO_layout, byte_alignment=16)

    # ----- G2S partitions (predicated) -----
    thr_g2s_a = g2s_tiled_copy_a.get_slice(tid)
    tAgA = thr_g2s_a.partition_S(gA)  # (CPY, CPY_M, CPY_K, k_tiles)
    tAsA = thr_g2s_a.partition_D(sA)  # (CPY, CPY_M, CPY_K)

    thr_g2s_b = g2s_tiled_copy_b.get_slice(tid)
    tBgB = thr_g2s_b.partition_S(gB)  # (CPY, CPY_N, CPY_K, k_tiles)
    tBsB = thr_g2s_b.partition_D(sB)  # (CPY, CPY_N, CPY_K)

    thr_g2s_c = g2s_tiled_copy_c.get_slice(tid)
    tCgC = thr_g2s_c.partition_S(gC)  # (CPY, CPY_M, CPY_N)
    tCsC = thr_g2s_c.partition_D(sC)  # (CPY, CPY_M, CPY_N)

    # ----- Identity tensors for predication -----
    cA = cute.make_identity_tensor((BLK_M, BLK_K))
    cB = cute.make_identity_tensor((BLK_N, BLK_K))
    cC = cute.make_identity_tensor((BLK_M, BLK_N))
    tAcA = thr_g2s_a.partition_S(cA)  # (CPY, CPY_M, CPY_K)
    tBcB = thr_g2s_b.partition_S(cB)  # (CPY, CPY_N, CPY_K)
    tCcC = thr_g2s_c.partition_S(cC)  # (CPY, CPY_M, CPY_N)

    # ----- M / N predicate (used for ik >= 1, i.e. all but the residue tile) -----
    # Three-mode layout (rest_v, CPY_M, CPY_K) with the K mode broadcast at
    # stride 0 so the same M/N predicate is replayed across every K iter.
    # K-bound is not needed here because the domain_offset shift guarantees
    # every K-tile from ik=1 onward is in-bounds; only the residue tile
    # (handled by ``tApA_first`` below) needs the per-element K check.
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (
                tAgA.shape[0][1],
                cute.size(tAsA, mode=[1]),
                cute.size(tAsA, mode=[2]),
            ),
            stride=(cute.size(tAsA, mode=[1]), 1, 0),
        ),
        cutlass.Boolean,
    )
    tBpB = cute.make_rmem_tensor(
        cute.make_layout(
            (
                tBgB.shape[0][1],
                cute.size(tBsB, mode=[1]),
                cute.size(tBsB, mode=[2]),
            ),
            stride=(cute.size(tBsB, mode=[1]), 1, 0),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for m in cutlass.range_constexpr(tApA.shape[1]):
            tApA[rest_v, m, 0] = cute.elem_less(
                tAcA[(0, rest_v), m, 0][0],
                m_max,
            )
    for rest_v in cutlass.range_constexpr(tBpB.shape[0]):
        for n in cutlass.range_constexpr(tBpB.shape[1]):
            tBpB[rest_v, n, 0] = cute.elem_less(
                tBcB[(0, rest_v), n, 0][0],
                n_max,
            )

    # ----- Preload C (if accumulating) -----
    # No dynamic loop here so we can inline a per-(m, n) bound check.
    # Out-of-bounds slots are left at zero from the smem fill.
    if cutlass.const_expr(not is_gemm):
        tCsC.fill(0)
        for m in cutlass.range_constexpr(cute.size(tCgC, mode=[1])):
            for n in cutlass.range_constexpr(cute.size(tCgC, mode=[2])):
                if cute.elem_less(tCcC[0, m, n][0], m_max) and cute.elem_less(
                    tCcC[0, m, n][1],
                    n_max,
                ):
                    cute.copy(
                        g2s_tiled_copy_c,
                        tCgC[None, m, n],
                        tCsC[None, m, n],
                    )

    # ----- Fragments -----
    thr_mma = tiled_mma.get_slice(tid)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gC))

    thr_s2r_a = s2r_tiled_copy_a.get_slice(tid)
    thr_s2r_b = s2r_tiled_copy_b.get_slice(tid)
    thr_s2r_c = s2r_tiled_copy_c.get_slice(tid)

    # ----- Mainloop: per K-tile predicated G2S, then S2R + MMA -----
    num_k_tiles = cute.size(tAgA, mode=[3])

    # Residue tile (ik = 0) needs a per-element predicate combining M-bound
    # AND K-bound. The per-element K check (rather than the C++ iteration-
    # level ``thread-0 K coord >= -k_residue`` gate) is required because
    # with CPY_K = 1 a single iter carries lanes spread across the whole
    # BLK_K range — some at valid in-bounds K-positions, others not — so the
    # gate must fire per lane / per val element rather than per iter.
    tApA_first = cute.make_rmem_tensor(
        cute.make_layout(
            (
                tAgA.shape[0][1],
                cute.size(tAsA, mode=[1]),
                cute.size(tAsA, mode=[2]),
            ),
            stride=(
                cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                cute.size(tAsA, mode=[2]),
                1,
            ),
        ),
        cutlass.Boolean,
    )
    tBpB_first = cute.make_rmem_tensor(
        cute.make_layout(
            (
                tBgB.shape[0][1],
                cute.size(tBsB, mode=[1]),
                cute.size(tBsB, mode=[2]),
            ),
            stride=(
                cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                cute.size(tBsB, mode=[2]),
                1,
            ),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA_first.shape[0]):
        for m in cutlass.range_constexpr(tApA_first.shape[1]):
            for k in cutlass.range_constexpr(tApA_first.shape[2]):
                tApA_first[rest_v, m, k] = cute.elem_less(
                    tAcA[(0, rest_v), m, k][0],
                    m_max,
                ) and cute.elem_less(
                    cutlass.Int32(-1),
                    tAcA[(0, rest_v), m, k][1] + k_residue,
                )
    for rest_v in cutlass.range_constexpr(tBpB_first.shape[0]):
        for n in cutlass.range_constexpr(tBpB_first.shape[1]):
            for k in cutlass.range_constexpr(tBpB_first.shape[2]):
                tBpB_first[rest_v, n, k] = cute.elem_less(
                    tBcB[(0, rest_v), n, k][0],
                    n_max,
                ) and cute.elem_less(
                    cutlass.Int32(-1),
                    tBcB[(0, rest_v), n, k][1] + k_residue,
                )

    # Zero the A/B smem ONCE before the K-loop (hoisted out of the mainloop).
    # The G2S copy is predicated, so predicated-off slots are never written by
    # cp.async; pre-zeroing lets them read as 0. A single clear suffices: the
    # M-boundary rows masked off by the M-only predicate are never written by
    # any K-tile, and the ik=0 K-residue columns are overwritten by every
    # later (full) K-tile.
    #
    # No sync between the fill and the cp.async: each thread's fill targets the
    # same g2s partition slots that its cp.async then writes, so program order
    # within a thread suffices. The post-wait_group sync_threads below is what
    # publishes both the zeros and the cp.async data to the s2r readers.
    tAsA.fill(0)
    tBsB.fill(0)
    cute.copy(
        g2s_tiled_copy_a,
        tAgA[None, None, None, 0],
        tAsA,
        pred=tApA_first,
    )
    cute.copy(
        g2s_tiled_copy_b,
        tBgB[None, None, None, 0],
        tBsB,
        pred=tBpB_first,
    )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    if cutlass.const_expr(is_gemm):
        tCrC.fill(0.0)
    else:
        # Load C from smem in its native dtype (ComputeTypeC), then convert
        # into the accumulator fragment. The intermediate is required when
        # the accumulator dtype differs from ComputeTypeC (e.g. fp16->fp32):
        # the DSL's ``cute.copy`` requires source/destination bit widths to
        # match, so the dtype change has to happen on a register-to-register
        # ``.to()`` step.
        tCrC_pre = cute.make_fragment_like(tCrC, mC.element_type)
        cute.copy(
            s2r_tiled_copy_c,
            thr_s2r_c.partition_S(sC),
            thr_s2r_c.retile(tCrC_pre),
        )
        tCrC.store(tCrC_pre.load().to(tCrC.element_type))

    cute.copy(
        s2r_tiled_copy_a,
        thr_s2r_a.partition_S(sA),
        thr_s2r_a.retile(tCrA),
    )
    cute.copy(
        s2r_tiled_copy_b,
        thr_s2r_b.partition_S(sB),
        thr_s2r_b.retile(tCrB),
    )
    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
    cute.arch.sync_threads()

    # Remaining K-tiles (ik = 1..num_k_tiles-1) — full M/N predication only,
    # K is guaranteed in-bounds by the residue shift.
    for ik in cutlass.range(num_k_tiles - 1, unroll_full=False):
        cute.copy(
            g2s_tiled_copy_a,
            tAgA[None, None, None, ik + 1],
            tAsA,
            pred=tApA,
        )
        cute.copy(
            g2s_tiled_copy_b,
            tBgB[None, None, None, ik + 1],
            tBsB,
            pred=tBpB,
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        cute.copy(
            s2r_tiled_copy_a,
            thr_s2r_a.partition_S(sA),
            thr_s2r_a.retile(tCrA),
        )
        cute.copy(
            s2r_tiled_copy_b,
            thr_s2r_b.partition_S(sB),
            thr_s2r_b.retile(tCrB),
        )
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
        cute.arch.sync_threads()

    # ----- Epilogue: R2S -> S2G via smem-staged write, mirroring the
    # ``IsCvtPrecision`` branch of dynamic_mma.cu (lines 254-294).
    #
    #   1. Narrow tCrC (accumulator dtype) to tCrO (out_dtype) via a
    #      register-to-register conversion. When acc_dtype == out_dtype
    #      this is a no-op cast.
    #   2. R2S: copy tCrO -> sO using an MMA-derived TiledCopy whose TV
    #      layout matches the accumulator fragment 1:1, so the per-thread
    #      register data lands in its natural smem slot under the
    #      Swizzle<3,3,3>-shaped sO layout.
    #   3. sync_threads: make the smem visible to all threads in the
    #      block before the S2G phase reads it back with a different
    #      thread-value layout.
    #   4. S2G: copy sO -> gO using a fresh TV layout that gives each
    #      thread a 128-bit contiguous N-tile. Because S2G threads are
    #      packed contiguously along N (val_layout = (1, 16/sizeof(O))),
    #      the predicate collapses to a clean 2-D (CCPY_M, CCPY_N) check
    #      identical to the C++ ``copy_if`` pattern.
    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))

    # R2S: register fragment -> swizzled smem buffer sO.
    thr_r2s_o = r2s_tiled_copy_o.get_slice(tid)
    tOrO_r2s = thr_r2s_o.retile(tCrO)
    tOsO_r2s = thr_r2s_o.partition_D(sO)
    cute.copy(r2s_tiled_copy_o, tOrO_r2s, tOsO_r2s)

    cute.arch.sync_threads()

    # S2G: smem -> gmem with a TV layout that packs threads contiguously
    # along the N dim, matching dynamic_mma.cu's TiledCopyO_S2G. The
    # partitioned tensors have 3-mode shape ``((CPY_INNER, REST_V), CPY_M, CPY_N)``.
    # The S2G's contiguous-N TV layout means each thread's CPY data falls
    # in a single (m, n) iter slot, so the 2-D iter-level predicate is
    # sufficient — no per-element ``rest_v`` axis required. We give the
    # rest_v mode size 1 with stride 0 so the same scalar is replayed
    # across the (trivial) CPY mode at the IR level.
    thr_s2g_o = s2g_tiled_copy_o.get_slice(tid)
    tOsO_s2g = thr_s2g_o.partition_S(sO)
    tOgO_s2g = thr_s2g_o.partition_D(gO)
    tOcO_s2g = thr_s2g_o.partition_S(cC)

    rest_v_size = tOgO_s2g.shape[0][1]
    ccpy_m_size = tOgO_s2g.shape[1]
    ccpy_n_size = tOgO_s2g.shape[2]
    tOpO_s2g = cute.make_rmem_tensor(
        cute.make_layout(
            (rest_v_size, ccpy_m_size, ccpy_n_size),
            stride=(0, 1, ccpy_m_size),
        ),
        cutlass.Boolean,
    )
    for m in cutlass.range_constexpr(ccpy_m_size):
        for n in cutlass.range_constexpr(ccpy_n_size):
            tOpO_s2g[0, m, n] = cute.elem_less(
                tOcO_s2g[(0, 0), m, n][0],
                m_max,
            ) and cute.elem_less(
                tOcO_s2g[(0, 0), m, n][1],
                n_max,
            )
    cute.copy(s2g_tiled_copy_o, tOsO_s2g, tOgO_s2g, pred=tOpO_s2g)


@cute.jit
def dynamic_mma_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    stream: CUstream,
    acc_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    # ----- Tiled MMA -----
    op = cute.nvgpu.warp.MmaF16BF16Op(
        mA.element_type,
        acc_dtype,
        MMA_INST_MNK,
    )
    tm = cute.make_tiled_mma(
        op,
        atom_layout_mnk=ATOM_LAYOUT_MNK,
        permutation_mnk=MMA_TILE_MNK,
    )

    # ----- Swizzled smem layouts (matches Swizzle<3,3,3>) -----
    swz = cute.make_swizzle(3, 3, 3)
    inner_AB = min(64, BLK_K)
    atom_AB = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_AB), stride=(inner_AB, 1)),
    )
    sA_layout = cute.tile_to_shape(atom_AB, (BLK_M, BLK_K), order=(0, 1))
    sB_layout = cute.tile_to_shape(atom_AB, (BLK_N, BLK_K), order=(0, 1))
    # sC mirrors dynamic_mma.cu's SmemLayoutC: Swizzle<3,3,3> o (8 x min(64, BLK_N)).
    inner_C = min(64, BLK_N)
    atom_C = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_C), stride=(inner_C, 1)),
    )
    sC_layout = cute.tile_to_shape(atom_C, (BLK_M, BLK_N), order=(0, 1))
    # sO mirrors dynamic_mma.cu's SmemLayoutO: Swizzle<3,3,3> o (8 x min(64, BLK_N)).
    inner_O = min(64, BLK_N)
    atom_O = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_O), stride=(inner_O, 1)),
    )
    sO_layout = cute.tile_to_shape(atom_O, (BLK_M, BLK_N), order=(0, 1))

    # ----- G2S copies (cp.async) -----
    g2s_op = cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
    )
    # AB copy is along (BLK_M / BLK_N, BLK_K). 128-bit cp.async loads
    # 16B/thread = 16 / sizeof(elt) elements along K.
    elt_bytes_ab = mA.element_type.width // 8
    k_chunk = min(64, BLK_K) // (16 // elt_bytes_ab)
    tlAB_thr = cute.make_layout(
        (NUM_THREADS // k_chunk, k_chunk),
        stride=(k_chunk, 1),
    )
    tlAB_val = cute.make_layout((1, 16 // elt_bytes_ab))
    g2s_atom_a = cute.make_copy_atom(g2s_op, mA.element_type, num_bits_per_copy=128)
    g2s_atom_b = cute.make_copy_atom(g2s_op, mB.element_type, num_bits_per_copy=128)
    g2s_tiled_copy_a = cute.make_tiled_copy_tv(g2s_atom_a, tlAB_thr, tlAB_val)
    g2s_tiled_copy_b = cute.make_tiled_copy_tv(g2s_atom_b, tlAB_thr, tlAB_val)

    # C copy is along (BLK_M, BLK_N). 16B/thread = 16 / sizeof(C_elt) along N.
    elt_bytes_c = mC.element_type.width // 8
    n_chunk = min(64, BLK_N) // (16 // elt_bytes_c)
    tlC_thr = cute.make_layout(
        (NUM_THREADS // n_chunk, n_chunk),
        stride=(n_chunk, 1),
    )
    tlC_val = cute.make_layout((1, 16 // elt_bytes_c))
    g2s_atom_c = cute.make_copy_atom(g2s_op, mC.element_type, num_bits_per_copy=128)
    g2s_tiled_copy_c = cute.make_tiled_copy_tv(g2s_atom_c, tlC_thr, tlC_val)

    universal = cute.nvgpu.CopyUniversalOp()

    # ----- S2R copies (ldmatrix for 16-bit operands) -----
    # A/B own 4 32-bit packets per thread (VAL_EXPAND_K=2), so the x4 ldmatrix
    # variant fits exactly. C has no K val-expand, so each thread only owns
    # 2 32-bit packets — use the x2 ldmatrix variant when C is 16-bit. For
    # wider C (e.g. fp32) fall back to a universal copy.
    ldm_op_ab = cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4)
    ldm_op_c = cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2)
    s2r_atom_a = cute.make_copy_atom(ldm_op_ab, mA.element_type)
    s2r_atom_b = cute.make_copy_atom(ldm_op_ab, mB.element_type)
    s2r_tiled_copy_a = cute.make_tiled_copy_A(s2r_atom_a, tm)
    s2r_tiled_copy_b = cute.make_tiled_copy_B(s2r_atom_b, tm)
    if cutlass.const_expr(mC.element_type.width == 16):
        s2r_atom_c = cute.make_copy_atom(ldm_op_c, mC.element_type)
    else:
        s2r_atom_c = cute.make_copy_atom(universal, mC.element_type)
    s2r_tiled_copy_c = cute.make_tiled_copy_C(s2r_atom_c, tm)

    # ----- R2S + S2G copies for the smem-staged epilogue -----
    # R2S: MMA-derived TiledCopy so the per-thread register fragment lands
    # at its natural smem position under the swizzled sO layout. Pick the
    # op the same way dynamic_mma.cu does: SM90+ with a 16-bit output uses
    # ``stmatrix`` (x2 because the C/O fragment has no K val-expand and so
    # owns 2 32-bit packets per thread); otherwise fall back to a universal
    # STS lowering.
    sm_major, _ = torch.cuda.get_device_capability()
    if cutlass.const_expr(sm_major >= 9 and mO.element_type.width == 16):
        stm_op = cute.nvgpu.warp.StMatrix8x8x16bOp(False, 2)
        r2s_atom_o = cute.make_copy_atom(stm_op, mO.element_type)
    else:
        r2s_atom_o = cute.make_copy_atom(universal, mO.element_type)
    r2s_tiled_copy_o = cute.make_tiled_copy_C(r2s_atom_o, tm)

    # S2G: explicit TV layout matching dynamic_mma.cu's TiledCopyO_S2G —
    # threads packed contiguously along N, each thread copies
    # ``16 / sizeof(out)`` elements in a single 128-bit transaction.
    elt_bytes_o = mO.element_type.width // 8
    n_chunk_o = min(64, BLK_N) // (16 // elt_bytes_o)
    tlO_thr = cute.make_layout(
        (NUM_THREADS // n_chunk_o, n_chunk_o),
        stride=(n_chunk_o, 1),
    )
    tlO_val = cute.make_layout((1, 16 // elt_bytes_o))
    s2g_atom_o = cute.make_copy_atom(universal, mO.element_type, num_bits_per_copy=128)
    s2g_tiled_copy_o = cute.make_tiled_copy_tv(s2g_atom_o, tlO_thr, tlO_val)

    # ----- Launch -----
    M, _ = mA.shape
    N, _ = mB.shape
    grid_n = (N + BLK_N - 1) // BLK_N
    grid_m = (M + BLK_M - 1) // BLK_M

    dynamic_mma_kernel(
        mA,
        mB,
        mC,
        mO,
        tm,
        g2s_tiled_copy_a,
        g2s_tiled_copy_b,
        g2s_tiled_copy_c,
        s2r_tiled_copy_a,
        s2r_tiled_copy_b,
        s2r_tiled_copy_c,
        r2s_tiled_copy_o,
        s2g_tiled_copy_o,
        sA_layout,
        sB_layout,
        sC_layout,
        sO_layout,
        out_dtype,
        is_gemm,
    ).launch(
        grid=(grid_n, grid_m, 1),
        block=(NUM_THREADS, 1, 1),
        stream=stream,
        # nvvm.minctasm hint: require >= 2 resident blocks per SM. Left
        # unconstrained the CuTeDSL register allocator is ILP-greedy and
        # inflates to ~150 registers/thread (not a spill -- just a loose
        # allocation), which caps occupancy at one block per SM on
        # H100/H200. The hint tightens the budget so the kernel fits at
        # 128 registers/thread with zero spill, roughly doubling achieved
        # occupancy. 2 is the largest spill-free value here -- 3+ would
        # force register spills to local memory.
        min_blocks_per_mp=2,
    )


# -----------------------------------------------------------------------------
# Host-side test harness (mirrors dynamic_mma.py)
# -----------------------------------------------------------------------------


PRINT_LENGTH = 100


def relative_error(target: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    diff = target - ref
    norm_diff = torch.norm(diff, p=2)
    norm_diff_ref = torch.norm(ref, p=2)
    return (norm_diff / (norm_diff_ref + eps)).item()


def compare_matrix(
    kernel_output: torch.Tensor,
    torch_output: torch.Tensor,
    counters: dict,
) -> None:
    kernel_output = kernel_output.float()
    torch_output = torch_output.float()
    max_diff = torch.max(torch.abs(torch_output - kernel_output))
    mean_diff = torch.mean(torch.abs(torch_output - kernel_output))
    re = relative_error(kernel_output, torch_output)
    is_correct = re < 0.001

    if not is_correct:
        counters["failed"] += 1
        print(f" Kernel Output: {tuple(kernel_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(kernel_output[:8, :8])
        print(f" Torch Output: {tuple(torch_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(torch_output[:8, :8])
    else:
        counters["succeed"] += 1
    status = "Success" if is_correct else "Failed"
    print(
        f" Result: {status}, Max diff = {max_diff:.5f}, Mean diff = {mean_diff:.5f}, RE = {(re * 100):.2f}% ".center(
            PRINT_LENGTH, "-"
        )
    )


def make_cute_tensor(t: torch.Tensor) -> cute.Tensor:
    divisibility = max(1, 16 // t.element_size())
    return (
        from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=divisibility)
    )


def _compile_pair(a_template, b_template, c_template, o_template, acc_dtype, out_dtype):
    """Pre-compile (is_gemm=True, False) specializations for the given dtype combo."""
    g_clear = cute.compile(
        dynamic_mma_gemm,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(o_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        True,
        options="--enable-tvm-ffi",
    )
    g_accum = cute.compile(
        dynamic_mma_gemm,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(o_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        False,
        options="--enable-tvm-ffi",
    )
    return g_clear, g_accum


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    # N and K must be multiples of 8 to keep the 128-bit cp.async vector copy
    # legal along the contiguous inner dim. Sweep matches dynamic_mma.py.
    Ms = [16, 64, 128, 192, 256, 1024, 4096]
    Ns = [16, 64, 128, 192, 256, 1024, 4096]
    Ks = [16, 64, 128, 192, 256, 1024, 4096]
    exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    counters = {"succeed": 0, "failed": 0}
    torch.cuda.manual_seed_all(9527)

    # Template tensors for compile-time tagging (shape doesn't bind the
    # compiled artifact — only dtype + leading-dim alignment do, which is
    # why the same compiled callable handles every (M, N, K) in the sweep).
    M0, N0, K0 = 128, 128, 64

    # ----- Spec 1: fp16 in, fp16 acc, fp16 out -----
    print(" Compiling fp16 in / fp16 acc / fp16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M0, K0, device="cuda", dtype=torch.float16)
    b_t = torch.empty(N0, K0, device="cuda", dtype=torch.float16)
    c_t = torch.empty(M0, N0, device="cuda", dtype=torch.float16)
    o_t = torch.empty(M0, N0, device="cuda", dtype=torch.float16)
    fp16_clear, fp16_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        o_t,
        cutlass.Float16,
        cutlass.Float16,
    )

    # ----- Spec 2: fp16 in, fp32 acc, fp16 out -----
    print(" Compiling fp16 in / fp32 acc / fp16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M0, K0, device="cuda", dtype=torch.float16)
    b_t = torch.empty(N0, K0, device="cuda", dtype=torch.float16)
    c_t = torch.empty(M0, N0, device="cuda", dtype=torch.float32)
    o_t = torch.empty(M0, N0, device="cuda", dtype=torch.float16)
    fp16f32_clear, fp16f32_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        o_t,
        cutlass.Float32,
        cutlass.Float16,
    )

    # ----- Spec 3: bf16 in, fp32 acc, bf16 out -----
    print(" Compiling bf16 in / fp32 acc / bf16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M0, K0, device="cuda", dtype=torch.bfloat16)
    b_t = torch.empty(N0, K0, device="cuda", dtype=torch.bfloat16)
    c_t = torch.empty(M0, N0, device="cuda", dtype=torch.float32)
    o_t = torch.empty(M0, N0, device="cuda", dtype=torch.bfloat16)
    bf16_clear, bf16_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        o_t,
        cutlass.Float32,
        cutlass.BFloat16,
    )

    # ----- Sweep: fp16 = fp16 * fp16 + fp16 (Spec 1, exercise only) -----
    # Matches dynamic_mma.py: the fp16-accumulator variant is launched but
    # NOT validated against torch — fp16 accumulation drops too many bits
    # for large K to match torch's fp32-accumulated reference.
    print(" fp16 = fp16 * fp16 + fp16 (exercise only) ".center(PRINT_LENGTH, "="))
    torch.cuda.manual_seed_all(9527)
    for m, n, k in exps:
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(n, k, device="cuda", dtype=torch.float16)
        c = torch.randn(m, n, device="cuda", dtype=torch.float16)

        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16_clear(a, b, c.clone(), out)
        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16_accum(a, b, c.clone(), out)
        torch.cuda.synchronize()

    # ----- Sweep: fp16 in, fp32 acc, fp16 out (Spec 2) -----
    print(" fp16 = fp32_acc(fp16 * fp16) + fp32 ".center(PRINT_LENGTH, "="))
    torch.cuda.manual_seed_all(9527)
    for m, n, k in exps:
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(n, k, device="cuda", dtype=torch.float16)
        c = torch.randn(m, n, device="cuda", dtype=torch.float32)

        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16f32_clear(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(out, torch.matmul(a, b.T), counters)

        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16f32_accum(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(out, torch.addmm(c, a.float(), b.T.float()).half(), counters)

    # ----- Sweep: bf16 in, fp32 acc, bf16 out (Spec 3) -----
    print(" bf16 = fp32_acc(bf16 * bf16) + fp32 ".center(PRINT_LENGTH, "="))
    torch.cuda.manual_seed_all(9527)
    for m, n, k in exps:
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(m, n, device="cuda", dtype=torch.float32)

        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        bf16_clear(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(out, torch.matmul(a.float(), b.T.float()).bfloat16(), counters)

        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        bf16_accum(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(
            out,
            torch.addmm(c, a.float(), b.T.float()).bfloat16(),
            counters,
        )

    print(f" Summary: {counters['succeed']} Succeed, {counters['failed']} Failed ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
