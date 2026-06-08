"""Multi-stage cp.async pipelined GEMM in CuTe DSL.

A near-verbatim copy of ``cutedsl_pipelining.py`` plus an in-kernel timeline
profiler (see ``cutedsl_sm_profiler.py``): the only kernel changes are
``%globaltimer`` stamps at the phase boundaries, gated by a ``profile_enabled``
constexpr so the kernel is unchanged when profiling is off.

CuTe DSL counterpart of ``pipelining.cu`` / ``pipelining.py``. Extends
``08-dynamic-mma`` with two pipelines stacked on top:

  1. **G2S (smem) pipeline** with ``NUM_STAGES`` ring slots (default 3).
     Stages ``0 .. NUM_STAGES-2`` are prefetched in the prologue. The
     mainloop's last k-block triggers ``cp_async_wait_group(NUM_STAGES-2)``
     so at most one outstanding cp.async group is in flight while the
     register pipeline consumes the freshly arrived stage.
  2. **S2R (register) pipeline**, gated by the ``REG_STAGES`` constant.
     With ``REG_STAGES >= 2`` (default) the smem -> rmem load for
     k_block + 1 is hoisted ahead of the MMA for k_block, so the LDS /
     LDSM latency is hidden behind the tensor-core math. Setting
     ``REG_STAGES == 1`` disables the register pipeline (``pipelining_no_
     reg_prefetch.cu``): each k-tile loads its whole A/B fragment, then
     runs every MMA, leaving only the G2S pipeline to overlap work.

The prologue mirrors ``pipelining.cu`` lines 156-214:
  * predicated K-residue copy for stage 0 (shifted by ``k_residue``);
  * full-tile copies for stages ``1 .. NUM_STAGES-2`` with an overshoot
    guard that clears the M/N predicates when ``ik == k_tile_count``;
  * ``cp_async_wait_group(NUM_STAGES-2)`` and a single ``ldmatrix`` of
    k_block=0 from smem stage 0 before entering the mainloop.

The mainloop walks all ``num_k_block`` MMA k-iterations per k-tile,
prefetches the next rmem at the top, fires the next stage's cp.async
on ``k_block == 0``, then issues the tensor-core gemm. Three counters
(``smem_pipe_read``, ``smem_pipe_write``, ``k_tile_index``) track the
ring-buffer state.

Three dtype specs are exercised, matching ``pipelining.py``:

  * fp16 in, fp16 acc, fp16 out  (exercise only)
  * fp16 in, fp32 acc, fp16 out  (validated)
  * bf16 in, fp32 acc, bf16 out  (validated)

Run with ``python cutedsl_pipelining.py``.

Smem footprint / occupancy:
  * Only the multi-stage A/B pipeline gets real smem (``A_pipe + B_pipe``;
    96KB for the default 128x128x64 tile at 3 stages). The C addend and the
    O store buffer are never live during the mainloop, so the epilogue
    aliases them over the (drained) A/B smem, mirroring 13-/14-*. This keeps
    the block under the ~113KB needed for two resident CTAs per SM.
  * To realise that second block the launch sets ``min_blocks_per_mp=2``
    (mirrors 08-dynamic-mma): without it CuTeDSL's allocator runs
    ILP-greedy to ~160 reg/thread and caps occupancy at one block/SM. The
    hint tightens the budget to 128 reg/thread, letting two blocks
    co-reside (~24% achieved occupancy vs ~12.5%). At that cap the
    no-prefetch path (``REG_STAGES == 1``) fits spill-free, while the
    register-prefetch path (``REG_STAGES >= 2``) keeps an extra k_block
    fragment live and spills lightly to (L1-resident) local memory — which
    is why the no-prefetch path is the faster configuration on H200 here
    (448us vs 527us at M=N=K=4096, bf16).
  * C is folded into the accumulator in the epilogue (acc starts at zero,
    acc += C after the matmul) rather than preloaded, so sC need not coexist
    with the pipeline. The epilogue then narrows to out_dtype and drains
    R2S -> S2G via a swizzled sO (Swizzle<3,3,3> o (8 x min(64, BLK_N))),
    with a 2-D (CCPY_M, CCPY_N) S2G predicate for M/N residue handling.
"""

from pathlib import Path

import cutedsl_sm_profiler as smprof
import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream


# Block tile (matches pipelining.cu's default KernelSpec defaults)
BLK_M = 128
BLK_N = 128
BLK_K = 64
# G2S (smem) ring-buffer depth.
NUM_STAGES = 3
# Register-pipeline (S2R) depth, mirroring ``prefetch_s2r_tiles`` in the C++:
#   * ``REG_STAGES >= 2`` double-buffers the smem -> rmem load, hoisting the
#     k_block + 1 fetch ahead of the k_block MMA (the default fast path).
#   * ``REG_STAGES == 1`` disables register prefetch entirely: each k-tile
#     loads its whole A/B fragment, then runs every MMA — only the
#     GMEM -> SMEM -> RF copy pipeline overlaps work. Mirrors
#     ``pipelining_no_reg_prefetch.cu``.
REG_STAGES = 1

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

# Profiler event types for THIS kernel (the profiler itself is generic and knows
# nothing about these -- they are registered with names on the host side). The
# four phases are recorded once each; compute/wait are recorded once per
# mainloop k-tile, so a warp emits 4 + 2*k_tile_count events.
PROF_TOTAL = 0
PROF_PROLOGUE = 1
PROF_MAINLOOP = 2
PROF_EPILOGUE = 3
PROF_COMPUTE = 4
PROF_WAIT = 5
PROF_EVENT_NAMES = {
    PROF_TOTAL: "total",
    PROF_PROLOGUE: "prologue",
    PROF_MAINLOOP: "mainloop",
    PROF_EPILOGUE: "epilogue",
    PROF_COMPUTE: "compute",
    PROF_WAIT: "wait",
}


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def pipelining_kernel(
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
    # Profiling is opt-in via this single bundled handle. A disabled ``prof``
    # (its buffer is None) lowers to no device argument and folds every
    # profiler call to nothing, so the kernel is then identical to
    # ``cutedsl_pipelining.py``.
    prof: smprof.ProfilerArgs = smprof.ProfilerArgs(None, False),
):
    tid, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    M, K = mA.shape
    N, _ = mB.shape

    # Open a profiling session once (resolves this warp's ids + buffer sizes a
    # single time, from the grid dims and the buffer header); call sites then
    # just name an event type.
    prof_ctx = prof.begin()
    prof_ctx.event_start(PROF_TOTAL)
    prof_ctx.event_start(PROF_PROLOGUE)

    # 3-D block tiler — gA / gB carry a K-tile iteration mode.
    tiler = (BLK_M, BLK_N, BLK_K)
    gA = cute.local_tile(mA, tiler=tiler, coord=(bidy, bidx, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=tiler, coord=(bidy, bidx, None), proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))
    gO = cute.local_tile(mO, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))

    # Per-block residue clamps.
    m_max = M - BLK_M * bidy
    n_max = N - BLK_N * bidx
    k_tile_count = cute.size(gA, mode=[2])
    # k_residue <= 0: how far the first K-tile's origin sits before the
    # true K=0. Shifting gA/gB along K by k_residue makes every K-tile
    # after the first trivially in-bounds.
    k_residue = K - BLK_K * k_tile_count
    gA = cute.domain_offset((0, k_residue, 0), gA)
    gB = cute.domain_offset((0, k_residue, 0), gB)

    # ----- Smem allocation -----
    # Only the multi-stage A/B pipeline gets real storage. sC (accumulate
    # addend) and sO (epilogue store buffer) are never live during the
    # mainloop, so the epilogue aliases them over this same A/B smem (see
    # below). Footprint stays at sA + sB so two CTAs fit per SM.
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, byte_alignment=16)

    # ----- G2S partitions (predicated) -----
    thr_g2s_a = g2s_tiled_copy_a.get_slice(tid)
    tAgA = thr_g2s_a.partition_S(gA)  # (CPY, CPY_M, CPY_K, k_tiles)
    tAsA = thr_g2s_a.partition_D(sA)  # (CPY, CPY_M, CPY_K, PIPE)

    thr_g2s_b = g2s_tiled_copy_b.get_slice(tid)
    tBgB = thr_g2s_b.partition_S(gB)  # (CPY, CPY_N, CPY_K, k_tiles)
    tBsB = thr_g2s_b.partition_D(sB)  # (CPY, CPY_N, CPY_K, PIPE)

    # C is loaded in the epilogue (folded into the accumulator there), so its
    # smem partition is built then; only the gmem-side partition is needed up
    # front for the identity-coord predicate setup.
    thr_g2s_c = g2s_tiled_copy_c.get_slice(tid)
    tCgC = thr_g2s_c.partition_S(gC)  # (CPY, CPY_M, CPY_N)

    # ----- Identity tensors for predication -----
    cA = cute.make_identity_tensor((BLK_M, BLK_K))
    cB = cute.make_identity_tensor((BLK_N, BLK_K))
    cC = cute.make_identity_tensor((BLK_M, BLK_N))
    tAcA = thr_g2s_a.partition_S(cA)
    tBcB = thr_g2s_b.partition_S(cB)
    tCcC = thr_g2s_c.partition_S(cC)

    # ----- M / N predicate (used for prologue stages 1.. and the mainloop) -----
    # Three-mode layout (rest_v, CPY_M, CPY_K) with the K mode broadcast at
    # stride 0 so the same M/N predicate is replayed across every K iter.
    # K-bound is not needed here because the domain_offset shift guarantees
    # every K-tile from stage 1 onward is in-bounds; only stage 0 (handled
    # by ``tApA_first`` below) needs the per-element K check.
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

    # ----- Fragments -----
    thr_mma = tiled_mma.get_slice(tid)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA)[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB)[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gC))

    thr_s2r_a = s2r_tiled_copy_a.get_slice(tid)
    thr_s2r_b = s2r_tiled_copy_b.get_slice(tid)
    thr_s2r_c = s2r_tiled_copy_c.get_slice(tid)

    tAsA_s2r = thr_s2r_a.partition_S(sA)  # (CPY, CPY_M, CPY_K, PIPE)
    tBsB_s2r = thr_s2r_b.partition_S(sB)  # (CPY, CPY_N, CPY_K, PIPE)
    tArA_s2r = thr_s2r_a.retile(tCrA)  # (CPY, CPY_M, CPY_K)
    tBrB_s2r = thr_s2r_b.retile(tCrB)  # (CPY, CPY_N, CPY_K)

    # =========================================================================
    # G2S prologue: prefetch NUM_STAGES-1 stages.
    # =========================================================================

    # Stage 0 (residue tile) needs a per-element predicate combining M-bound
    # AND K-bound. With CPY_K = 1 a single iter carries lanes spread across
    # the whole BLK_K range — some at valid in-bounds K-positions, others
    # not after the domain_offset shift — so the gate must fire per lane /
    # per val element rather than at the C++ iteration level.
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

    # Pre-zero stage 0 so predicate-off slots read as 0. Each thread fills its
    # own g2s partition slots and the following cp.async writes that same
    # partition, so program order suffices — the post-wait_group sync_threads
    # below publishes both the zeros and the loads to the s2r readers.
    tAsA[None, None, None, 0].fill(0)
    tBsB[None, None, None, 0].fill(0)
    cute.copy(
        g2s_tiled_copy_a,
        tAgA[None, None, None, 0],
        tAsA[None, None, None, 0],
        pred=tApA_first,
    )
    cute.copy(
        g2s_tiled_copy_b,
        tBgB[None, None, None, 0],
        tBsB[None, None, None, 0],
        pred=tBpB_first,
    )
    cute.arch.cp_async_commit_group()

    # Stages 1 .. NUM_STAGES-2: full-tile copies, with overshoot guard.
    k_tile_index = cutlass.Int32(1)
    for ik in cutlass.range_constexpr(1, NUM_STAGES - 1):
        # Once we'd read past the end of K, mask everything off.
        if k_tile_index >= k_tile_count:
            tApA.fill(False)
            tBpB.fill(False)
        cute.copy(
            g2s_tiled_copy_a,
            tAgA[None, None, None, k_tile_index],
            tAsA[None, None, None, ik],
            pred=tApA,
        )
        cute.copy(
            g2s_tiled_copy_b,
            tBgB[None, None, None, k_tile_index],
            tBsB[None, None, None, ik],
            pred=tBpB,
        )
        cute.arch.cp_async_commit_group()
        k_tile_index = k_tile_index + 1

    # =========================================================================
    # Wait for the prologue's stage-0 cp.async, then prime the accumulator.
    # =========================================================================
    cute.arch.cp_async_wait_group(NUM_STAGES - 2)
    cute.arch.sync_threads()
    # Prologue prefetch done; close the prologue range and open the mainloop in
    # one stamp (the two slices abut).
    prof_ctx.event_switch(PROF_PROLOGUE, PROF_MAINLOOP)

    num_k_block = cute.size(tCrA, mode=[2])
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(NUM_STAGES - 1)

    # ----- Initialise accumulator to zero -----
    # The C addend (accumulate path) is folded in during the epilogue, not
    # preloaded here, so sC stays out of the mainloop and can alias the A/B
    # smem. Both mainloops therefore start from a zeroed accumulator.
    tCrC.fill(0.0)

    if cutlass.const_expr(REG_STAGES >= 2):
        # =====================================================================
        # Register-prefetch (S2R) pipeline. The smem -> rmem load for
        # k_block + 1 is hoisted ahead of the MMA for k_block so the LDS /
        # LDSM latency hides behind the tensor-core math. Double-buffered at
        # the register level. The outer loop's iteration variable is unused —
        # gmem progress is tracked by ``k_tile_index`` (the cp.async read
        # pointer, NUM_STAGES-1 ahead of the smem read pointer because the
        # prologue prefetched those stages) and smem ring positions by
        # ``smem_pipe_{read,write}``. Mirrors ``pipelining.cu``.
        # =====================================================================
        # Prefetch first k_block from stage 0 into registers.
        cute.copy(
            s2r_tiled_copy_a,
            tAsA_s2r[None, None, 0, smem_pipe_read],
            tArA_s2r[None, None, 0],
        )
        cute.copy(
            s2r_tiled_copy_b,
            tBsB_s2r[None, None, 0, smem_pipe_read],
            tBrB_s2r[None, None, 0],
        )

        for _ in cutlass.range(k_tile_count, unroll_full=False):
            for k_block in cutlass.range_constexpr(num_k_block):
                # When we're on the last k_block of this tile, wait for the
                # next smem stage to arrive and bump smem_pipe_read.
                if k_block == num_k_block - 1:
                    cute.arch.cp_async_wait_group(NUM_STAGES - 2)
                    cute.arch.sync_threads()
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == NUM_STAGES:
                        smem_pipe_read = cutlass.Int32(0)

                # Prefetch next k_block from smem to rmem. After the bump
                # above (which only fires on the last k_block of the tile),
                # smem_pipe_read already points at the freshly-synced stage,
                # so the wrap to k_block=0 reads from the new stage.
                k_block_next = (k_block + 1) % num_k_block
                cute.copy(
                    s2r_tiled_copy_a,
                    tAsA_s2r[None, None, k_block_next, smem_pipe_read],
                    tArA_s2r[None, None, k_block_next],
                )
                cute.copy(
                    s2r_tiled_copy_b,
                    tBsB_s2r[None, None, k_block_next, smem_pipe_read],
                    tBrB_s2r[None, None, k_block_next],
                )

                # On the first k_block of the tile, fire cp.async for the
                # next smem stage (writes ahead of the current read pointer).
                if k_block == 0:
                    # Overshoot guard: when k_tile_index has stepped past
                    # the end of K, mask everything off so we don't issue
                    # OOB cp.async.
                    if k_tile_index >= k_tile_count:
                        tApA.fill(False)
                        tBpB.fill(False)
                    cute.copy(
                        g2s_tiled_copy_a,
                        tAgA[None, None, None, k_tile_index],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )
                    cute.copy(
                        g2s_tiled_copy_b,
                        tBgB[None, None, None, k_tile_index],
                        tBsB[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    k_tile_index = k_tile_index + 1
                    smem_pipe_write = smem_pipe_write + 1
                    if smem_pipe_write == NUM_STAGES:
                        smem_pipe_write = cutlass.Int32(0)

                # Tensor-core gemm for this k_block.
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )
    else:
        # =====================================================================
        # No register prefetch: only the GMEM -> SMEM -> RF copy pipeline
        # overlaps work. Each k-tile loads its whole A/B fragment from smem in
        # one copy, fires the next smem stage's cp.async, then runs the MMA
        # over every k_block. Because all S2R loads finish before the first
        # MMA, the smem -> rmem latency is *not* hidden behind the math — this
        # is the baseline the register pipeline above improves on. Mirrors
        # ``pipelining_no_reg_prefetch.cu``'s mainloop.
        # =====================================================================
        for _ in cutlass.range(k_tile_count, unroll_full=False):
            # Per-tile "compute" event opens here (S2R load + MMAs). The append
            # counter gives it the next event_id, so compute events land in
            # k-tile order; folds to nothing when profiling is off.
            prof_ctx.event_start(PROF_COMPUTE)

            # Whole-tile smem -> rmem load (every k_block at once).
            cute.copy(
                s2r_tiled_copy_a,
                tAsA_s2r[None, None, None, smem_pipe_read],
                tArA_s2r,
            )
            cute.copy(
                s2r_tiled_copy_b,
                tBsB_s2r[None, None, None, smem_pipe_read],
                tBrB_s2r,
            )

            # Fire cp.async for the next smem stage (writes ahead of read ptr).
            if k_tile_index >= k_tile_count:
                tApA.fill(False)
                tBpB.fill(False)
            cute.copy(
                g2s_tiled_copy_a,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, smem_pipe_write],
                pred=tApA,
            )
            cute.copy(
                g2s_tiled_copy_b,
                tBgB[None, None, None, k_tile_index],
                tBsB[None, None, None, smem_pipe_write],
                pred=tBpB,
            )
            cute.arch.cp_async_commit_group()
            k_tile_index = k_tile_index + 1
            smem_pipe_write = smem_pipe_write + 1
            if smem_pipe_write == NUM_STAGES:
                smem_pipe_write = cutlass.Int32(0)

            # MMA over every k_block of the freshly-loaded tile.
            for k_block in cutlass.range_constexpr(num_k_block):
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

            # Close "compute" and open "wait" with a single stamp (after this
            # tile's MMAs, before the cp.async wait). This boundary is the one
            # mark not anchored by a __syncthreads; the compiler barrier inside
            # the timestamp read keeps it from drifting. Fusing the two halves the
            # boundary's timer reads and makes the slices abut exactly.
            prof_ctx.event_switch(PROF_COMPUTE, PROF_WAIT)

            # Wait for the next stage to land, advance the smem read pointer.
            cute.arch.cp_async_wait_group(NUM_STAGES - 2)
            cute.arch.sync_threads()
            # Close "wait" (barrier-anchored by the sync above).
            prof_ctx.event_end(PROF_WAIT)
            smem_pipe_read = smem_pipe_read + 1
            if smem_pipe_read == NUM_STAGES:
                smem_pipe_read = cutlass.Int32(0)

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    # Mainloop drained; close the mainloop range and open the epilogue (one stamp).
    prof_ctx.event_switch(PROF_MAINLOOP, PROF_EPILOGUE)

    # =========================================================================
    # Epilogue. C and O reuse the now-drained A/B smem: the mainloop's final
    # cp_async_wait_group(0) + sync above guarantees every thread is done
    # reading sA / sB, so the region is free to overwrite.
    # =========================================================================

    # ----- Fold the C addend into the accumulator (accumulate path only) -----
    if cutlass.const_expr(not is_gemm):
        # sC (compute-C dtype) aliases sA's storage. An fp32 BLK_M x BLK_N
        # tile spans sA plus part of the contiguous sB, both dead here. The
        # swizzle must stay in the make_tensor layout (where partition_D
        # composes it correctly); recast_ptr only adjusts the element dtype
        # (and strips to a bare pointer). Splitting the swizzle into
        # recast_ptr instead mis-maps partition_D under Swizzle<3,3,3> on
        # SM80-class ldmatrix/cp.async copies — that form is for SM90 TMA.
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=mC.element_type),
            sC_layout,
        )
        tCsC = thr_g2s_c.partition_D(sC)

        # Coalesced gmem -> smem load of C with the M / N residue predicate
        # (C has no K mode, so no K bound). Predicate-off slots stay zero and
        # add nothing.
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
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # smem -> rmem into a C-shaped fragment, then acc += C (in acc dtype).
        tCrC_add = cute.make_fragment_like(tCrC, mC.element_type)
        cute.copy(
            s2r_tiled_copy_c,
            thr_s2r_c.partition_S(sC),
            thr_s2r_c.retile(tCrC_add),
        )
        tCrC.store(tCrC.load() + tCrC_add.load().to(tCrC.element_type))

        # All threads must finish reading sC before sO overwrites the region.
        cute.arch.sync_threads()

    # ----- Narrow to out_dtype, R2S -> S2G via a swizzled smem buffer -----
    # sO (out dtype, which equals A's dtype for every spec) aliases sA's
    # storage via a plain full-ComposedLayout make_tensor. The split
    # .outer / .inner form mis-maps swizzled upper-half columns in the r2s
    # partition_D store, so the plain form is mandatory here.
    sO = cute.make_tensor(sA.iterator, sO_layout)

    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))

    # R2S: register fragment -> swizzled smem buffer sO.
    thr_r2s_o = r2s_tiled_copy_o.get_slice(tid)
    tOrO_r2s = thr_r2s_o.retile(tCrO)
    tOsO_r2s = thr_r2s_o.partition_D(sO[None, None, 0])
    cute.copy(r2s_tiled_copy_o, tOrO_r2s, tOsO_r2s)

    cute.arch.sync_threads()

    # S2G: smem -> gmem with a TV layout that packs threads contiguously
    # along the N dim, matching pipelining.cu's TiledCopyO_S2G. The
    # partitioned tensors are 3-mode ``((CPY_INNER, REST_V), CPY_M, CPY_N)``;
    # the pred uses a stride-0 rest_v slot so the same iter-level scalar
    # is replayed across the (trivial) CPY mode.
    thr_s2g_o = s2g_tiled_copy_o.get_slice(tid)
    tOsO_s2g = thr_s2g_o.partition_S(sO[None, None, 0])
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

    # Close the epilogue and total ranges. The compiler barrier inside event_end
    # already keeps the final clock read from being hoisted above the S2G store.
    prof_ctx.event_end(PROF_EPILOGUE)
    prof_ctx.event_end(PROF_TOTAL)


@cute.jit
def pipelining_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    stream: CUstream,
    acc_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
    prof_buffer=None,
    profile_enabled: cutlass.Constexpr[bool] = False,
    profile_show_overhead: cutlass.Constexpr[bool] = False,
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

    # ----- Swizzled smem layouts (Swizzle<3,3,3>) ; A/B carry a PIPE mode -----
    swz = cute.make_swizzle(3, 3, 3)
    inner_AB = min(64, BLK_K)
    atom_AB = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_AB), stride=(inner_AB, 1)),
    )
    sA_layout = cute.tile_to_shape(
        atom_AB,
        (BLK_M, BLK_K, NUM_STAGES),
        order=(0, 1, 2),
    )
    sB_layout = cute.tile_to_shape(
        atom_AB,
        (BLK_N, BLK_K, NUM_STAGES),
        order=(0, 1, 2),
    )
    # sC mirrors pipelining.cu's SmemLayoutC: Swizzle<3,3,3> o (8 x min(64, BLK_N)).
    inner_C = min(64, BLK_N)
    atom_C = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_C), stride=(inner_C, 1)),
    )
    sC_layout = cute.tile_to_shape(atom_C, (BLK_M, BLK_N), order=(0, 1))
    # sO mirrors pipelining.cu's SmemLayoutO: Swizzle<3,3,3> o (8 x min(64, BLK_N)).
    # Single-stage — the epilogue runs after the mainloop finishes draining,
    # so no PIPE mode is needed.
    inner_O = min(64, BLK_N)
    atom_O = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_O), stride=(inner_O, 1)),
    )
    sO_layout = cute.tile_to_shape(atom_O, (BLK_M, BLK_N, 1), order=(0, 1, 2))

    # ----- G2S copies (cp.async) -----
    g2s_op = cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
    )
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
    # R2S: MMA-derived TiledCopy so each thread's accumulator fragment lands
    # at its natural smem position under the swizzled sO layout. Pick the op
    # the same way pipelining.cu does: SM90+ with a 16-bit output uses
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

    # S2G: explicit TV layout — threads packed contiguously along N,
    # each thread emits 16 / sizeof(out) elements in a 128-bit store.
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

    # Bundle into the single handle the kernel takes. With profiling off the
    # default args make this ProfilerArgs(None, False) -> no buffer arg.
    prof = smprof.ProfilerArgs(prof_buffer, profile_enabled, profile_show_overhead)

    pipelining_kernel(
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
        prof,
    ).launch(
        grid=(grid_n, grid_m, 1),
        block=(NUM_THREADS, 1, 1),
        stream=stream,
        # Require >= 2 resident blocks per SM (nvvm.minctasm). Unconstrained,
        # CuTeDSL's register allocator runs ILP-greedy to ~160 reg/thread (not
        # a spill, just loose) and caps occupancy at one block/SM on H100/H200.
        # The hint tightens the budget to <=128 reg/thread so two blocks share
        # an SM; combined with the sC/sO smem aliasing the footprint (96KB for
        # the default tile) leaves room for that second block.
        min_blocks_per_mp=2,
    )


# -----------------------------------------------------------------------------
# Host-side test harness (mirrors pipelining.py / cutedsl_dynamic_mma.py)
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
    g_clear = cute.compile(
        pipelining_gemm,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(o_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        True,
        options="--enable-tvm-ffi --generate-line-info",
    )
    g_accum = cute.compile(
        pipelining_gemm,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(o_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        False,
        options="--enable-tvm-ffi --generate-line-info",
    )
    return g_clear, g_accum


def _events_per_warp(K: int) -> int:
    """How many range events one warp emits for a given K.

    4 phase ranges (total / prologue / mainloop / epilogue) plus, when the
    register pipeline is disabled (``REG_STAGES == 1``), 2 ranges (compute, wait)
    per k-tile. Each ``event_start`` bumps the per-warp append counter once, so
    this is exactly the largest ``event_id`` the warp reaches; the profiler's
    ``max_events_per_warp`` must be at least this or trailing events are dropped.
    """
    k_tiles = (K + BLK_K - 1) // BLK_K
    return 4 + (2 * k_tiles if REG_STAGES < 2 else 0)


def _profile_once(M: int, N: int, K: int, out_dir: Path, show_overhead: bool) -> None:
    """Compile + run ONE profiled bf16 GEMM and dump its summary + Perfetto trace.

    ``show_overhead`` is a compile-time constant: with it on, the kernel records
    the extra start/end stamps and the export draws the two overhead bands; with
    it off, neither the device nor the trace carries any overhead data (the
    minimal-overhead reference). The two settings write to distinct files.
    """
    num_warps = NUM_THREADS // 32
    num_blocks = ((M + BLK_M - 1) // BLK_M) * ((N + BLK_N - 1) // BLK_N)

    torch.cuda.manual_seed_all(9527)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # Per-warp event capacity, sized from K with a small margin so nothing is
    # dropped (events beyond it are silently dropped — the counter keeps
    # advancing but no record is written).
    max_events = _events_per_warp(K) + 8
    profiler = smprof.SmProfiler(num_blocks, num_warps, max_events)
    for event_no, name in PROF_EVENT_NAMES.items():
        profiler.register_event(event_no, name)
    g = cute.compile(
        pipelining_gemm,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_cute_tensor(torch.empty(M, N, device="cuda", dtype=torch.bfloat16)),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.Float32,
        cutlass.BFloat16,
        False,
        profiler.cute_buffer(),
        True,
        show_overhead,
        options="--enable-tvm-ffi --generate-line-info",
    )
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    profiler.reset()
    g(a, b, c.clone(), out, profiler.buffer)
    torch.cuda.synchronize()

    ref = torch.addmm(c, a.float(), b.T.float()).bfloat16()
    tag = "overhead-on" if show_overhead else "overhead-off"
    print(
        f" profiled bf16 {M}x{N}x{K} [{tag}]: RE = {relative_error(out.float(), ref.float()) * 100:.2f}% ".center(
            PRINT_LENGTH, "="
        )
    )
    profiler.summarize(f"bf16 {M}x{N}x{K} [{tag}]")

    # Discriminating dump: with the bands recorded, show the raw start/end
    # overhead widths for a handful of events so a zero band (CSE'd reads /
    # sub-resolution timer) is distinguishable from a real one. Also persist the
    # raw buffer for offline re-analysis.
    if show_overhead:
        print(" raw overhead widths (ns) for first events ".center(PRINT_LENGTH, "-"))
        print(f"{'event':<12}{'start_ovh':>12}{'end_ovh':>12}{'interval':>12}")
        for _ in range(8):
            for record in profiler._records():
                name = PROF_EVENT_NAMES.get(record.event_no, f"event_{record.event_no}")
                # A band is only defined when its inner stamp was recorded (> 0). An
                # event closed by event_switch shares the boundary stamp and has no
                # end band, so en_early stays 0 -> report 0 rather than en - 0.
                start_ovh = record.st_late - record.st if record.st_late > 0 else 0
                end_ovh = record.en - record.en_early if record.en_early > 0 else 0
                print(f"{name:<12}{start_ovh:>12d}{end_ovh:>12d}{record.en - record.st:>12d}")
        buf_path = out_dir / f"pipelining_bf16_M{M}_N{N}_K{K}_buffer.pt"
        torch.save(profiler.buffer.cpu(), str(buf_path))
        print(f" raw buffer -> {buf_path} ".center(PRINT_LENGTH, "-"))

    suffix = "_overhead" if show_overhead else ""
    json_path = out_dir / f"pipelining_bf16_M{M}_N{N}_K{K}{suffix}.json"
    n_events = profiler.export_perfetto(str(json_path), show_overhead=show_overhead)
    print(f" Perfetto: {n_events} slices -> {json_path} ".center(PRINT_LENGTH, "-"))


def profile_demo() -> None:
    """Dump a profiled 4096x4096x4096 bf16 GEMM trace (overhead bands OFF).

    This is the only path that passes a profiler buffer (profiling on); the
    sweep in ``main`` compiles with profiling off, so no buffer is threaded
    through and the kernel matches ``cutedsl_pipelining.py``.

    Exports the full all-block trace without overhead bands (the overhead-on
    variant at this size would be a ~3x larger JSON; use a smaller M/N if you
    want to inspect the per-event overhead bands). ``max_events`` is derived from
    K so the per-warp capacity always covers the kernel's event count.
    """
    out_dir = Path(__file__).resolve().parent.parent / ".claude" / "perf" / "inkernel_profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    M, N, K = 4096, 4096, 4096
    _profile_once(M, N, K, out_dir, show_overhead=False)


def bench_overhead(M: int = 512, N: int = 512, K: int = 4096, iters: int = 100, repeats: int = 10) -> None:
    """Profiling-off vs profiling-on wall-clock comparison + overhead breakdown.

    Compiles three variants of the SAME bf16 GEMM and times each over
    ``repeats`` batches of ``iters`` launches (CUDA events, warmup discarded):

      * off          -- profiling disabled (the production kernel)
      * on (phases)  -- only the 4 coarse phase events (outside the hot loop)
      * on (full)    -- phases + the 2 per-k-tile events inside the mainloop

    ``on(phases) - off`` is the profiler's FIXED cost; ``on(full) - on(phases)``
    is the per-tile (hot-loop) cost -- i.e. where the overhead lives.
    """
    num_warps = NUM_THREADS // 32
    num_blocks = ((M + BLK_M - 1) // BLK_M) * ((N + BLK_N - 1) // BLK_N)
    k_tiles = (K + BLK_K - 1) // BLK_K

    torch.cuda.manual_seed_all(9527)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32).clone()
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    o_tmpl = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def _compile(profile_enabled, prof_buffer):
        return cute.compile(
            pipelining_gemm,
            make_cute_tensor(a),
            make_cute_tensor(b),
            make_cute_tensor(c),
            make_cute_tensor(o_tmpl),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            cutlass.Float32,
            cutlass.BFloat16,
            False,
            prof_buffer,
            profile_enabled,
            options="--enable-tvm-ffi --generate-line-info",
        )

    g_off = _compile(False, None)
    profiler = smprof.SmProfiler(num_blocks, num_warps, _events_per_warp(K) + 8)
    g_on = _compile(True, profiler.cute_buffer())

    def _time(run):
        for _ in range(10):  # warmup
            run()
        torch.cuda.synchronize()
        per_iter = []
        for _ in range(repeats):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                run()
            stop.record()
            torch.cuda.synchronize()
            per_iter.append(start.elapsed_time(stop) / iters * 1000.0)  # us
        t = torch.tensor(per_iter)
        return t.mean().item(), t.std().item()

    off = _time(lambda: g_off(a, b, c, out))
    profiler.reset()
    on = _time(lambda: g_on(a, b, c, out, profiler.buffer))

    events_per_warp = 4 + 2 * k_tiles  # 4 phases + 2 (compute, wait) per k-tile
    overhead = on[0] - off[0]
    print(
        f" Profiler overhead: bf16 {M}x{N}x{K}, {num_blocks} blocks x {num_warps} warps, "
        f"{k_tiles} k-tiles, {iters}x{repeats} ".center(PRINT_LENGTH, "=")
    )
    print(f"{'variant':<18}{'us/iter':>12}{'std':>10}{'vs off':>12}")
    print(f"{'off (no profile)':<18}{off[0]:>12.2f}{off[1]:>10.2f}{'--':>12}")
    print(f"{'on (profiled)':<18}{on[0]:>12.2f}{on[1]:>10.2f}{overhead:>+12.2f}")
    print("-" * PRINT_LENGTH)
    print(
        f"total overhead: {overhead:+.2f} us ({overhead / off[0] * 100:+.1f}% of off), "
        f"{events_per_warp} events/warp -> ~{overhead / events_per_warp * 1000:.0f} ns/event, "
        f"{overhead / k_tiles * 1000:.0f} ns/k-tile"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    # Sweep matches pipelining.py exactly.
    Ms = [16, 64, 128, 192, 256, 1024, 4096, 8192]
    Ns = [16, 64, 128, 192, 256, 1024, 4096, 8192]
    Ks = [16, 64, 128, 192, 256, 1024, 4096, 8192]
    exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    counters = {"succeed": 0, "failed": 0}
    torch.cuda.manual_seed_all(9527)

    M0, N0, K0 = 128, 128, 64

    # ----- Spec 1: fp16 in, fp16 acc, fp16 out (exercise only) -----
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
    profile_demo()
    main()
