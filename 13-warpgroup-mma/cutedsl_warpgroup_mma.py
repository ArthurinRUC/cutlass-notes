"""WGMMA (warpgroup-MMA) GEMM in CuTe DSL.

CuTe DSL counterpart of ``warpgroup_mma.cu`` / ``warpgroup_mma.py``.
This is the Hopper (SM90+) warpgroup-MMA port that follows the
TMA-load / TMA-store skeleton from ``11-tma-load-store`` but replaces
the SM80 16x8x16 warp-level MMA mainloop with Hopper's WGMMA:

  * **Tiled MMA**: 64x256x16 WGMMA atom (``MmaF16BF16Op`` with
    ``OperandSource.SMEM``) tiled (2, 1, 1) across M -> 128x256x16 per
    issue. Two warpgroups (256 threads) cooperate.
  * **Load path**: TMA loads of A/B into smem with the
    ``cutlass.pipeline.PipelineTmaAsync`` helper. Multi-stage (default
    NUM_STAGES=3) to match the C++ reference.
  * **Mainloop**: per K-tile, ``warpgroup.fence`` -> loop over
    MMA_K=BLK_K/16 sub-blocks issuing ``cute.gemm`` -> ``commit_group``
    -> ``wait_group(0)``. WGMMA reads A and B as smem descriptors;
    accumulator lives in registers across all K-tiles.
  * **Store path**: accumulator R2S'd into smem, then TMA-stored
    smem -> gmem by warp 0.

Three dtype specs are exercised (mirrors the C++ harness):

  * fp16 = fp16 * fp16 + fp16  (fp16 accumulator)
  * fp16 = fp16 * fp16 + fp32  (fp32 accumulator, fp16 out)
  * bf16 = bf16 * bf16 + fp32  (fp32 accumulator, bf16 out)

**Shape sweep**: The C++ harness sweeps 8x8x8 = 512 shapes per dtype
spec. This DSL port runs the full sweep; sub-tile residues (e.g. N=16
< instruction N=256, K=16 < BLK_K=64) are handled by TMA OOB zero-fill.
See ``EXPS`` below.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.nvgpu.warpgroup import (
    OperandMajorMode,
    OperandSource,
)
from cutlass.cute.runtime import from_dlpack, make_fake_stream


# Block tile (matches the C++ KernelSpec defaults in warpgroup_mma.cu).
BLK_M = 128
BLK_N = 256
BLK_K = 64
NUM_STAGES = 3

# WGMMA atom is 64x256x16 with atom layout (2, 1, 1) over M -> CTA tile
# 128x256x16 per issue. Two warpgroups => 256 threads.
ATOM_LAYOUT_MNK = (2, 1, 1)
NUM_WARPGROUPS = ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2]
NUM_THREADS_PER_WARPGROUP = 128
NUM_THREADS = NUM_WARPGROUPS * NUM_THREADS_PER_WARPGROUP  # 256
NUM_WARPS = NUM_THREADS // 32  # 8


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def warpgroup_mma_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_c: cute.CopyAtom,
    mC: cute.Tensor,
    tma_atom_d: cute.CopyAtom,
    mD: cute.Tensor,
    tiled_mma: cute.TiledMma,
    s2r_tiled_copy_c: cute.TiledCopy,
    r2s_tiled_copy_d: cute.TiledCopy,
    sA_layout_staged: cute.ComposedLayout,
    sB_layout_staged: cute.ComposedLayout,
    sC_layout: cute.ComposedLayout,
    sD_layout: cute.ComposedLayout,
    tx_count_ab: cutlass.Constexpr,
    tx_count_c: cutlass.Constexpr,
    cta_layout_vmnk: cute.Layout,
    acc_dtype: cutlass.Constexpr,
    c_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
    shared_storage_cls: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ----- Smem allocation -----
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage_cls)

    sA_full = storage.sA.get_tensor(
        sA_layout_staged.outer,
        swizzle=sA_layout_staged.inner,
    )
    sB_full = storage.sB.get_tensor(
        sB_layout_staged.outer,
        swizzle=sB_layout_staged.inner,
    )
    # Alias sC/sD on top of sB's smem region (sB has the larger byte
    # footprint of the two AB staged buffers, and the mainloop fully
    # completes — with sync_threads — before the epilogue touches sC/sD,
    # so this overlap is safe and saves ~128KB of dynamic smem.
    sC_ptr = cute.recast_ptr(sB_full.iterator, sC_layout.inner, dtype=c_dtype)
    sC = cute.make_tensor(sC_ptr, sC_layout.outer)
    sD_ptr = cute.recast_ptr(sB_full.iterator, sD_layout.inner, dtype=out_dtype)
    sD = cute.make_tensor(sD_ptr, sD_layout.outer)

    # ----- Pipeline (TMA load mainloop barriers) -----
    # Producer = single thread that issues TMA; consumer = all warps in the
    # CTA (DSL pipeline counts warps, not threads, for arrival count on
    # non-clustered SM90 TMA).
    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mainloop_mbar_array.data_ptr(),
        num_stages=NUM_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            NUM_WARPS,
        ),
        tx_count=tx_count_ab,
        cta_layout_vmnk=cta_layout_vmnk,
    )

    # ----- Tile global tensors -----
    tiler = (BLK_M, BLK_N, BLK_K)
    gA = cute.local_tile(mA, tiler=tiler, coord=(bidy, bidx, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=tiler, coord=(bidy, bidx, None), proj=(None, 1, 1))
    gC = cute.local_tile(mC, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))
    gD = cute.local_tile(mD, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))

    # ----- TMA partition for A/B loads (per K-tile, indexed by stage) -----
    sA_for_tma = cute.group_modes(sA_full, 0, 2)  # ((bM,bK), STAGE)
    gA_for_tma = cute.group_modes(gA, 0, 2)  # ((bM,bK), RestK)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        sA_for_tma,
        gA_for_tma,
    )
    sB_for_tma = cute.group_modes(sB_full, 0, 2)
    gB_for_tma = cute.group_modes(gB, 0, 2)
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        sB_for_tma,
        gB_for_tma,
    )

    # C/D are rank-2; pad with a singleton outer mode for tma_partition.
    sC_for_tma = cute.group_modes(
        cute.make_tensor(sC.iterator, cute.append(sC.layout, cute.make_layout(1))),
        0,
        2,
    )
    gC_for_tma = cute.group_modes(
        cute.make_tensor(gC.iterator, cute.append(gC.layout, cute.make_layout(1))),
        0,
        2,
    )
    tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        sC_for_tma,
        gC_for_tma,
    )
    sD_for_tma = cute.group_modes(
        cute.make_tensor(sD.iterator, cute.append(sD.layout, cute.make_layout(1))),
        0,
        2,
    )
    gD_for_tma = cute.group_modes(
        cute.make_tensor(gD.iterator, cute.append(gD.layout, cute.make_layout(1))),
        0,
        2,
    )
    tDsD, tDgD = cute.nvgpu.cpasync.tma_partition(
        tma_atom_d,
        0,
        cute.make_layout(1),
        sD_for_tma,
        gD_for_tma,
    )

    # ----- WGMMA fragments (smem descriptors for A/B; regs for C) -----
    warpgroup_idx = cute.arch.make_warp_uniform(tid // NUM_THREADS_PER_WARPGROUP)
    warpgroup_thread_layout = cute.make_layout(
        NUM_WARPGROUPS,
        stride=NUM_THREADS_PER_WARPGROUP,
    )
    thr_mma = tiled_mma.get_slice(warpgroup_thread_layout(warpgroup_idx))

    # sA_full has shape (BLK_M, BLK_K, NUM_STAGES). partition_A walks all
    # three modes -> tCsA has (MMA, MMA_M, MMA_K, NUM_STAGES).
    tCsA = thr_mma.partition_A(sA_full)
    tCsB = thr_mma.partition_B(sB_full)
    tCrA = tiled_mma.make_fragment_A(tCsA)
    tCrB = tiled_mma.make_fragment_B(tCsB)

    # Accumulator (regs) sized to this warpgroup's slice of the CTA tile.
    tCgC_shape = thr_mma.partition_C(gC).shape
    accumulators = cute.make_rmem_tensor(tCgC_shape, acc_dtype)
    accumulators.fill(0.0)

    num_k_tiles = cute.size(gA, mode=[2])

    mainloop_prod_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        NUM_STAGES,
    )
    mainloop_cons_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        NUM_STAGES,
    )

    # =========================================================================
    # Prefetch: warp 0 issues TMA loads for up to NUM_STAGES K-tiles.
    # =========================================================================
    prefetch_count = cutlass.max(cutlass.min(NUM_STAGES, num_k_tiles), 0)
    if warp_idx == 0:
        for prefetch_idx in cutlass.range(prefetch_count, unroll=1):
            mainloop_pipeline.producer_acquire(mainloop_prod_state)
            cute.copy(
                tma_atom_a,
                tAgA[None, mainloop_prod_state.count],
                tAsA[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                    mainloop_prod_state,
                ),
            )
            cute.copy(
                tma_atom_b,
                tBgB[None, mainloop_prod_state.count],
                tBsB[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                    mainloop_prod_state,
                ),
            )
            mainloop_pipeline.producer_commit(mainloop_prod_state)
            mainloop_prod_state.advance()

    # =========================================================================
    # MAINLOOP: per K-tile, fence -> WGMMA over MMA_K sub-blocks -> commit
    # -> wait_group(0). Then warp 0 issues the next TMA load (if any).
    # =========================================================================
    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
    num_k_blocks = cute.size(tCrA, mode=[2])

    for ik in cutlass.range(num_k_tiles, unroll=1):
        # Wait for stage `mainloop_cons_state.index` buffer to be full.
        mainloop_pipeline.consumer_wait(mainloop_cons_state)

        cute.nvgpu.warpgroup.fence()
        for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
            k_block_coord = (None, None, k_block_idx, mainloop_cons_state.index)
            cute.gemm(
                tiled_mma,
                accumulators,
                tCrA[k_block_coord],
                tCrB[k_block_coord],
                accumulators,
            )
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)

        mainloop_pipeline.consumer_release(mainloop_cons_state)
        mainloop_cons_state.advance()

        # Issue next TMA load (if there are still K-tiles to fetch).
        if warp_idx == 0 and mainloop_prod_state.count < num_k_tiles:
            mainloop_pipeline.producer_acquire(mainloop_prod_state)
            cute.copy(
                tma_atom_a,
                tAgA[None, mainloop_prod_state.count],
                tAsA[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                    mainloop_prod_state,
                ),
            )
            cute.copy(
                tma_atom_b,
                tBgB[None, mainloop_prod_state.count],
                tBsB[None, mainloop_prod_state.index],
                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                    mainloop_prod_state,
                ),
            )
            mainloop_pipeline.producer_commit(mainloop_prod_state)
            mainloop_prod_state.advance()

    # All warps must agree before we start mutating sC/sD (which alias sA).
    cute.arch.sync_threads()

    # =========================================================================
    # Optional C-add: TMA-load C, S2R it, fold into accumulator.
    # =========================================================================
    if cutlass.const_expr(not is_gemm):
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_mbar_array.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_WARPS,
            ),
            tx_count=tx_count_c,
            cta_layout_vmnk=cta_layout_vmnk,
        )
        c_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            1,
        )
        c_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            1,
        )

        if warp_idx == 0:
            c_pipeline.producer_acquire(c_prod_state)
            cute.copy(
                tma_atom_c,
                tCgC[None, 0],
                tCsC[None, 0],
                tma_bar_ptr=c_pipeline.producer_get_barrier(c_prod_state),
            )
            c_pipeline.producer_commit(c_prod_state)

        c_pipeline.consumer_wait(c_cons_state)

        # The S2R-side tiled_copy_C is built off the same TiledMMA, so the
        # per-thread fragment shape matches `accumulators` exactly.
        tCrC_load = cute.make_fragment_like(accumulators, c_dtype)
        thr_s2r_c = s2r_tiled_copy_c.get_slice(tid)
        tCsC_s2r = thr_s2r_c.partition_S(sC)
        tCrC_s2r = thr_s2r_c.retile(tCrC_load)
        cute.copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r)

        c_pipeline.consumer_release(c_cons_state)

        c_vec = tCrC_load.load().to(accumulators.element_type)
        accumulators.store(accumulators.load() + c_vec)

        cute.arch.sync_threads()

    # =========================================================================
    # EPILOGUE: narrow to out_dtype, R2S into smem, TMA store.
    # =========================================================================
    tCrD = cute.make_fragment_like(accumulators, out_dtype)
    tCrD.store(accumulators.load().to(out_dtype))

    thr_r2s_d = r2s_tiled_copy_d.get_slice(tid)
    tDrD_r2s = thr_r2s_d.retile(tCrD)
    tDsD_r2s = thr_r2s_d.partition_D(sD)
    cute.copy(r2s_tiled_copy_d, tDrD_r2s, tDsD_r2s)

    cute.arch.fence_proxy(
        cute.arch.ProxyKind.async_shared,
        space=cute.arch.SharedSpace.shared_cta,
    )
    cute.arch.sync_threads()

    if warp_idx == 0:
        with cute.arch.elect_one():
            cute.copy(tma_atom_d, tDsD[None, 0], tDgD[None, 0])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=False)


# -----------------------------------------------------------------------------
# Host (jit) wrapper
# -----------------------------------------------------------------------------


@cute.jit
def warpgroup_mma_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mD: cute.Tensor,
    stream: CUstream,
    acc_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    # ----- TiledMMA (Hopper WGMMA, A from SMEM, K-major A/B) -----
    op = cute.nvgpu.warpgroup.MmaF16BF16Op(
        mA.element_type,
        acc_dtype,
        (64, BLK_N, 16),
        OperandSource.SMEM,
        OperandMajorMode.K,
        OperandMajorMode.K,
    )
    tm = cute.make_tiled_mma(cute.make_mma_atom(op), ATOM_LAYOUT_MNK)

    # ----- Swizzled smem layouts (GMMA K_SW128 atom) -----
    a_atom = sm90_utils.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        mA.element_type,
    )
    b_atom = sm90_utils.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        mB.element_type,
    )
    c_atom = sm90_utils.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        mC.element_type,
    )
    d_atom = sm90_utils.make_smem_layout_atom(
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128,
        out_dtype,
    )
    sA_layout_staged = cute.tile_to_shape(
        a_atom,
        (BLK_M, BLK_K, NUM_STAGES),
        order=(0, 1, 2),
    )
    sB_layout_staged = cute.tile_to_shape(
        b_atom,
        (BLK_N, BLK_K, NUM_STAGES),
        order=(0, 1, 2),
    )
    sC_layout = cute.tile_to_shape(c_atom, (BLK_M, BLK_N), order=(0, 1))
    sD_layout = cute.tile_to_shape(d_atom, (BLK_M, BLK_N), order=(0, 1))

    sA_layout_one = cute.slice_(sA_layout_staged, (None, None, 0))
    sB_layout_one = cute.slice_(sB_layout_staged, (None, None, 0))

    # ----- TMA atoms / tensors -----
    tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        sA_layout_one,
        (BLK_M, BLK_K),
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mB,
        sB_layout_one,
        (BLK_N, BLK_K),
    )
    tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mC,
        sC_layout,
        (BLK_M, BLK_N),
    )
    tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        mD,
        sD_layout,
        (BLK_M, BLK_N),
    )

    # ----- S2R / R2S copy atoms (universal, built off TiledMMA) -----
    universal = cute.nvgpu.CopyUniversalOp()
    s2r_atom_c = cute.make_copy_atom(universal, mC.element_type)
    r2s_atom_d = cute.make_copy_atom(universal, out_dtype)
    s2r_tiled_copy_c = cute.make_tiled_copy_C(s2r_atom_c, tm)
    r2s_tiled_copy_d = cute.make_tiled_copy_C(r2s_atom_d, tm)

    # ----- Transaction byte counts -----
    a_bytes = mA.element_type.width // 8
    b_bytes = mB.element_type.width // 8
    c_bytes = mC.element_type.width // 8
    tx_count_ab = BLK_M * BLK_K * a_bytes + BLK_N * BLK_K * b_bytes
    tx_count_c = BLK_M * BLK_N * c_bytes

    # ----- Shared storage -----
    @cute.struct
    class SharedStorage:
        # 2 Int64 per stage (full+empty) per pipeline.
        mainloop_mbar_array: cute.struct.MemRange[cutlass.Int64, 2 * NUM_STAGES]
        c_mbar_array: cute.struct.MemRange[cutlass.Int64, 2]
        sA: cute.struct.Align[
            cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout_staged)],
            1024,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout_staged)],
            1024,
        ]

    cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))

    M, _ = mA.shape
    N, _ = mB.shape
    grid_n = (N + BLK_N - 1) // BLK_N
    grid_m = (M + BLK_M - 1) // BLK_M

    warpgroup_mma_kernel(
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_c,
        tma_tensor_c,
        tma_atom_d,
        tma_tensor_d,
        tm,
        s2r_tiled_copy_c,
        r2s_tiled_copy_d,
        sA_layout_staged,
        sB_layout_staged,
        sC_layout,
        sD_layout,
        tx_count_ab,
        tx_count_c,
        cta_layout_vmnk,
        acc_dtype,
        mC.element_type,
        out_dtype,
        is_gemm,
        SharedStorage,
    ).launch(
        grid=(grid_n, grid_m, 1),
        block=(NUM_THREADS, 1, 1),
        stream=stream,
    )


# -----------------------------------------------------------------------------
# Host-side test harness
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
    return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)


def _compile_pair(a_template, b_template, c_template, d_template, acc_dtype, out_dtype):
    g_clear = cute.compile(
        warpgroup_mma_host,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(d_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        True,
        options="--enable-tvm-ffi",
    )
    g_accum = cute.compile(
        warpgroup_mma_host,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(d_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        False,
        options="--enable-tvm-ffi",
    )
    return g_clear, g_accum


# Full shape sweep — matches warpgroup_mma.py exactly.
_Ms = [16, 64, 128, 192, 256, 1024, 4096, 8192]
_Ns = [16, 64, 128, 192, 256, 1024, 4096, 8192]
_Ks = [16, 64, 128, 192, 256, 1024, 4096, 8192]
EXPS = [(m, n, k) for m in _Ms for n in _Ns for k in _Ks]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    counters = {"succeed": 0, "failed": 0}
    torch.cuda.manual_seed_all(9527)

    M0, N0, K0 = BLK_M, BLK_N, BLK_K

    # ----- Spec 1: fp16 in, fp16 acc, fp16 out -----
    print(" Compiling fp16 in / fp16 acc / fp16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M0, K0, device="cuda", dtype=torch.float16)
    b_t = torch.empty(N0, K0, device="cuda", dtype=torch.float16)
    c_t = torch.empty(M0, N0, device="cuda", dtype=torch.float16)
    d_t = torch.empty(M0, N0, device="cuda", dtype=torch.float16)
    fp16_clear, fp16_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        d_t,
        cutlass.Float16,
        cutlass.Float16,
    )

    # ----- Spec 2: fp16 in, fp32 acc, fp16 out -----
    print(" Compiling fp16 in / fp32 acc / fp16 out ... ".center(PRINT_LENGTH, "-"))
    fp16f32_clear, fp16f32_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        d_t,
        cutlass.Float32,
        cutlass.Float16,
    )

    # ----- Spec 3: bf16 in, fp32 acc, bf16 out -----
    print(" Compiling bf16 in / fp32 acc / bf16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M0, K0, device="cuda", dtype=torch.bfloat16)
    b_t = torch.empty(N0, K0, device="cuda", dtype=torch.bfloat16)
    c_t = torch.empty(M0, N0, device="cuda", dtype=torch.bfloat16)
    d_t = torch.empty(M0, N0, device="cuda", dtype=torch.bfloat16)
    bf16_clear, bf16_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        d_t,
        cutlass.Float32,
        cutlass.BFloat16,
    )

    # ----- Sweep: Spec 1 (fp16 / fp16 / fp16, exercise only) -----
    print(" fp16 = fp16 * fp16 + fp16 (exercise only) ".center(PRINT_LENGTH, "="))
    for m, n, k in EXPS:
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(n, k, device="cuda", dtype=torch.float16)
        c = torch.randn(m, n, device="cuda", dtype=torch.float16)

        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16_clear(a, b, c.clone(), out)
        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16_accum(a, b, c.clone(), out)
        torch.cuda.synchronize()

    # ----- Sweep: Spec 2 (fp16 in, fp32 acc, fp16 out) -----
    print(" fp16 = fp32_acc(fp16 * fp16) + fp16 ".center(PRINT_LENGTH, "="))
    for m, n, k in EXPS:
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(n, k, device="cuda", dtype=torch.float16)
        c = torch.randn(m, n, device="cuda", dtype=torch.float16)

        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16f32_clear(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(out, torch.matmul(a, b.T), counters)

        out = torch.empty(m, n, device="cuda", dtype=torch.float16)
        fp16f32_accum(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(out, torch.addmm(c.float(), a.float(), b.T.float()).half(), counters)

    # ----- Sweep: Spec 3 (bf16 in, fp32 acc, bf16 out) -----
    print(" bf16 = fp32_acc(bf16 * bf16) + bf16 ".center(PRINT_LENGTH, "="))
    for m, n, k in EXPS:
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        bf16_clear(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(out, torch.matmul(a.float(), b.T.float()).bfloat16(), counters)

        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        bf16_accum(a, b, c.clone(), out)
        torch.cuda.synchronize()
        compare_matrix(
            out,
            torch.addmm(c.float(), a.float(), b.T.float()).bfloat16(),
            counters,
        )

    print(f" Summary: {counters['succeed']} Succeed, {counters['failed']} Failed ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
