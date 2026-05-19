"""TMA multicast + TMA reduce-add GEMM in CuTe DSL.

CuTe DSL counterpart of ``tma_multicast_reduce.cu`` / ``.py``. Builds on
the SM80 warp-MMA + TMA load/store skeleton from
``11-tma-load-store`` (NOT WGMMA / 13-warpgroup-mma) and adds three
Hopper features:

  * **Cluster launch** -- a 2D thread-block cluster (``cluster_shape =
    Shape<_2,_1,_1>`` in the C++ KernelSpec, i.e. ``dim3(2,1,1)``). Two
    CTAs along grid.x cooperate over a shared smem multicast for A.
  * **TMA multicast loads** -- when ``cluster.x > 1`` the A-tile is the
    same for every CTA in that cluster row, so A is loaded via
    ``CopyBulkTensorTileG2SMulticastOp``. B is loaded with a plain
    ``CopyBulkTensorTileG2SOp`` because ``cluster.y == 1`` here.
  * **TMA reduce-add store** -- when the C accumulator dtype equals the
    output dtype (Spec 1 only), the kernel converts the in-register
    accumulator into smem C and uses
    ``CopyReduceBulkTensorTileS2GOp(ReductionOp.ADD)`` to atomically add
    the tile into the gmem C buffer. When ``c is None`` the C++ harness
    aliases ``c = d = zeros`` so the result is just `c <- 0 + acc`; when
    ``c`` is provided we accumulate on top of the user-supplied tile.

The MMA atom mirrors the C++ KernelSpec exactly:
  * SM80 ``16x8x16`` warp-MMA atom (fp16xfp16=fp16, fp16xfp16=fp32,
    bf16xbf16=fp32) tiled with ``atom_layout_mnk = (2, 4, 1)`` and
    per-thread val expansion ``(1, 2, 2)`` -> tile (32, 64, 32),
    NUM_THREADS = 256.
  * S2R via universal copy (DSL lowers fp16/bf16 to LDSM.x4).

Dtype specs (mirrors the C++ harness):

  * fp16 = fp16 * fp16 + fp16   (fp16 accumulator, TMA reduce-add)
  * fp16 = fp16 * fp16 + fp32   (fp32 accumulator, fp16 out, TMA load-C)
  * bf16 = bf16 * bf16 + fp32   (fp32 accumulator, bf16 out, TMA load-C)

Sweep matches the C++ harness:
  Ms = [256, 1024, 4096, 8192]
  Ns = [256, 1024, 4096, 8192]
  Ks = [16, 64, 128, 192, 256, 1024, 4096, 8192]
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cutlass.utils.layout import LayoutEnum


# Block tile (matches the C++ KernelSpec defaults).
BLK_M = 128
BLK_N = 128
BLK_K = 64

# Cluster shape (matches the C++ ``ClusterShape = Shape<_2, _1, _1>``).
# cluster.x == 2 → two CTAs along grid.x share the same M-tile (A is
# multicast), each handles a distinct N-tile.
CLUSTER_X = 2
CLUSTER_Y = 1
CLUSTER_SIZE = CLUSTER_X * CLUSTER_Y

# SM80 warp-MMA configuration (mirrors the C++ KernelSpec).
MMA_INST_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 4, 1)
VAL_EXPAND_MNK = (1, 2, 2)
MMA_TILE_MNK = (
    ATOM_LAYOUT_MNK[0] * VAL_EXPAND_MNK[0] * MMA_INST_MNK[0],  # 32
    ATOM_LAYOUT_MNK[1] * VAL_EXPAND_MNK[1] * MMA_INST_MNK[1],  # 64
    ATOM_LAYOUT_MNK[2] * VAL_EXPAND_MNK[2] * MMA_INST_MNK[2],  # 32
)
NUM_THREADS = (
    ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2] * 32  # 256
)
NUM_WARPS = NUM_THREADS // 32  # 8


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def tma_multicast_reduce_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    tma_atom_c_load: cute.CopyAtom,
    mC_load: cute.Tensor,
    tma_atom_c_reduce: cute.CopyAtom,
    mC_reduce: cute.Tensor,
    tma_atom_d: cute.CopyAtom,
    mD: cute.Tensor,
    tiled_mma: cute.TiledMma,
    s2r_tiled_copy_a: cute.TiledCopy,
    s2r_tiled_copy_b: cute.TiledCopy,
    s2r_tiled_copy_c: cute.TiledCopy,
    r2s_tiled_copy_d: cute.TiledCopy,
    r2s_tiled_copy_c: cute.TiledCopy,
    sA_layout_staged: cute.ComposedLayout,
    sB_layout_staged: cute.ComposedLayout,
    sC_layout: cute.ComposedLayout,
    sD_layout: cute.ComposedLayout,
    tx_count_ab: cutlass.Constexpr,
    tx_count_c: cutlass.Constexpr,
    cta_layout_mnk: cute.Layout,
    cta_layout_vmnk: cute.Layout,
    c_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
    use_reduce_add: cutlass.Constexpr[bool],
    shared_storage_cls: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # ----- Cluster coord (rank-3 over M, N, K-singleton) -----
    block_rank_in_cluster = cute.arch.block_idx_in_cluster()
    cluster_coord_mnk = cta_layout_mnk.get_flat_coord(block_rank_in_cluster)

    # ----- Smem allocation via @cute.struct (dense_gemm pattern) -----
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
    sC = storage.sC.get_tensor(
        sC_layout.outer,
        swizzle=sC_layout.inner,
    )
    sD = storage.sD.get_tensor(
        sD_layout.outer,
        swizzle=sD_layout.inner,
    )
    sA = sA_full[None, None, 0]
    sB = sB_full[None, None, 0]

    # ----- Pipeline setup for the AB-load mbarriers -----
    # With A-multicast (CLUSTER_X=2) and B non-multicast (CLUSTER_Y=1),
    # mcast_size = num_mcast_a + num_mcast_b - 1 = 2 + 1 - 1 = 2.
    # Every consumer_release arrives on the partner CTA's empty barrier
    # too, so consumer arrive count must scale by mcast_size.
    mcast_size: cutlass.Constexpr = CLUSTER_X + CLUSTER_Y - 1
    consumer_arrive_cnt: cutlass.Constexpr = mcast_size * NUM_WARPS
    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.mainloop_mbar_array.data_ptr(),
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_arrive_cnt,
        ),
        tx_count=tx_count_ab,
        cta_layout_vmnk=cta_layout_vmnk,
    )

    # ----- (Optional) C-load pipeline -- only used in Specs 2/3 when
    # use_reduce_add=False AND is_gemm=False. Must be created before any
    # warp-spec / data-flow split.
    # NOTE: the C-load TMA atom is NON-multicast (plain G2S), so the
    # pipeline must be configured with a single-CTA cta_layout_vmnk even
    # when the kernel is launched in a cluster. Using the clustered
    # cta_layout_vmnk here causes empty-barrier arrival mismatches that
    # surface as CUDA_ERROR_LAUNCH_FAILED at large M*N with K=16.
    if cutlass.const_expr(not use_reduce_add):
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_mbar_array.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                NUM_THREADS,
            ),
            tx_count=tx_count_c,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

    # Ensure barrier init is visible cluster-wide before any TMA load.
    if cutlass.const_expr(CLUSTER_SIZE > 1):
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
    else:
        cute.arch.sync_threads()

    # ----- Tile global tensors -----
    tiler = (BLK_M, BLK_N, BLK_K)
    gA = cute.local_tile(mA, tiler=tiler, coord=(bidy, bidx, None), proj=(1, None, 1))
    gB = cute.local_tile(mB, tiler=tiler, coord=(bidy, bidx, None), proj=(None, 1, 1))
    gD = cute.local_tile(mD, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))
    gC_load = cute.local_tile(mC_load, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))
    gC_reduce = cute.local_tile(mC_reduce, tiler=tiler, coord=(bidy, bidx, 0), proj=(1, 1, None))

    # ----- TMA partition for A (multicast across cluster.x) -----
    sA_for_tma = cute.group_modes(sA_full, 0, 2)  # ((bM,bK), STAGE=1)
    gA_for_tma = cute.group_modes(gA, 0, 2)  # ((bM,bK), RestK)
    if cutlass.const_expr(CLUSTER_X > 1):
        # A is shared across cluster-mode-0 (grid.x) -- multicast across
        # axis 0 of the cluster layout. Pass per-axis coord & layout.
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape,
        )
        a_cta_crd = cluster_coord_mnk[0]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma,
            gA_for_tma,
        )
        # Mode along which the loaded tile is REPLICATED.
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk,
            cluster_coord_mnk,
            mode=0,
        )
    else:
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            sA_for_tma,
            gA_for_tma,
        )
        a_mcast_mask = cutlass.Int16(0)

    # ----- TMA partition for B (plain G2S; cluster.y == 1) -----
    sB_for_tma = cute.group_modes(sB_full, 0, 2)
    gB_for_tma = cute.group_modes(gB, 0, 2)
    if cutlass.const_expr(CLUSTER_Y > 1):
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape,
        )
        b_cta_crd = cluster_coord_mnk[1]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma,
            gB_for_tma,
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk,
            cluster_coord_mnk,
            mode=1,
        )
    else:
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            sB_for_tma,
            gB_for_tma,
        )
        b_mcast_mask = cutlass.Int16(0)

    # ----- TMA partitions for C-load and D-store (no multicast) -----
    sC_for_tma = cute.group_modes(
        cute.make_tensor(sC.iterator, cute.append(sC.layout, cute.make_layout(1))),
        0,
        2,
    )
    gC_load_for_tma = cute.group_modes(
        cute.make_tensor(gC_load.iterator, cute.append(gC_load.layout, cute.make_layout(1))),
        0,
        2,
    )
    tCsC_load, tCgC_load = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c_load,
        0,
        cute.make_layout(1),
        sC_for_tma,
        gC_load_for_tma,
    )

    gC_reduce_for_tma = cute.group_modes(
        cute.make_tensor(gC_reduce.iterator, cute.append(gC_reduce.layout, cute.make_layout(1))),
        0,
        2,
    )
    tCsC_reduce, tCgC_reduce = cute.nvgpu.cpasync.tma_partition(
        tma_atom_c_reduce,
        0,
        cute.make_layout(1),
        sC_for_tma,
        gC_reduce_for_tma,
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

    # ----- MMA fragments -----
    thr_mma = tiled_mma.get_slice(tid)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(sD))
    tCrC.fill(0.0)

    # ----- S2R partitions for A/B -----
    thr_s2r_a = s2r_tiled_copy_a.get_slice(tid)
    tAsA_s2r = thr_s2r_a.partition_S(sA)
    tArA_s2r = thr_s2r_a.retile(tCrA)
    thr_s2r_b = s2r_tiled_copy_b.get_slice(tid)
    tBsB_s2r = thr_s2r_b.partition_S(sB)
    tBrB_s2r = thr_s2r_b.retile(tCrB)

    num_k_tiles = cute.size(gA, mode=[2])

    mainloop_prod_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer,
        1,
    )
    mainloop_cons_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer,
        1,
    )

    # =========================================================================
    # MAINLOOP -- TMA multicast-load A and load B then warp-MMA, 1 stage.
    # =========================================================================
    for ik in cutlass.range(num_k_tiles, unroll=1):
        if warp_idx == 0:
            mainloop_pipeline.producer_acquire(mainloop_prod_state)
            if cutlass.const_expr(CLUSTER_X > 1):
                cute.copy(
                    tma_atom_a,
                    tAgA[None, ik],
                    tAsA[None, mainloop_prod_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                    mcast_mask=a_mcast_mask,
                )
            else:
                cute.copy(
                    tma_atom_a,
                    tAgA[None, ik],
                    tAsA[None, mainloop_prod_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                )
            if cutlass.const_expr(CLUSTER_Y > 1):
                cute.copy(
                    tma_atom_b,
                    tBgB[None, ik],
                    tBsB[None, mainloop_prod_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                    mcast_mask=b_mcast_mask,
                )
            else:
                cute.copy(
                    tma_atom_b,
                    tBgB[None, ik],
                    tBsB[None, mainloop_prod_state.index],
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_prod_state),
                )
            mainloop_pipeline.producer_commit(mainloop_prod_state)
            mainloop_prod_state.advance()

        mainloop_pipeline.consumer_wait(mainloop_cons_state)

        cute.copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r)
        cute.copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r)
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

        mainloop_pipeline.consumer_release(mainloop_cons_state)
        mainloop_cons_state.advance()

    # Cluster fence between mainloop and epilogue: smem sA/sB will be
    # aliased over by sC/sD writes, and we need every CTA in the cluster
    # to be done consuming A/B before that aliasing kicks in.
    if cutlass.const_expr(CLUSTER_SIZE > 1):
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
    else:
        cute.arch.sync_threads()

    # =========================================================================
    # EPILOGUE.
    # =========================================================================
    if cutlass.const_expr(use_reduce_add):
        # Spec 1: ComputeTypeC == OutType -- convert acc to OutType, R2S
        # into sC, then TMA reduce-add into gmem C (which the host has
        # bound to the user's c tensor, or to a zero-init D when c is
        # None).
        tCrD = cute.make_fragment_like(tCrC, out_dtype)
        tCrD.store(tCrC.load().to(out_dtype))

        thr_r2s_c = r2s_tiled_copy_c.get_slice(tid)
        tCrD_r2s = thr_r2s_c.retile(tCrD)
        tCsC_r2s = thr_r2s_c.partition_D(sC)
        cute.copy(r2s_tiled_copy_c, tCrD_r2s, tCsC_r2s)

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        cute.arch.sync_threads()

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.copy(tma_atom_c_reduce, tCsC_reduce[None, 0], tCgC_reduce[None, 0])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)
    else:
        # Specs 2/3: optionally TMA-load-C and fold into the accumulator,
        # then narrow to out_dtype, R2S into sD, TMA-store sD into gmem D.
        if cutlass.const_expr(not is_gemm):
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
                    tma_atom_c_load,
                    tCgC_load[None, 0],
                    tCsC_load[None, 0],
                    tma_bar_ptr=c_pipeline.producer_get_barrier(c_prod_state),
                )
                c_pipeline.producer_commit(c_prod_state)

            c_pipeline.consumer_wait(c_cons_state)

            tCrC_load = cute.make_fragment_like(tCrC, c_dtype)
            thr_s2r_c = s2r_tiled_copy_c.get_slice(tid)
            tCsC_s2r = thr_s2r_c.partition_S(sC)
            tCrC_s2r = thr_s2r_c.retile(tCrC_load)
            cute.copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r)

            c_pipeline.consumer_release(c_cons_state)

            c_vec = tCrC_load.load().to(tCrC.element_type)
            tCrC.store(tCrC.load() + c_vec)

            cute.arch.sync_threads()

        tCrD = cute.make_fragment_like(tCrC, out_dtype)
        tCrD.store(tCrC.load().to(out_dtype))

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
# Host (jit) wrapper -- build TMA atoms, smem layouts, copies, then launch.
# -----------------------------------------------------------------------------


@cute.jit
def tma_multicast_reduce(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mD: cute.Tensor,
    stream: CUstream,
    acc_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
    use_reduce_add: cutlass.Constexpr[bool],
):
    # ----- Tiled MMA (SM80 16x8x16 warp-MMA) -----
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

    # ----- Swizzled smem layouts selected from the major-mode extent -----
    a_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, mA.element_type, BLK_K),
        mA.element_type,
    )
    b_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, mB.element_type, BLK_K),
        mB.element_type,
    )
    c_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, mC.element_type, BLK_N),
        mC.element_type,
    )
    d_atom = sm90_utils.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, out_dtype, BLK_N),
        out_dtype,
    )
    sA_layout_staged = cute.tile_to_shape(a_atom, (BLK_M, BLK_K, 1), order=(0, 1, 2))
    sB_layout_staged = cute.tile_to_shape(b_atom, (BLK_N, BLK_K, 1), order=(0, 1, 2))
    sC_layout = cute.tile_to_shape(c_atom, (BLK_M, BLK_N), order=(0, 1))
    sD_layout = cute.tile_to_shape(d_atom, (BLK_M, BLK_N), order=(0, 1))

    sA_layout = cute.slice_(sA_layout_staged, (None, None, 0))
    sB_layout = cute.slice_(sB_layout_staged, (None, None, 0))

    # ----- TMA atoms -----
    # A: multicast iff cluster.x > 1.
    if cutlass.const_expr(CLUSTER_X > 1):
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(),
            mA,
            sA_layout,
            (BLK_M, BLK_K),
            num_multicast=CLUSTER_X,
        )
    else:
        tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mA,
            sA_layout,
            (BLK_M, BLK_K),
        )

    # B: multicast iff cluster.y > 1.
    if cutlass.const_expr(CLUSTER_Y > 1):
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(),
            mB,
            sB_layout,
            (BLK_N, BLK_K),
            num_multicast=CLUSTER_Y,
        )
    else:
        tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            mB,
            sB_layout,
            (BLK_N, BLK_K),
        )

    # C-load: plain G2S TMA (used only when use_reduce_add=False).
    tma_atom_c_load, tma_tensor_c_load = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mC,
        sC_layout,
        (BLK_M, BLK_N),
    )

    # C-reduce-add: S2G + ADD (used only when use_reduce_add=True).
    tma_atom_c_reduce, tma_tensor_c_reduce = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(
            reduction_kind=cute.ReductionOp.ADD,
        ),
        mC,
        sC_layout,
        (BLK_M, BLK_N),
    )

    # D-store: plain S2G TMA.
    tma_atom_d, tma_tensor_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        mD,
        sD_layout,
        (BLK_M, BLK_N),
    )

    # ----- S2R / R2S copy atoms -----
    universal = cute.nvgpu.CopyUniversalOp()
    s2r_atom_a = cute.make_copy_atom(universal, mA.element_type)
    s2r_atom_b = cute.make_copy_atom(universal, mB.element_type)
    s2r_atom_c = cute.make_copy_atom(universal, mC.element_type)
    r2s_atom_d = cute.make_copy_atom(universal, out_dtype)
    r2s_atom_c = cute.make_copy_atom(universal, out_dtype)
    s2r_tiled_copy_a = cute.make_tiled_copy_A(s2r_atom_a, tm)
    s2r_tiled_copy_b = cute.make_tiled_copy_B(s2r_atom_b, tm)
    s2r_tiled_copy_c = cute.make_tiled_copy_C(s2r_atom_c, tm)
    r2s_tiled_copy_d = cute.make_tiled_copy_C(r2s_atom_d, tm)
    r2s_tiled_copy_c = cute.make_tiled_copy_C(r2s_atom_c, tm)

    # ----- Transaction byte counts -----
    a_bytes = mA.element_type.width // 8
    b_bytes = mB.element_type.width // 8
    c_bytes = mC.element_type.width // 8
    tx_count_ab = BLK_M * BLK_K * a_bytes + BLK_N * BLK_K * b_bytes
    tx_count_c = BLK_M * BLK_N * c_bytes

    # ----- Shared storage layout -----
    @cute.struct
    class SharedStorage:
        mainloop_mbar_array: cute.struct.MemRange[cutlass.Int64, 2]
        c_mbar_array: cute.struct.MemRange[cutlass.Int64, 2]
        sA: cute.struct.Align[
            cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout_staged)],
            1024,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout_staged)],
            1024,
        ]
        sC: cute.struct.Align[
            cute.struct.MemRange[mC.element_type, cute.cosize(sC_layout)],
            1024,
        ]
        sD: cute.struct.Align[
            cute.struct.MemRange[out_dtype, cute.cosize(sD_layout)],
            1024,
        ]

    # Cluster layouts.
    cta_layout_mnk = cute.make_layout((CLUSTER_X, CLUSTER_Y, 1))
    cta_layout_vmnk = cute.make_layout((1, CLUSTER_X, CLUSTER_Y, 1))

    # ----- Grid -----
    M, _ = mA.shape
    N, _ = mB.shape
    grid_n = (N + BLK_N - 1) // BLK_N
    grid_m = (M + BLK_M - 1) // BLK_M

    tma_multicast_reduce_kernel(
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        tma_atom_c_load,
        tma_tensor_c_load,
        tma_atom_c_reduce,
        tma_tensor_c_reduce,
        tma_atom_d,
        tma_tensor_d,
        tm,
        s2r_tiled_copy_a,
        s2r_tiled_copy_b,
        s2r_tiled_copy_c,
        r2s_tiled_copy_d,
        r2s_tiled_copy_c,
        sA_layout_staged,
        sB_layout_staged,
        sC_layout,
        sD_layout,
        tx_count_ab,
        tx_count_c,
        cta_layout_mnk,
        cta_layout_vmnk,
        mC.element_type,
        out_dtype,
        is_gemm,
        use_reduce_add,
        SharedStorage,
    ).launch(
        grid=(grid_n, grid_m, 1),
        block=(NUM_THREADS, 1, 1),
        cluster=(CLUSTER_X, CLUSTER_Y, 1),
        stream=stream,
    )


# -----------------------------------------------------------------------------
# Host-side test harness (mirrors tma_multicast_reduce.py exactly).
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


def _compile_variants(a_template, b_template, c_template, d_template, acc_dtype, out_dtype, use_reduce_add):
    """Compile (is_gemm=True, is_gemm=False) pair for one dtype spec."""
    g_clear = cute.compile(
        tma_multicast_reduce,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(d_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        True,
        use_reduce_add,
        options="--enable-tvm-ffi",
    )
    g_accum = cute.compile(
        tma_multicast_reduce,
        make_cute_tensor(a_template),
        make_cute_tensor(b_template),
        make_cute_tensor(c_template),
        make_cute_tensor(d_template),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        acc_dtype,
        out_dtype,
        False,
        use_reduce_add,
        options="--enable-tvm-ffi",
    )
    return g_clear, g_accum


# Full sweep -- must match the C++ harness exactly.
Ms = [256, 1024, 4096, 8192]
Ns = [256, 1024, 4096, 8192]
Ks = [16, 64, 128, 192, 256, 1024, 4096, 8192]
EXPS = [(m, n, k) for m in Ms for n in Ns for k in Ks]


def _spec1_mm_call(fp16_reduce_clear, a, b, m, n):
    """MM-path for Spec 1: returns kernel(a,b,None) + c via the
    reduce-add atom. The host has to mimic the C++ harness:
      c_buf = zeros(M, N)
      kernel writes c_buf += acc(a,b)
      return c_buf  (this IS the kernel's output)
    """
    c_buf = torch.zeros(m, n, device="cuda", dtype=torch.float16)
    # d is unused for the reduce-add path; pass any same-shape buffer.
    d_dummy = torch.empty(m, n, device="cuda", dtype=torch.float16)
    fp16_reduce_clear(a, b, c_buf, d_dummy)
    return c_buf


def _spec1_mma_call(fp16_reduce_accum, a, b, c, m, n):
    """MMA-path for Spec 1: kernel writes c += acc(a,b) in place."""
    c_buf = c.clone()
    d_dummy = torch.empty(m, n, device="cuda", dtype=torch.float16)
    fp16_reduce_accum(a, b, c_buf, d_dummy)
    return c_buf


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    counters = {"succeed": 0, "failed": 0}
    torch.cuda.manual_seed_all(9527)

    M0, N0, K0 = 256, 256, 64

    # ----- Spec 1: fp16 in, fp16 acc, fp16 out (TMA reduce-add) -----
    print(" Compiling fp16 in / fp16 acc / fp16 out (reduce-add) ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M0, K0, device="cuda", dtype=torch.float16)
    b_t = torch.empty(N0, K0, device="cuda", dtype=torch.float16)
    c_t = torch.empty(M0, N0, device="cuda", dtype=torch.float16)
    d_t = torch.empty(M0, N0, device="cuda", dtype=torch.float16)
    fp16_reduce_clear, fp16_reduce_accum = _compile_variants(
        a_t,
        b_t,
        c_t,
        d_t,
        cutlass.Float16,
        cutlass.Float16,
        True,
    )

    # ----- Spec 2: fp16 in, fp32 acc, fp16 out (TMA load-C) -----
    print(" Compiling fp16 in / fp32 acc / fp16 out (load-C) ... ".center(PRINT_LENGTH, "-"))
    fp16f32_clear, fp16f32_accum = _compile_variants(
        a_t,
        b_t,
        c_t,
        d_t,
        cutlass.Float32,
        cutlass.Float16,
        False,
    )

    # ----- Spec 3: bf16 in, fp32 acc, bf16 out (TMA load-C) -----
    print(" Compiling bf16 in / fp32 acc / bf16 out (load-C) ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M0, K0, device="cuda", dtype=torch.bfloat16)
    b_t = torch.empty(N0, K0, device="cuda", dtype=torch.bfloat16)
    c_t = torch.empty(M0, N0, device="cuda", dtype=torch.bfloat16)
    d_t = torch.empty(M0, N0, device="cuda", dtype=torch.bfloat16)
    bf16_clear, bf16_accum = _compile_variants(
        a_t,
        b_t,
        c_t,
        d_t,
        cutlass.Float32,
        cutlass.BFloat16,
        False,
    )

    # ----- Sweep: Spec 1 (fp16 / fp16 / fp16) MM vs MMA equality -----
    print(" fp16 = fp16 * fp16 + fp16 (MM vs MMA) ".center(PRINT_LENGTH, "="))
    for m, n, k in EXPS:
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(n, k, device="cuda", dtype=torch.float16)
        c = torch.randn(m, n, device="cuda", dtype=torch.float16)

        o1 = _spec1_mm_call(fp16_reduce_clear, a, b, m, n) + c
        o2 = _spec1_mma_call(fp16_reduce_accum, a, b, c, m, n)
        torch.cuda.synchronize()
        compare_matrix(o1, o2, counters)

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
