"""Swizzled smem (128x128x64) in CuTe DSL.

CuTe DSL counterpart of ``swizzling.cu`` / ``swizzling.py``. Same flow
as ``06-block-copy``: gmem --(cp.async G2S)--> smem --(ldmatrix S2R)-->
rmem -> gemm -> rmem --(R2S)--> smem --(S2G)--> gmem. The new
ingredient is swizzled smem layouts on A and B, which kill the bank
conflicts that the strided ldmatrix pattern would otherwise generate:

  swizzle_atom = Swizzle<3,3,3> o (8 x min(64, kBlockK)):(min(64, kBlockK), 1)
  sA_layout    = tile_to_shape(swizzle_atom, (kBlockM, kBlockK, 1), order=(0,1,2))
  sB_layout    = tile_to_shape(swizzle_atom, (kBlockN, kBlockK, 1), order=(0,1,2))

and 16-bit-wide ``ldmatrix`` ops (``cute.nvgpu.warp.LdMatrix8x8x16bOp``)
on the S2R copies.

The 3D ``tile_to_shape`` form (with a degenerate pipeline-stage dim of 1)
mirrors the canonical CuTe DSL recipe from
``cutlass/examples/python/CuTeDSL/ampere/tensorop_gemm.py``: the full
``ComposedLayout`` is handed to ``allocate_tensor`` directly so the
allocator sees the swizzle composition. Because ``gA``/``gB`` here are
2D (single-iteration, no K-tiling), the partitioned smem tensors are
sliced with ``[None, None, None, 0]`` on the stage dim before each
``cute.copy`` to rank-match against the 2D global partitions.

For consistency with the other ports, the bf16 -> fp32 -> bf16 config is
exercised here. The epilogue mirrors ``swizzling.cu``'s ``TiledCopyO_R2S``
/ ``TiledCopyO_S2G`` pair: accumulator is narrowed to out_dtype in
registers, R2S'd into a swizzled smem buffer sO (Swizzle<3,3,3> o
(8 x min(64, BLK_N))), then drained to gmem with a contiguous-N TV
layout. Single (1, 1, 1) grid means no M/N residues to predicate; the
S2G copy is unconditional.

Run with ``python cutedsl_swizzling.py``.
"""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream


M = 128
N = 128
K = 64

MMA_INST_MNK = (16, 8, 16)
ATOM_LAYOUT_MNK = (2, 2, 1)
VAL_EXPAND_MNK = (1, 2, 2)
MMA_TILE_MNK = (
    ATOM_LAYOUT_MNK[0] * VAL_EXPAND_MNK[0] * MMA_INST_MNK[0],  # 32
    ATOM_LAYOUT_MNK[1] * VAL_EXPAND_MNK[1] * MMA_INST_MNK[1],  # 32
    ATOM_LAYOUT_MNK[2] * VAL_EXPAND_MNK[2] * MMA_INST_MNK[2],  # 32
)
NUM_THREADS = ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2] * 32  # 128


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def swizzling_kernel(
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
    sC_layout: cute.Layout,
    sO_layout: cute.ComposedLayout,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()

    gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))
    gB = cute.local_tile(mB, tiler=(N, K), coord=(0, 0))
    gC = cute.local_tile(mC, tiler=(M, N), coord=(0, 0))
    gO = cute.local_tile(mO, tiler=(M, N), coord=(0, 0))

    # ----- Smem allocation (swizzled atoms for A, B, O; non-swizzled for C) -----
    # ``sA_layout`` / ``sB_layout`` / ``sO_layout`` are 3D ``ComposedLayout``s
    # with a degenerate stage dim of 1 (mirrors tensorop_gemm.py). The
    # partitioned smem tensors are sliced with ``[None, None, None, 0]``
    # before each ``cute.copy`` to rank-match against the 2D global
    # partitions.
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.BFloat16, sB_layout, byte_alignment=16)
    sC = smem.allocate_tensor(cutlass.Float32, sC_layout, byte_alignment=16)
    sO = smem.allocate_tensor(out_dtype, sO_layout, byte_alignment=16)

    # ----- Phase 1: gmem -> smem via cp.async -----
    thr_g2s_a = g2s_tiled_copy_a.get_slice(tid)
    cute.copy(
        g2s_tiled_copy_a,
        thr_g2s_a.partition_S(gA),
        thr_g2s_a.partition_D(sA)[None, None, None, 0],
    )

    thr_g2s_b = g2s_tiled_copy_b.get_slice(tid)
    cute.copy(
        g2s_tiled_copy_b,
        thr_g2s_b.partition_S(gB),
        thr_g2s_b.partition_D(sB)[None, None, None, 0],
    )

    if cutlass.const_expr(not is_gemm):
        thr_g2s_c = g2s_tiled_copy_c.get_slice(tid)
        cute.copy(g2s_tiled_copy_c, thr_g2s_c.partition_S(gC), thr_g2s_c.partition_D(sC))

    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # ----- Fragments -----
    thr_mma = tiled_mma.get_slice(tid)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA)[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB)[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gC))

    # ----- Phase 2: smem -> rmem via ldmatrix-backed TiledCopies -----
    thr_s2r_a = s2r_tiled_copy_a.get_slice(tid)
    cute.copy(
        s2r_tiled_copy_a,
        thr_s2r_a.partition_S(sA)[None, None, None, 0],
        thr_s2r_a.retile(tCrA),
    )

    thr_s2r_b = s2r_tiled_copy_b.get_slice(tid)
    cute.copy(
        s2r_tiled_copy_b,
        thr_s2r_b.partition_S(sB)[None, None, None, 0],
        thr_s2r_b.retile(tCrB),
    )

    if cutlass.const_expr(is_gemm):
        tCrC.fill(0.0)
    else:
        thr_s2r_c = s2r_tiled_copy_c.get_slice(tid)
        cute.copy(
            s2r_tiled_copy_c,
            thr_s2r_c.partition_S(sC),
            thr_s2r_c.retile(tCrC),
        )

    # ----- Phase 3: compute -----
    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # ----- Phase 4: epilogue R2S -> S2G via swizzled smem buffer sO,
    # mirroring swizzling.cu's TiledCopyO_R2S / TiledCopyO_S2G pair.
    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))

    # R2S: register fragment -> swizzled smem buffer sO.
    thr_r2s_o = r2s_tiled_copy_o.get_slice(tid)
    tOrO_r2s = thr_r2s_o.retile(tCrO)
    tOsO_r2s = thr_r2s_o.partition_D(sO[None, None, 0])
    cute.copy(r2s_tiled_copy_o, tOrO_r2s, tOsO_r2s)

    cute.arch.sync_threads()

    # S2G: smem -> gmem with a TV layout that packs threads contiguously
    # along the N dim, matching swizzling.cu's TiledCopyO_S2G. Single
    # (1, 1, 1) grid means no M/N residues — no S2G predicate needed.
    thr_s2g_o = s2g_tiled_copy_o.get_slice(tid)
    tOsO_s2g = thr_s2g_o.partition_S(sO[None, None, 0])
    tOgO_s2g = thr_s2g_o.partition_D(gO)
    cute.copy(s2g_tiled_copy_o, tOsO_s2g, tOgO_s2g)


@cute.jit
def swizzling_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    stream: CUstream,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    # ----- Tiled MMA -----
    op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        MMA_INST_MNK,
    )
    tm = cute.make_tiled_mma(
        op,
        atom_layout_mnk=ATOM_LAYOUT_MNK,
        permutation_mnk=MMA_TILE_MNK,
    )

    # ----- Swizzled smem layouts (matches the C++ Swizzle<3,3,3>) -----
    # 3D-tiler form with a degenerate stage dim of 1 — the canonical CuTe DSL
    # recipe (see ``tensorop_gemm.py``). The kernel slices the stage off after
    # allocation so consumers see 2D smem tensors.
    swz = cute.make_swizzle(3, 3, 3)
    inner_AB = min(64, K)
    atom_AB = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_AB), stride=(inner_AB, 1)),
    )
    sA_layout = cute.tile_to_shape(atom_AB, (M, K, 1), order=(0, 1, 2))
    sB_layout = cute.tile_to_shape(atom_AB, (N, K, 1), order=(0, 1, 2))
    # The C preload buffer doesn't need swizzling for our flow; use a plain
    # row-major layout.
    sC_layout = cute.make_layout((M, N), stride=(N, 1))
    # sO mirrors swizzling.cu's SmemLayoutO: Swizzle<3,3,3> o (8 x min(64, BLK_N)).
    inner_O = min(64, N)
    atom_O = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_O), stride=(inner_O, 1)),
    )
    sO_layout = cute.tile_to_shape(atom_O, (M, N, 1), order=(0, 1, 2))

    # ----- G2S tiled copies (cp.async) -----
    g2s_op = cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
    )
    # The block has NUM_THREADS = 128 threads. We follow the C++ recipe:
    # ``kBlockK_Copy = min(64, kBlockK) / 8 = 8`` -> thread layout (16, 8) with
    # stride (8, 1), and value layout (1, 8) so each thread covers 8 elems
    # along the contiguous inner dim per TV iteration.
    block_k_copy = min(64, K) // 8
    block_n_copy = min(64, N) // 8
    tlAB_thr = cute.make_layout(
        (NUM_THREADS // block_k_copy, block_k_copy),
        stride=(block_k_copy, 1),
    )
    tlAB_val = cute.make_layout((1, 8))
    g2s_atom_a = cute.make_copy_atom(g2s_op, mA.element_type, num_bits_per_copy=128)
    g2s_atom_b = cute.make_copy_atom(g2s_op, mB.element_type, num_bits_per_copy=128)
    g2s_tiled_copy_a = cute.make_tiled_copy_tv(g2s_atom_a, tlAB_thr, tlAB_val)
    g2s_tiled_copy_b = cute.make_tiled_copy_tv(g2s_atom_b, tlAB_thr, tlAB_val)
    # For C: thread layout (16, 8) along (M, N), val (1, 8). Each thread issues
    # two 128-bit (= 4 fp32) cp.async ops to cover 8 fp32 along N.
    tlC_thr = cute.make_layout(
        (NUM_THREADS // block_n_copy, block_n_copy),
        stride=(block_n_copy, 1),
    )
    tlC_val = cute.make_layout((1, 8))
    g2s_atom_c = cute.make_copy_atom(g2s_op, mC.element_type, num_bits_per_copy=128)
    g2s_tiled_copy_c = cute.make_tiled_copy_tv(g2s_atom_c, tlC_thr, tlC_val)

    # ----- S2R tiled copies (ldmatrix for the 16-bit operands) -----
    ldm_op = cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4)
    s2r_atom_a = cute.make_copy_atom(ldm_op, mA.element_type)
    s2r_atom_b = cute.make_copy_atom(ldm_op, mB.element_type)
    s2r_tiled_copy_a = cute.make_tiled_copy_A(s2r_atom_a, tm)
    s2r_tiled_copy_b = cute.make_tiled_copy_B(s2r_atom_b, tm)
    # fp32 C uses universal copy for s2r (ldmatrix is 16-bit-only).
    universal = cute.nvgpu.CopyUniversalOp()
    s2r_atom_c = cute.make_copy_atom(universal, mC.element_type)
    s2r_tiled_copy_c = cute.make_tiled_copy_C(s2r_atom_c, tm)

    # ----- R2S + S2G copies for the smem-staged epilogue -----
    # R2S: MMA-derived TiledCopy so each thread's accumulator fragment
    # lands at its natural smem position under the swizzled sO layout.
    r2s_atom_o = cute.make_copy_atom(universal, mO.element_type)
    r2s_tiled_copy_o = cute.make_tiled_copy_C(r2s_atom_o, tm)

    # S2G: explicit TV layout — threads packed contiguously along N,
    # each thread emits 16 / sizeof(out) elements in a 128-bit store.
    elt_bytes_o = mO.element_type.width // 8
    n_chunk_o = min(64, N) // (16 // elt_bytes_o)
    tlO_thr = cute.make_layout(
        (NUM_THREADS // n_chunk_o, n_chunk_o),
        stride=(n_chunk_o, 1),
    )
    tlO_val = cute.make_layout((1, 16 // elt_bytes_o))
    s2g_atom_o = cute.make_copy_atom(universal, mO.element_type, num_bits_per_copy=128)
    s2g_tiled_copy_o = cute.make_tiled_copy_tv(s2g_atom_o, tlO_thr, tlO_val)

    swizzling_kernel(
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
    ).launch(grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream)


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
    divisibility = max(1, 16 // t.element_size())
    return (
        from_dlpack(t, assumed_align=16, enable_tvm_ffi=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=divisibility)
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    counters = {"succeed": 0, "failed": 0}
    torch.cuda.manual_seed_all(9527)

    a = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    print("Compiling CuTe DSL swizzling_gemm kernels ...")
    gemm_clear = cute.compile(
        swizzling_gemm,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_cute_tensor(o),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.BFloat16,
        True,
        options="--enable-tvm-ffi",
    )
    gemm_accum = cute.compile(
        swizzling_gemm,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_cute_tensor(o),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.BFloat16,
        False,
        options="--enable-tvm-ffi",
    )

    print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)

    c_out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    out_buf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    gemm_clear(a, b, c_out, out_buf)
    torch.cuda.synchronize()
    compare_matrix(out_buf, torch.matmul(a.float(), b.T.float()).bfloat16(), counters)

    c_inout = c.clone()
    out_buf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    gemm_accum(a, b, c_inout, out_buf)
    torch.cuda.synchronize()
    compare_matrix(
        out_buf,
        torch.addmm(c, a.float(), b.T.float()).bfloat16(),
        counters,
    )

    print(f" Summary: {counters['succeed']} Succeed, {counters['failed']} Failed ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
