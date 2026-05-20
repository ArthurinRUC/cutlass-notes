"""Block-level GEMM with shared-memory cached inputs (128x128x64) in CuTe DSL.

CuTe DSL counterpart of ``block_copy.cu`` / ``block_copy.py``. Extends
``05-block-mma`` by routing the A / B / C operands through shared memory
and writing the accumulator back through smem too:

  gmem --(cp.async G2S)--> smem --(autovec S2R)--> rmem -> gemm
   -> rmem --(R2S)--> smem --(S2G)--> gmem

The G2S TiledCopy uses 128-bit ``cp.async`` instructions distributed across
the block's 256 threads (32x8 threads, each copying 8xbf16 = 16B per call
for A/B and 4xfp32 = 16B per call for C). The S2R TiledCopies are built
from the same ``TiledMma`` used in ``05-block-mma`` so they match the
register fragment layout out of the box.

The epilogue mirrors ``block_copy.cu``'s ``TiledCopyO_R2S`` / ``TiledCopyO_S2G``
pair:

  * **R2S**: MMA-derived TiledCopy so each thread's accumulator
    fragment lands at its natural smem position under the swizzled
    sO layout (Swizzle<3,3,3> o (8 x min(64, BLK_N))).
  * **S2G**: explicit TV layout — threads packed contiguously along
    N, each thread emits 16 / sizeof(out) elements in a 128-bit store.

The fp32 ``C`` smem buffer is only populated when ``is_gemm`` is False;
when ``is_gemm`` is True the accumulator is simply zeroed in registers.
Per-block smem footprint: sA (16 KB) + sB (16 KB) + sC (64 KB) + sO
(32 KB for bf16 output) ~= 128 KB, well under H100/H200's 228 KB
dynamic smem cap.

Single-tile (1, 1, 1) grid means no M/N residues to predicate; the S2G
copy is unconditional.

Run with ``python cutedsl_block_copy.py``.
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
def block_copy_kernel(
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
    sA_layout: cute.Layout,
    sB_layout: cute.Layout,
    sC_layout: cute.Layout,
    sO_layout: cute.ComposedLayout,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()

    # Global tile views (whole-problem, single block).
    gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))
    gB = cute.local_tile(mB, tiler=(N, K), coord=(0, 0))
    gC = cute.local_tile(mC, tiler=(M, N), coord=(0, 0))
    gO = cute.local_tile(mO, tiler=(M, N), coord=(0, 0))

    # ----- Shared memory allocation -----
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.BFloat16, sB_layout, byte_alignment=16)
    sC = smem.allocate_tensor(cutlass.Float32, sC_layout, byte_alignment=16)
    sO = smem.allocate_tensor(out_dtype, sO_layout, byte_alignment=16)

    # ----- Phase 1: gmem -> smem via cp.async -----
    thr_g2s_a = g2s_tiled_copy_a.get_slice(tid)
    tAgA_g2s = thr_g2s_a.partition_S(gA)
    tAsA_g2s = thr_g2s_a.partition_D(sA)
    cute.copy(g2s_tiled_copy_a, tAgA_g2s, tAsA_g2s)

    thr_g2s_b = g2s_tiled_copy_b.get_slice(tid)
    tBgB_g2s = thr_g2s_b.partition_S(gB)
    tBsB_g2s = thr_g2s_b.partition_D(sB)
    cute.copy(g2s_tiled_copy_b, tBgB_g2s, tBsB_g2s)

    if cutlass.const_expr(not is_gemm):
        thr_g2s_c = g2s_tiled_copy_c.get_slice(tid)
        tCgC_g2s = thr_g2s_c.partition_S(gC)
        tCsC_g2s = thr_g2s_c.partition_D(sC)
        cute.copy(g2s_tiled_copy_c, tCgC_g2s, tCsC_g2s)

    # Commit all outstanding cp.async issues into a group, then wait for it.
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # ----- Register fragments -----
    thr_mma = tiled_mma.get_slice(tid)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gC))

    # ----- Phase 2: smem -> rmem for A, B (and C if preloaded) -----
    thr_s2r_a = s2r_tiled_copy_a.get_slice(tid)
    tAsA_s2r = thr_s2r_a.partition_S(sA)
    tArA_s2r = thr_s2r_a.retile(tCrA)
    cute.copy(s2r_tiled_copy_a, tAsA_s2r, tArA_s2r)

    thr_s2r_b = s2r_tiled_copy_b.get_slice(tid)
    tBsB_s2r = thr_s2r_b.partition_S(sB)
    tBrB_s2r = thr_s2r_b.retile(tCrB)
    cute.copy(s2r_tiled_copy_b, tBsB_s2r, tBrB_s2r)

    if cutlass.const_expr(is_gemm):
        tCrC.fill(0.0)
    else:
        thr_s2r_c = s2r_tiled_copy_c.get_slice(tid)
        tCsC_s2r = thr_s2r_c.partition_S(sC)
        tCrC_s2r = thr_s2r_c.retile(tCrC)
        cute.copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r)

    # ----- Phase 3: compute -----
    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # ----- Phase 4: epilogue R2S -> S2G via swizzled smem buffer sO,
    # mirroring block_copy.cu's TiledCopyO_R2S / TiledCopyO_S2G pair.
    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))

    # R2S: register fragment -> swizzled smem buffer sO.
    thr_r2s_o = r2s_tiled_copy_o.get_slice(tid)
    tOrO_r2s = thr_r2s_o.retile(tCrO)
    tOsO_r2s = thr_r2s_o.partition_D(sO[None, None, 0])
    cute.copy(r2s_tiled_copy_o, tOrO_r2s, tOsO_r2s)

    cute.arch.sync_threads()

    # S2G: smem -> gmem with a TV layout that packs threads contiguously
    # along the N dim, matching block_copy.cu's TiledCopyO_S2G. Single
    # (1,1,1) grid means no M/N residues — no S2G predicate needed.
    thr_s2g_o = s2g_tiled_copy_o.get_slice(tid)
    tOsO_s2g = thr_s2g_o.partition_S(sO[None, None, 0])
    tOgO_s2g = thr_s2g_o.partition_D(gO)
    cute.copy(s2g_tiled_copy_o, tOsO_s2g, tOgO_s2g)


@cute.jit
def block_copy_gemm(
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

    # ----- Smem layouts -----
    sA_layout = cute.make_layout((M, K), stride=(K, 1))
    sB_layout = cute.make_layout((N, K), stride=(K, 1))
    sC_layout = cute.make_layout((M, N), stride=(N, 1))
    # sO mirrors block_copy.cu's SmemLayoutO: Swizzle<3,3,3> o (8 x min(64, BLK_N)).
    swz = cute.make_swizzle(3, 3, 3)
    inner_O = min(64, N)
    atom_O = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_O), stride=(inner_O, 1)),
    )
    sO_layout = cute.tile_to_shape(atom_O, (M, N, 1), order=(0, 1, 2))

    # ----- G2S (gmem -> smem) tiled copies using cp.async -----
    # 128-bit-per-thread cp.async: 16 bytes / sizeof(elt) elements per copy.
    g2s_op = cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
    )

    # AB copy is along (M/N, K). Threads packed contiguously along K, each
    # thread emits 16 / sizeof(elt) elements in a 128-bit cp.async. Mirrors
    # block_copy.cu's TiledCopyA_G2S / TiledCopyB_G2S derived from
    # kBlockK_Copy = min(64, kBlockK) / (16 / sizeof(elt)).
    elt_bytes_ab = mA.element_type.width // 8
    block_k_copy = min(64, K) // (16 // elt_bytes_ab)
    tlAB_thr = cute.make_layout(
        (NUM_THREADS // block_k_copy, block_k_copy),
        stride=(block_k_copy, 1),
    )
    tlAB_val = cute.make_layout((1, 16 // elt_bytes_ab))
    g2s_atom_a = cute.make_copy_atom(g2s_op, mA.element_type, num_bits_per_copy=128)
    g2s_atom_b = cute.make_copy_atom(g2s_op, mB.element_type, num_bits_per_copy=128)
    g2s_tiled_copy_a = cute.make_tiled_copy_tv(g2s_atom_a, tlAB_thr, tlAB_val)
    g2s_tiled_copy_b = cute.make_tiled_copy_tv(g2s_atom_b, tlAB_thr, tlAB_val)

    # C copy is along (M, N). Mirrors block_copy.cu's TiledCopyC_G2S derived
    # from kBlockN_Copy = min(64, kBlockN) / (16 / sizeof(elt)).
    elt_bytes_c = mC.element_type.width // 8
    block_n_copy = min(64, N) // (16 // elt_bytes_c)
    tlC_thr = cute.make_layout(
        (NUM_THREADS // block_n_copy, block_n_copy),
        stride=(block_n_copy, 1),
    )
    tlC_val = cute.make_layout((1, 16 // elt_bytes_c))
    g2s_atom_c = cute.make_copy_atom(g2s_op, mC.element_type, num_bits_per_copy=128)
    g2s_tiled_copy_c = cute.make_tiled_copy_tv(g2s_atom_c, tlC_thr, tlC_val)

    # ----- S2R (smem -> rmem) tiled copies, MMA-matching -----
    universal = cute.nvgpu.CopyUniversalOp()
    s2r_atom_a = cute.make_copy_atom(universal, mA.element_type)
    s2r_atom_b = cute.make_copy_atom(universal, mB.element_type)
    s2r_atom_c = cute.make_copy_atom(universal, mC.element_type)
    s2r_tiled_copy_a = cute.make_tiled_copy_A(s2r_atom_a, tm)
    s2r_tiled_copy_b = cute.make_tiled_copy_B(s2r_atom_b, tm)
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

    block_copy_kernel(
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
        grid=(1, 1, 1),
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
    # cp.async issues 128-bit-per-thread loads, which the IR verifier checks
    # for *pointer* alignment along the contiguous axis. ``assumed_align``
    # marks the base pointer aligned to 16 B; ``mark_compact_shape_dynamic``
    # tells CuTe the inner-dim *stride* is a multiple of (16 B / elem size)
    # elements, so partition offsets stay 16-byte aligned.
    elem_bytes = t.element_size()
    divisibility = max(1, 16 // elem_bytes)
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

    # Pre-compile two specializations (is_gemm=True/False) up front.
    a = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    print("Compiling CuTe DSL block_copy_gemm kernels ...")
    gemm_clear = cute.compile(
        block_copy_gemm,
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
        block_copy_gemm,
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

    # ----- Case 1: MM -----
    c_out = torch.empty(M, N, device="cuda", dtype=torch.float32)
    out_buf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    gemm_clear(a, b, c_out, out_buf)
    torch.cuda.synchronize()
    torch_output = torch.matmul(a.float(), b.T.float()).bfloat16()
    compare_matrix(out_buf, torch_output, counters)

    # ----- Case 2: MMA -----
    c_inout = c.clone()
    out_buf = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    gemm_accum(a, b, c_inout, out_buf)
    torch.cuda.synchronize()
    torch_output = torch.addmm(c, a.float(), b.T.float()).bfloat16()
    compare_matrix(out_buf, torch_output, counters)

    print(f" Summary: {counters['succeed']} Succeed, {counters['failed']} Failed ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
