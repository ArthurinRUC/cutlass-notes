"""Swizzled smem (128x128x64) in CuTe DSL.

CuTe DSL counterpart of ``swizzling.cu`` / ``swizzling.py``. Same flow
as ``06-block-copy``: gmem --(cp.async G2S)--> smem --(ldmatrix S2R)-->
rmem -> gemm -> rmem --(R2S)--> smem --(S2G)--> gmem. The new
ingredient is swizzled smem layouts on A and B, which kill the bank
conflicts that the strided ldmatrix pattern would otherwise generate:

  swizzle_atom = Swizzle<3,3,3> o (8 x min(64, kBlockK)):(min(64, kBlockK), 1)
  sA_layout    = tile_to_shape(swizzle_atom, (kBlockM, kBlockK))
  sB_layout    = tile_to_shape(swizzle_atom, (kBlockN, kBlockK))

and 16-bit-wide ``ldmatrix`` ops (``cute.nvgpu.warp.LdMatrix8x8x16bOp``)
on the S2R copies.

The full ``ComposedLayout`` is handed to ``allocate_tensor`` directly so
the allocator sees the swizzle composition. Since this kernel has no
pipelining, the smem layouts are plain 2D — ``(BLK_M, BLK_K)`` for A and
B, ``(BLK_M, BLK_N)`` for O — matching the 2D ``gA`` / ``gB`` / ``gO``
tiles 1:1.

Three dtype specs are exercised, matching ``swizzling.py``:

  * fp16 in, fp32 acc, fp16 out  (validated)
  * fp16 in, fp16 acc, fp16 out  (exercise only)
  * bf16 in, fp32 acc, bf16 out  (validated)

The epilogue mirrors ``swizzling.cu``'s ``TiledCopyO_R2S`` /
``TiledCopyO_S2G`` pair: accumulator is narrowed to out_dtype in
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
ATOM_LAYOUT_MNK = (2, 4, 1)
VAL_EXPAND_MNK = (1, 1, 2)
MMA_TILE_MNK = (
    ATOM_LAYOUT_MNK[0] * VAL_EXPAND_MNK[0] * MMA_INST_MNK[0],  # 32
    ATOM_LAYOUT_MNK[1] * VAL_EXPAND_MNK[1] * MMA_INST_MNK[1],  # 32
    ATOM_LAYOUT_MNK[2] * VAL_EXPAND_MNK[2] * MMA_INST_MNK[2],  # 32
)
NUM_THREADS = ATOM_LAYOUT_MNK[0] * ATOM_LAYOUT_MNK[1] * ATOM_LAYOUT_MNK[2] * 32  # 256


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
    sC_layout: cute.ComposedLayout,
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
    # ``sA_layout`` / ``sB_layout`` / ``sO_layout`` are 2D ``ComposedLayout``s
    # matching the 2D gA / gB / gO tiles directly. No pipeline-stage dim.
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=16)
    sB = smem.allocate_tensor(mB.element_type, sB_layout, byte_alignment=16)
    sC = smem.allocate_tensor(mC.element_type, sC_layout, byte_alignment=16)
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

    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # ----- Fragments -----
    thr_mma = tiled_mma.get_slice(tid)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
    tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gC))

    # ----- Phase 2: smem -> rmem via ldmatrix-backed TiledCopies -----
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
        # Load C from smem in its native dtype (mC.element_type), then convert
        # into the accumulator fragment. The intermediate is required when
        # the accumulator dtype differs from mC.element_type (e.g. fp16->fp32):
        # the DSL's ``cute.copy`` requires source/destination bit widths to
        # match, so the dtype change has to happen on a register-to-register
        # ``.to()`` step. When acc dtype == mC.element_type this is a no-op cast.
        thr_s2r_c = s2r_tiled_copy_c.get_slice(tid)
        tCrC_pre = cute.make_fragment_like(tCrC, mC.element_type)
        tCsC_s2r = thr_s2r_c.partition_S(sC)
        tCrC_s2r = thr_s2r_c.retile(tCrC_pre)
        cute.copy(s2r_tiled_copy_c, tCsC_s2r, tCrC_s2r)
        tCrC.store(tCrC_pre.load().to(tCrC.element_type))

    # ----- Phase 3: compute -----
    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # ----- Phase 4: epilogue R2S -> S2G via swizzled smem buffer sO,
    # mirroring swizzling.cu's TiledCopyO_R2S / TiledCopyO_S2G pair.
    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))

    # R2S: register fragment -> swizzled smem buffer sO.
    thr_r2s_o = r2s_tiled_copy_o.get_slice(tid)
    tOrO_r2s = thr_r2s_o.retile(tCrO)
    tOsO_r2s = thr_r2s_o.partition_D(sO)
    cute.copy(r2s_tiled_copy_o, tOrO_r2s, tOsO_r2s)

    cute.arch.sync_threads()

    # S2G: smem -> gmem with a TV layout that packs threads contiguously
    # along the N dim, matching swizzling.cu's TiledCopyO_S2G. Single
    # (1, 1, 1) grid means no M/N residues — no S2G predicate needed.
    thr_s2g_o = s2g_tiled_copy_o.get_slice(tid)
    tOsO_s2g = thr_s2g_o.partition_S(sO)
    tOgO_s2g = thr_s2g_o.partition_D(gO)
    cute.copy(s2g_tiled_copy_o, tOsO_s2g, tOgO_s2g)


@cute.jit
def swizzling_gemm(
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
    # MmaF16BF16Op handles both fp16 and bf16 input dtypes — the input/acc
    # dtype pair (e.g. fp16/fp32, fp16/fp16, bf16/fp32) selects the
    # underlying SM80_16x8x16 PTX op.
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

    # ----- Swizzled smem layouts (matches the C++ Swizzle<3,3,3>) -----
    # Plain 2D ComposedLayouts — no pipelining in this kernel, so no
    # degenerate stage dim is needed.
    swz = cute.make_swizzle(3, 3, 3)
    inner_AB = min(64, K)
    atom_AB = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_AB), stride=(inner_AB, 1)),
    )
    sA_layout = cute.tile_to_shape(atom_AB, (M, K), order=(0, 1))
    sB_layout = cute.tile_to_shape(atom_AB, (N, K), order=(0, 1))
    # sC mirrors swizzling.cu's SmemLayoutC: Swizzle<3,3,3> o (8 x min(64, N)).
    inner_C = min(64, N)
    atom_C = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_C), stride=(inner_C, 1)),
    )
    sC_layout = cute.tile_to_shape(atom_C, (M, N), order=(0, 1))
    # sO mirrors swizzling.cu's SmemLayoutO: Swizzle<3,3,3> o (8 x min(64, BLK_N)).
    inner_O = min(64, N)
    atom_O = cute.make_composed_layout(
        swz,
        0,
        cute.make_layout((8, inner_O), stride=(inner_O, 1)),
    )
    sO_layout = cute.tile_to_shape(atom_O, (M, N), order=(0, 1))

    # ----- G2S tiled copies (cp.async) -----
    g2s_op = cute.nvgpu.cpasync.CopyG2SOp(
        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
    )
    # AB copy is along (M/N, K). 128-bit cp.async loads 16B/thread =
    # 16 / sizeof(elt) elements along K per TV iteration.
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
    # C copy is along (M, N). 128-bit cp.async = 16/sizeof(C_elt) along N
    # per TV iteration.
    elt_bytes_c = mC.element_type.width // 8
    block_n_copy = min(64, N) // (16 // elt_bytes_c)
    tlC_thr = cute.make_layout(
        (NUM_THREADS // block_n_copy, block_n_copy),
        stride=(block_n_copy, 1),
    )
    tlC_val = cute.make_layout((1, 16 // elt_bytes_c))
    g2s_atom_c = cute.make_copy_atom(g2s_op, mC.element_type, num_bits_per_copy=128)
    g2s_tiled_copy_c = cute.make_tiled_copy_tv(g2s_atom_c, tlC_thr, tlC_val)

    # ----- S2R tiled copies (ldmatrix for the 16-bit operands) -----
    # A/B own 4 32-bit packets per thread (VAL_EXPAND_K=2), so the x4
    # ldmatrix variant fits exactly. C has no K val-expand, so each thread
    # only owns 2 32-bit packets — use the x2 ldmatrix variant for the C
    # s2r, mirroring swizzling.cu's Copy_S2R_op_C = SM75_U32x2_LDSM_N.
    ldm_op_ab = cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4)
    ldm_op_c = cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2)
    s2r_atom_a = cute.make_copy_atom(ldm_op_ab, mA.element_type)
    s2r_atom_b = cute.make_copy_atom(ldm_op_ab, mB.element_type)
    s2r_tiled_copy_a = cute.make_tiled_copy_A(s2r_atom_a, tm)
    s2r_tiled_copy_b = cute.make_tiled_copy_B(s2r_atom_b, tm)
    # C s2r: use ldmatrix if C is 16-bit, else universal copy (e.g. fp32).
    universal = cute.nvgpu.CopyUniversalOp()
    if cutlass.const_expr(mC.element_type.width == 16):
        s2r_atom_c = cute.make_copy_atom(ldm_op_c, mC.element_type)
    else:
        s2r_atom_c = cute.make_copy_atom(universal, mC.element_type)
    s2r_tiled_copy_c = cute.make_tiled_copy_C(s2r_atom_c, tm)

    # ----- R2S + S2G copies for the smem-staged epilogue -----
    # R2S: MMA-derived TiledCopy so each thread's accumulator fragment lands
    # at its natural smem position under the swizzled sO layout. Pick the op
    # the same way swizzling.cu does: SM90+ with a 16-bit output uses
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


def _compile_pair(a_template, b_template, c_template, o_template, acc_dtype, out_dtype):
    """Pre-compile (is_gemm=True, False) specializations for the given dtype combo."""
    g_clear = cute.compile(
        swizzling_gemm,
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
        swizzling_gemm,
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

    counters = {"succeed": 0, "failed": 0}
    torch.cuda.manual_seed_all(9527)

    # ----- Compile all three specs upfront. Template tensors are used only to
    # tag dtype + leading-dim alignment for the compiled artifact. -----

    print(" Compiling fp16 in / fp32 acc / fp16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M, K, device="cuda", dtype=torch.float16)
    b_t = torch.empty(N, K, device="cuda", dtype=torch.float16)
    c_t = torch.empty(M, N, device="cuda", dtype=torch.float16)
    o_t = torch.empty(M, N, device="cuda", dtype=torch.float16)
    fp16f32_clear, fp16f32_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        o_t,
        cutlass.Float32,
        cutlass.Float16,
    )

    print(" Compiling fp16 in / fp16 acc / fp16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M, K, device="cuda", dtype=torch.float16)
    b_t = torch.empty(N, K, device="cuda", dtype=torch.float16)
    c_t = torch.empty(M, N, device="cuda", dtype=torch.float16)
    o_t = torch.empty(M, N, device="cuda", dtype=torch.float16)
    fp16_clear, fp16_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        o_t,
        cutlass.Float16,
        cutlass.Float16,
    )

    print(" Compiling bf16 in / fp32 acc / bf16 out ... ".center(PRINT_LENGTH, "-"))
    a_t = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    b_t = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    c_t = torch.empty(M, N, device="cuda", dtype=torch.float32)
    o_t = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    bf16_clear, bf16_accum = _compile_pair(
        a_t,
        b_t,
        c_t,
        o_t,
        cutlass.Float32,
        cutlass.BFloat16,
    )

    print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

    # ----- Spec 1: fp16 = fp16 * fp16 + fp32 (validated) -----
    print(" fp16 = fp32_acc(fp16 * fp16) + fp16 ".center(PRINT_LENGTH, "="))
    torch.cuda.manual_seed_all(9527)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # Case 1: MM (fp16 = fp16 * fp16)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    fp16f32_clear(a, b, c.clone().half(), out)
    torch.cuda.synchronize()
    # For fp16 input, torch.matmul uses fp32 as the accumulator precision
    compare_matrix(out, torch.matmul(a.float(), b.T.float()).half(), counters)

    # Case 2: MMA (fp16 = fp16 * fp16 + fp16)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    fp16f32_accum(a, b, c.clone().half(), out)
    torch.cuda.synchronize()
    compare_matrix(out, torch.addmm(c, a.float(), b.T.float()).half(), counters)

    # ----- Spec 2: fp16 = fp16 * fp16 + fp16 (exercise only, not validated) -----
    # Matches swizzling.py: the fp16-accumulator variant is launched but not
    # compared — fp16 accumulation drops too many bits to match torch's
    # fp32-accumulated reference.
    print(" fp16 = fp16 * fp16 + fp16 (exercise only) ".center(PRINT_LENGTH, "="))
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    fp16_clear(a, b, c.clone().half(), out)
    out = torch.empty(M, N, device="cuda", dtype=torch.float16)
    fp16_accum(a, b, c.clone().half(), out)
    torch.cuda.synchronize()

    # ----- Spec 3: bf16 = bf16 * bf16 + fp32 (validated) -----
    print(" bf16 = fp32_acc(bf16 * bf16) + fp32 ".center(PRINT_LENGTH, "="))
    torch.cuda.manual_seed_all(9527)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(M, N, device="cuda", dtype=torch.float32)

    # Case 1: MM (bf16 = bf16 * bf16)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    bf16_clear(a, b, c.clone(), out)
    torch.cuda.synchronize()
    compare_matrix(out, torch.matmul(a.float(), b.T.float()).bfloat16(), counters)

    # Case 2: MMA (bf16 = bf16 * bf16 + fp32)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    bf16_accum(a, b, c.clone(), out)
    torch.cuda.synchronize()
    compare_matrix(out, torch.addmm(c, a.float(), b.T.float()).bfloat16(), counters)

    print(f" Summary: {counters['succeed']} Succeed, {counters['failed']} Failed ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
