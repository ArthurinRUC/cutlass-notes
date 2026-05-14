"""Tiled MMA + Tiled Copy (32x32x16) implemented with CuTe DSL (Python).

CuTe DSL counterpart of ``tiled_copy.cu`` / ``tiled_copy.py``. Identical
problem shape and TiledMMA as ``03-tiled-mma`` (16x8x8 bf16 atom expanded
to a 32x32x16 thread-block tile across 256 threads), but every
global-memory <-> register copy is now driven by an explicit ``TiledCopy``
that matches the MMA's thread/value layout:

  * ``TiledCopyA = make_tiled_copy_A(CopyUniversalOp, tiled_mma)``
  * ``TiledCopyB = make_tiled_copy_B(CopyUniversalOp, tiled_mma)``
  * ``TiledCopyC = make_tiled_copy_C(CopyUniversalOp, tiled_mma)``  (also used for O)

Inside the kernel we use ``thr_copy.partition_S(gX)`` to view the global
tile in the copy's source TV layout, and ``thr_copy.retile(tCrX)`` to view
the MMA register fragment in the copy's destination TV layout (this is
the Python equivalent of the C++ ``retile_S/retile_D`` pair).

Output is narrowed bf16 from an fp32 accumulator.

Run with ``python cutedsl_tiled_copy.py``.
"""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream


M = 32
N = 32
K = 16

MMA_INST_MNK = (16, 8, 8)
ATOM_LAYOUT_MNK = (2, 4, 1)
VAL_EXPAND_MNK = (1, 1, 2)
PERMUTATION_MNK = (
    ATOM_LAYOUT_MNK[0] * VAL_EXPAND_MNK[0] * MMA_INST_MNK[0],  # 32
    ATOM_LAYOUT_MNK[1] * VAL_EXPAND_MNK[1] * MMA_INST_MNK[1],  # 32
    ATOM_LAYOUT_MNK[2] * VAL_EXPAND_MNK[2] * MMA_INST_MNK[2],  # 16
)
assert PERMUTATION_MNK == (M, N, K)


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def tiled_copy_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    tiled_mma: cute.TiledMma,
    tiled_copy_a: cute.TiledCopy,
    tiled_copy_b: cute.TiledCopy,
    tiled_copy_c: cute.TiledCopy,
    tiled_copy_o: cute.TiledCopy,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()

    gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))
    gB = cute.local_tile(mB, tiler=(N, K), coord=(0, 0))
    gC = cute.local_tile(mC, tiler=(M, N), coord=(0, 0))
    gO = cute.local_tile(mO, tiler=(M, N), coord=(0, 0))

    # MMA register fragments.
    thr_mma = tiled_mma.get_slice(tid)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(gA))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(gB))
    tCrC = tiled_mma.make_fragment_C(tCgC)

    # gmem -> rmem via explicit TiledCopy for A, B.
    thr_copy_a = tiled_copy_a.get_slice(tid)
    tAgA = thr_copy_a.partition_S(gA)
    tArA = thr_copy_a.retile(tCrA)

    thr_copy_b = tiled_copy_b.get_slice(tid)
    tBgB = thr_copy_b.partition_S(gB)
    tBrB = thr_copy_b.retile(tCrB)

    cute.copy(tiled_copy_a, tAgA, tArA)
    cute.copy(tiled_copy_b, tBgB, tBrB)

    if cutlass.const_expr(is_gemm):
        tCrC.fill(0.0)
    else:
        # gmem -> rmem for the C preload, also via TiledCopy.
        thr_copy_c = tiled_copy_c.get_slice(tid)
        tCgC_g2r = thr_copy_c.partition_S(gC)
        tCrC_g2r = thr_copy_c.retile(tCrC)
        cute.copy(tiled_copy_c, tCgC_g2r, tCrC_g2r)

    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # Narrow accumulator and write back to mO via the O-side TiledCopy.
    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))

    thr_copy_o = tiled_copy_o.get_slice(tid)
    tOrC_r2g = thr_copy_o.retile(tCrO)
    tOgO_r2g = thr_copy_o.partition_D(gO)
    cute.copy(tiled_copy_o, tOrC_r2g, tOgO_r2g)


@cute.jit
def tiled_copy_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    stream: CUstream,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        MMA_INST_MNK,
    )
    tm = cute.make_tiled_mma(
        op,
        atom_layout_mnk=ATOM_LAYOUT_MNK,
        permutation_mnk=PERMUTATION_MNK,
    )

    # ``CopyUniversalOp`` is CuTe DSL's equivalent of the C++
    # ``AutoVectorizingCopy``: the toolchain picks a vector width.
    universal = cute.nvgpu.CopyUniversalOp()
    atom_a = cute.make_copy_atom(universal, mA.element_type)
    atom_b = cute.make_copy_atom(universal, mB.element_type)
    atom_c = cute.make_copy_atom(universal, mC.element_type)
    atom_o = cute.make_copy_atom(universal, mO.element_type)

    tiled_copy_a = cute.make_tiled_copy_A(atom_a, tm)
    tiled_copy_b = cute.make_tiled_copy_B(atom_b, tm)
    tiled_copy_c = cute.make_tiled_copy_C(atom_c, tm)
    tiled_copy_o = cute.make_tiled_copy_C(atom_o, tm)

    num_threads = tm.size  # 256

    tiled_copy_kernel(
        mA,
        mB,
        mC,
        mO,
        tm,
        tiled_copy_a,
        tiled_copy_b,
        tiled_copy_c,
        tiled_copy_o,
        out_dtype,
        is_gemm,
    ).launch(
        grid=(1, 1, 1),
        block=(num_threads, 1, 1),
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

    print("Compiling CuTe DSL tiled_copy_gemm kernels ...")
    gemm_clear = cute.compile(
        tiled_copy_gemm,
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
        tiled_copy_gemm,
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
