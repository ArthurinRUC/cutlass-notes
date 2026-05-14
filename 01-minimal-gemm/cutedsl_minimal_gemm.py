"""Minimal 16x8x8 fp16 MMA implemented with CuTe DSL (Python).

This is the CuTe DSL counterpart of ``minimal_gemm.cu`` / ``minimal_gemm.py``
in this directory. It performs a single warp's SM80 ``mma.m16n8k8`` operation
on fp16 inputs with fp16 accumulator and reproduces the exact two test cases
from ``minimal_gemm.py``:

  * Case 1 (``is_gemm=True``):  C = A @ B.T          (accumulator cleared)
  * Case 2 (``is_gemm=False``): C = A @ B.T + C_in   (accumulator preloaded)

Just run ``python cutedsl_minimal_gemm.py``. The output mirrors that of
``minimal_gemm.py``: each case prints ``Max diff``, ``Mean diff``, and
relative error against a torch reference, followed by a success summary.

Requires the ``nvidia-cutlass-dsl`` Python package (CUTLASS >= 4.x) and a
CUDA-capable GPU of compute capability >= sm_80.
"""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream


# The instruction shape baked into the kernel. SM80 ``mma.m16n8k8.f16.f16.f16``
# operates on a single 16x8x8 tile per warp; we make this the whole problem.
M = 16
N = 8
K = 8


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def minimal_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    is_gemm: cutlass.Constexpr[bool],
):
    """Single-tile, single-warp 16x8x8 fp16 MMA.

    Mirrors the C++ kernel in ``minimal_gemm.cu``:
        - Partition global tensors by the tiled MMA's thread slice.
        - Allocate matching register fragments.
        - Copy gmem -> rmem for A, B (and C when accumulating).
        - Invoke the tensor-core gemm.
        - Copy the result back rmem -> gmem.
    """
    tid, _, _ = cute.arch.thread_idx()

    # The whole problem fits in a single (M, N, K) tile, so we just take the
    # zero-th tile at coord (0, 0). ``local_tile`` keeps the static tile shape
    # for partition.
    gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))  # (M, K)
    gB = cute.local_tile(mB, tiler=(N, K), coord=(0, 0))  # (N, K)
    gC = cute.local_tile(mC, tiler=(M, N), coord=(0, 0))  # (M, N)

    thr_mma = tiled_mma.get_slice(tid)
    tCgA = thr_mma.partition_A(gA)  # (MMA, MMA_M, MMA_K)
    tCgB = thr_mma.partition_B(gB)  # (MMA, MMA_N, MMA_K)
    tCgC = thr_mma.partition_C(gC)  # (MMA, MMA_M, MMA_N)

    tCrA = tiled_mma.make_fragment_A(tCgA)
    tCrB = tiled_mma.make_fragment_B(tCgB)
    tCrC = tiled_mma.make_fragment_C(tCgC)

    cute.autovec_copy(tCgA, tCrA)
    cute.autovec_copy(tCgB, tCrB)

    if cutlass.const_expr(is_gemm):
        # Zero accumulators (pure C = A @ B.T)
        tCrC.fill(0.0)
    else:
        # Preload existing C (so the kernel produces C += A @ B.T)
        cute.autovec_copy(tCgC, tCrC)

    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    cute.autovec_copy(tCrC, tCgC)


@cute.jit
def minimal_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: CUstream,
    is_gemm: cutlass.Constexpr[bool],
):
    """Host entry point: build the tiled MMA and launch the kernel."""
    # SM80 warp-level ``mma.m16n8k8.f16.f16.f16.f16`` — fp16 in, fp16 acc.
    op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16,
        cutlass.Float16,
        (M, N, K),
    )
    tiled_mma = cute.make_tiled_mma(op)

    # One atom == one warp == 32 threads.
    num_threads = tiled_mma.size

    minimal_gemm_kernel(mA, mB, mC, tiled_mma, is_gemm).launch(
        grid=(1, 1, 1),
        block=(num_threads, 1, 1),
        stream=stream,
    )


# -----------------------------------------------------------------------------
# Host-side runner / test harness (mirrors ``minimal_gemm.py``)
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
    """Convert a torch tensor to a CuTe DSL tensor for kernel input.

    Marks the last (contiguous) dim as the leading dim so CuTe DSL knows the
    strides at compile time without baking the full static shape in.
    """
    return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    Ms = [M]
    Ns = [N]
    Ks = [K]
    exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    torch.cuda.manual_seed_all(9527)
    counters = {"succeed": 0, "failed": 0}

    # Pre-compile two specializations of the kernel: one with the accumulator
    # cleared (``is_gemm=True``, plain MM) and one with the accumulator
    # preloaded from C (``is_gemm=False``, MMA / addmm). Because ``is_gemm``
    # is a Constexpr in the kernel signature, each value of it requires its
    # own compiled artifact; the compiled callable no longer takes that arg.
    # ``cute.compile`` only inspects dtype + layout tags on the tensors, so
    # any tensor of the canonical shape works; we reuse the same names for
    # the actual test data inside the loop below.
    a = torch.empty(M, K, device="cuda", dtype=torch.half)
    b = torch.empty(N, K, device="cuda", dtype=torch.half)
    c = torch.empty(M, N, device="cuda", dtype=torch.half)

    print("Compiling CuTe DSL minimal_gemm kernels ...")
    gemm_clear = cute.compile(
        minimal_gemm,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        True,
        options="--enable-tvm-ffi",
    )
    gemm_accum = cute.compile(
        minimal_gemm,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        False,
        options="--enable-tvm-ffi",
    )

    for exp in exps:
        m, n, k = exp
        print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))

        a = torch.randn(m, k, device="cuda", dtype=torch.half)
        b = torch.randn(n, k, device="cuda", dtype=torch.half)
        c = torch.randn(m, n, device="cuda", dtype=torch.half)

        # ----- Case 1: MM (C = A @ B.T) -----
        c_out = torch.empty(m, n, device="cuda", dtype=torch.half)
        gemm_clear(a, b, c_out)
        torch.cuda.synchronize()
        torch_output = torch.matmul(a, b.T)
        compare_matrix(c_out, torch_output, counters)

        # ----- Case 2: MMA (C = A @ B.T + C_in) -----
        c_inout = c.clone()
        gemm_accum(a, b, c_inout)
        torch.cuda.synchronize()
        # Mathematically equivalent to torch.matmul(a, b.T) + c, but the kernel
        # accumulates in fp16 inside the MMA op so torch.addmm is closer to the
        # bit pattern produced by the kernel (slight FP-order differences may
        # remain).
        torch_output = torch.addmm(c, a, b.T)
        compare_matrix(c_inout, torch_output, counters)

    print(f" Summary: {counters['succeed']} Succeed, {counters['failed']} Failed ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
