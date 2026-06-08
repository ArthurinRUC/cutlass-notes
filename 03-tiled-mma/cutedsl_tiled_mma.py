"""Tiled MMA (32x32x16) implemented with CuTe DSL (Python).

CuTe DSL counterpart of ``tiled_mma.cu`` / ``tiled_mma.py``. Builds a
tiled MMA on top of the SM80 ``mma.m16n8k8.bf16.bf16.f32.f32`` atom by
expanding it across threads and values:

  * thread layout (atom_layout_mnk) = (2, 4, 1)  → 2 atoms along M, 4 along N
  * value expansion along K  = 2                 → 2 K instructions per thread

This yields a (32, 32, 16) thread-block tile that exactly matches the
problem size, and ``size(TiledMMA) = 2 * 4 * 1 * 32 = 256`` threads (= 8
warps) in a single block.

Output is narrowed bf16; the fp32 accumulator is converted in registers
before the final write to gmem (same as case 2 of 02).

Run with ``python cutedsl_tiled_mma.py``.
"""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream


# Problem and tile sizes — chosen so the whole problem is exactly one block.
M = 32
N = 32
K = 16

# Atom instruction shape (SM80 bf16 16x8x8).
MMA_INST_MNK = (16, 8, 8)
# Thread-level replication of the atom in M/N/K.
ATOM_LAYOUT_MNK = (2, 4, 1)
# Value-level expansion of the atom in M/N/K (only K gets a factor of 2 here).
VAL_EXPAND_MNK = (1, 1, 2)
# Resulting per-block tile.
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
def tiled_mma_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    tiled_mma: cute.TiledMma,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()

    gA = cute.local_tile(mA, tiler=(M, K), coord=(0, 0))
    gB = cute.local_tile(mB, tiler=(N, K), coord=(0, 0))
    gC = cute.local_tile(mC, tiler=(M, N), coord=(0, 0))
    gO = cute.local_tile(mO, tiler=(M, N), coord=(0, 0))

    thr_mma = tiled_mma.get_slice(tid)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = tiled_mma.make_fragment_A(tCgA)
    tCrB = tiled_mma.make_fragment_B(tCgB)
    tCrC = tiled_mma.make_fragment_C(tCgC)

    cute.autovec_copy(tCgA, tCrA)
    cute.autovec_copy(tCgB, tCrB)

    if cutlass.const_expr(is_gemm):
        tCrC.fill(0.0)
    else:
        cute.autovec_copy(tCgC, tCrC)

    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # Narrow fp32 accumulator to ``out_dtype`` and write to mO.
    tCrO = cute.make_fragment_like(tCrC, out_dtype)
    tCrO.store(tCrC.load().to(out_dtype))
    tCgO = thr_mma.partition_C(gO)
    cute.autovec_copy(tCrO, tCgO)


@cute.jit
def tiled_mma(
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
    num_threads = tm.size  # 256

    tiled_mma_kernel(mA, mB, mC, mO, tm, out_dtype, is_gemm).launch(
        grid=(1, 1, 1),
        block=(num_threads, 1, 1),
        stream=stream,
    )


# -----------------------------------------------------------------------------
# Host-side runner / test harness (mirrors ``tiled_mma.py``)
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

    # Pre-compile two specializations (is_gemm=True/False) up front. The
    # tensors only need the right dtype/shape for tracing; we overwrite
    # them with real test data below.
    a = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    print("Compiling CuTe DSL tiled_mma kernels ...")
    gemm_clear = cute.compile(
        tiled_mma,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_cute_tensor(o),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.BFloat16,
        True,
        options="--enable-tvm-ffi --generate-line-info",
    )
    gemm_accum = cute.compile(
        tiled_mma,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_cute_tensor(o),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        cutlass.BFloat16,
        False,
        options="--enable-tvm-ffi --generate-line-info",
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
