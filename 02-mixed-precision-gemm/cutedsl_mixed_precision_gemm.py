"""Mixed-precision 16x8x8 MMA implemented with CuTe DSL (Python).

This is the CuTe DSL counterpart of ``mixed_precision_gemm.cu`` /
``mixed_precision_gemm.py`` in this directory. It performs single-warp
SM80 bf16-input tensor-core MMAs with an fp32 accumulator, optionally
narrowing the result to bf16 in registers before writing back to gmem.

Compared to the CUDA original, two configurations are reproduced:
  1. ``out = bf16 @ bf16.T + fp32``  → output dtype fp32 (no narrowing)
  2. ``out = (bf16 @ bf16.T + fp32) -> bf16``  → output narrowed to bf16

The original CUDA file additionally exposes two ``e4m3 × e5m2`` mixed-fp8
configurations using a hand-rolled PTX MMA. The CuTe DSL public warp-level
API (``cute.nvgpu.warp.MmaFP8Op``) only supports a single shared fp8 dtype
for both A and B; a *mixed* fp8 MMA op is not exported. We therefore omit
those two cases here.

Run with ``python cutedsl_mixed_precision_gemm.py``. Output mirrors the
original script's ``Success / Failed`` summary.
"""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream


# The instruction shape baked into the kernel: SM80 ``mma.m16n8k8`` operates
# on a single 16x8x8 tile per warp, and we make this the whole problem.
M = 16
N = 8
K = 8


# -----------------------------------------------------------------------------
# Device kernel
# -----------------------------------------------------------------------------


@cute.kernel
def mixed_precision_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    tiled_mma: cute.TiledMma,
    out_dtype: cutlass.Constexpr,  # cutlass.Numeric subclass
    is_gemm: cutlass.Constexpr[bool],
    is_cvt_precision: cutlass.Constexpr[bool],
):
    """Single-tile, single-warp 16x8x8 bf16 MMA with fp32 accumulator.

    If ``is_cvt_precision`` is True the accumulator is narrowed to
    ``out_dtype`` in registers and written to ``mO`` instead of ``mC``.
    """
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

    if cutlass.const_expr(is_cvt_precision):
        # Narrow accumulator to out_dtype in registers, then store to mO.
        tCrO = cute.make_fragment_like(tCrC, out_dtype)
        tCrO.store(tCrC.load().to(out_dtype))
        tCgO = thr_mma.partition_C(gO)
        cute.autovec_copy(tCrO, tCgO)
    else:
        cute.autovec_copy(tCrC, tCgC)


@cute.jit
def mixed_precision_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    mO: cute.Tensor,
    stream: CUstream,
    out_dtype: cutlass.Constexpr,
    is_gemm: cutlass.Constexpr[bool],
    is_cvt_precision: cutlass.Constexpr[bool],
):
    """Host entry point: build the tiled MMA and launch the kernel."""
    # SM80 warp-level ``mma.m16n8k8.bf16.bf16.f32.f32`` — bf16 in, fp32 acc.
    op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        (M, N, K),
    )
    tiled_mma = cute.make_tiled_mma(op)

    # One atom == one warp == 32 threads.
    num_threads = tiled_mma.size

    mixed_precision_gemm_kernel(
        mA,
        mB,
        mC,
        mO,
        tiled_mma,
        out_dtype,
        is_gemm,
        is_cvt_precision,
    ).launch(
        grid=(1, 1, 1),
        block=(num_threads, 1, 1),
        stream=stream,
    )


# -----------------------------------------------------------------------------
# Host-side runner / test harness (mirrors ``mixed_precision_gemm.py``)
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


def run_case(
    out_torch_dtype: torch.dtype,
    gemm_clear,
    gemm_accum,
    counters: dict,
) -> None:
    """Run one (out_dtype) config using pre-compiled kernels."""
    m, n, k = M, N, K
    is_cvt = out_torch_dtype != torch.float32

    print(f" M={m}, N={n}, K={k} ".center(PRINT_LENGTH, "-"))

    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(m, n, device="cuda", dtype=torch.float32)

    # ----- Case 1: MM -----
    c_out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    # ``mO`` is unused when is_cvt is False but the kernel signature requires
    # a valid tensor; pass c_out as a dummy of the right shape.
    out_buf = torch.empty(m, n, device="cuda", dtype=out_torch_dtype) if is_cvt else c_out
    gemm_clear(a, b, c_out, out_buf)
    torch.cuda.synchronize()
    kernel_output = out_buf if is_cvt else c_out
    # Promote inputs to fp32 to keep the torch reference accurate.
    torch_output = torch.matmul(a.float(), b.T.float())
    if is_cvt:
        torch_output = torch_output.to(out_torch_dtype)
    compare_matrix(kernel_output, torch_output, counters)

    # ----- Case 2: MMA -----
    c_inout = c.clone()
    out_buf = torch.empty(m, n, device="cuda", dtype=out_torch_dtype) if is_cvt else c_inout
    gemm_accum(a, b, c_inout, out_buf)
    torch.cuda.synchronize()
    kernel_output = out_buf if is_cvt else c_inout
    torch_output = torch.addmm(c, a.float(), b.T.float())
    if is_cvt:
        torch_output = torch_output.to(out_torch_dtype)
    compare_matrix(kernel_output, torch_output, counters)


def compile_for(out_torch_dtype: torch.dtype, out_cutlass_dtype):
    """Pre-compile the (is_gemm=True, is_gemm=False) pair for one out dtype."""
    is_cvt = out_torch_dtype != torch.float32
    a = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    o = torch.empty(M, N, device="cuda", dtype=out_torch_dtype)
    mA = make_cute_tensor(a)
    mB = make_cute_tensor(b)
    mC = make_cute_tensor(c)
    mO = make_cute_tensor(o)

    gemm_clear = cute.compile(
        mixed_precision_gemm,
        mA,
        mB,
        mC,
        mO,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        out_cutlass_dtype,
        True,
        is_cvt,
        options="--enable-tvm-ffi",
    )
    gemm_accum = cute.compile(
        mixed_precision_gemm,
        mA,
        mB,
        mC,
        mO,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        out_cutlass_dtype,
        False,
        is_cvt,
        options="--enable-tvm-ffi",
    )
    return gemm_clear, gemm_accum


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    counters = {"succeed": 0, "failed": 0}

    print("Compiling CuTe DSL mixed_precision_gemm kernels ...")
    fp32_clear, fp32_accum = compile_for(torch.float32, cutlass.Float32)
    bf16_clear, bf16_accum = compile_for(torch.bfloat16, cutlass.BFloat16)

    print(" ---------------- fp32 = bf16 * bf16 + fp32 ---------------- ".center(PRINT_LENGTH))
    torch.cuda.manual_seed_all(9527)
    run_case(torch.float32, fp32_clear, fp32_accum, counters)

    print(" ---------------- bf16 = bf16 * bf16 + fp32 ---------------- ".center(PRINT_LENGTH))
    torch.cuda.manual_seed_all(9527)
    run_case(torch.bfloat16, bf16_clear, bf16_accum, counters)

    # The original CUDA file additionally tests two e4m3*e5m2 mixed-fp8 ops
    # (16x8x32). CuTe DSL's public warp-level MMA op
    # ``cute.nvgpu.warp.MmaFP8Op`` only supports a single fp8 dtype for both
    # A and B operands, so the *mixed* (e4m3 x e5m2) variant cannot be
    # reproduced here without dropping to inline PTX. Skipping those two
    # cases.

    print(f" Summary: {counters['succeed']} Succeed, {counters['failed']} Failed ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
