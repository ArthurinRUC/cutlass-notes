# CUTLASS Notes

The CUTLASS notes series will begin with a minimal GEMM implementation, gradually expand to incorporate CuTe and various CUTLASS components, as well as features of new architectures, e.g. Hopper and Blackwell, ultimately achieving a high-performance fused GEMM operator.


## Usage

```bash
git clone https://github.com/ArthurinRUC/cutlass-notes.git

make update  # clone cutlass
```

## Run sample code

All example code in this GitHub repository can be compiled and run by simply executing the Python script. For example:

```bash
cd 01-minimal-gemm
python minimal_gemm.py
```

## CuTe DSL versions

Each example also ships a CuTe DSL Python port (`cutedsl_*.py`) alongside the original `.cu` / `.py` pair. The DSL port skips the C++ build step entirely — there is nothing to compile, just an `import` of the official CuTe DSL Python package.

The ports also enable [TVM-FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html), so each pre-compiled callable accepts raw `torch.Tensor` arguments directly and runs on `torch.cuda.current_stream()` without per-call dlpack conversion or explicit stream plumbing. The extra runtime requirements are:

```bash
pip install \
  "nvidia-cutlass-dsl>=4.3.5" \
  "cuda-python>=12.9" \
  "cuda-bindings>=12.9" \
  "apache-tvm-ffi>=0.1.8" \
  "torch_c_dlpack_ext>=0.1.5"
```

`cuda-bindings` provides `cuda.bindings.driver.CUstream` (used in the `@cute.jit` signature). `apache-tvm-ffi` + `torch_c_dlpack_ext` back the `--enable-tvm-ffi` codegen path and the `from_dlpack(..., enable_tvm_ffi=True)` runtime tensor binding.

Then run the DSL example the same way you'd run the original Python launcher:

```bash
cd 01-minimal-gemm
python cutedsl_minimal_gemm.py
```

The DSL ports use `@cute.jit` / `@cute.kernel` to express the kernel and `cute.compile(..., options="--enable-tvm-ffi")` to pre-compile each `is_gemm` / dtype specialization once per run. Compile templates use `make_cute_tensor(...)` (a `from_dlpack(..., enable_tvm_ffi=True)` wrapper) plus `make_fake_stream(use_tvm_ffi_env_stream=True)`; at runtime the compiled callable is invoked with bare `torch.Tensor` args — the DSL syncs to `torch.cuda.current_stream()` automatically (the environment-stream pattern). The host-side test harness mirrors the original — same seed, same `Success / Failed` summary — so a passing CUDA build and a passing DSL build print structurally identical output.

## Note list

| Notes                     | Summary                                                                                              | Links                                                                 |
|---------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **00-Intro**              | Brief introduction to CUTLASS | [intro](https://zhuanlan.zhihu.com/p/1937220431728845963) |
| **01-minimal-gemm**       | - Introduces CuTe fundamentals<br>- Implements 16x8x8 GEMM kernel using single MMA instruction from scratch<br>- Python kernel invocation, precision validation & performance benchmarking<br>- Profiling with Nsight Compute (ncu) | [minimal-gemm](https://zhuanlan.zhihu.com/p/1937517614084650073) |
| **02-mixed-precision-gemm** | - Implements mixed-precision GEMM supporting varying input/output/accumulation precisions<br>- Explores technical details for numerical precision conversion within kernels<br>- Demonstrates custom FP8 GEMM kernel implementation via PTX instructions (for CUTLASS-unsupported MMA ops) | [mixed-precision-gemm](https://zhuanlan.zhihu.com/p/1940158874255602181) |
| **03-tiled-mma** | - Introduces the key conceptual model of GEMM operator: Three-Level Tiling<br>- Details the implementation of Tiled MMA operations in CUTLASS CuTe<br>- Explains the usage and semantics of various parameters in the Tiled MMA API<br>- Extends the GEMM kernel from single instruction to single tile operation | [tiled-mma](https://zhuanlan.zhihu.com/p/1950555644814946318) |
| **04-tiled-copy** | - Explains the core principles of CuTe TiledCopy and its role in data movement between global and shared memory<br>- Describes the API parameters and semantics of TiledCopy<br>- Demonstrates how to implement data copying at the Tile level<br>- Introduces foundational knowledge of GPU global memory access characteristics | [tiled-copy](https://zhuanlan.zhihu.com/p/1968745447741972494) |
| **05-block-mma** | - Extends Tiled MMA to the Block level for larger-scale GEMM computations<br>- Explains how multiple Tiled MMA operations are combined within a thread block<br>- Describes the tiling and coordination of TiledCopy and TiledMMA at the Block level<br>- Illustrates the hierarchical dataflow from global memory to shared memory to registers for Block-level MMA | [block-mma](https://zhuanlan.zhihu.com/p/1970162570636816559) |
| **06-block-copy** | - Stages A / B / (C) through shared memory before the tensor-core MMA<br>- Introduces the gmem→smem→rmem dataflow, with explicit G2S / S2R / R2S / S2G TiledCopies<br>- Uses 128-bit `cp.async` for the gmem→smem path and `AutoVectorizingCopy` (`CopyUniversalOp` in CuTe DSL) for the rest<br>- Walks through dynamic shared-memory sizing and the lifecycle of A/B/C/O buffers within a single block | [block-copy](https://zhuanlan.zhihu.com/p/2004627053077627913) |


## License

This project is licensed under the MIT License - see the LICENSE file for details.