# CUTLASS Notes

The CUTLASS notes series will begin with a minimal Minimal GEMM implementation, gradually expand to incorporate CuTe and various CUTLASS components, as well as features of new architectures like Hopper and Blackwell, ultimately achieving a high-performance fused GEMM operator.


## Usage

```bash
git clone https://github.com/ArthurinRUC/cutlass-notes.git
# clone cutlass
cd cutlass-notes
git submodule update --init --recursive
```

## Note list

| Notes                     | Summary                                                                                              | Links                                                                 |
|---------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **00-Intro**              | Brief introduction to CUTLASS | [intro](https://zhuanlan.zhihu.com/p/1937220431728845963)                                   |
| **01-minimal-gemm**       | <li>Introduces CuTe fundamentals</li><li>Implements 16x8x8 GEMM kernel using single MMA instruction from scratch</li><li>Python kernel invocation, precision validation & performance benchmarking</li><li>Profiling with Nsight Compute (ncu)</li> | [minimal-gemm](https://zhuanlan.zhihu.com/p/1937517614084650073)                      |
| **02-mixed-precision-gemm** | <li>Implements mixed-precision GEMM supporting varying input/output/accumulation precisions</li><li>Explores technical details for numerical precision conversion within kernels</li><li>Demonstrates custom FP8 GEMM kernel implementation via PTX instructions (for CUTLASS-unsupported MMA ops)</li> | [mixed-precision-gemm](https://zhuanlan.zhihu.com/p/1940158874255602181) |
| **03-tiled-mma** | *Coming soon*    | *Stay tuned*              |


## License

This project is licensed under the MIT License - see the LICENSE file for details.