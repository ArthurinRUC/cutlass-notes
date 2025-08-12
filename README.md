# CUTLASS Notes

The CUTLASS notes series will begin with a minimal Minimal GEMM implementation, gradually expand to incorporate CuTe and various CUTLASS components, as well as features of new architectures like Hopper and Blackwell, ultimately achieving a high-performance fused GEMM operator.


## Usage

```bash
git clone https://github.com/ArthurinRUC/cutlass-notes.git
# clone cutlass
git submodule update --init --recursive
```

## Note list

| Notes                     | Summary                                                                                              | Links                                                                 |
|---------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **00-Intro**              | Brief introduction to CUTLASS | [intro](https://zhuanlan.zhihu.com/p/1937220431728845963)                                   |
| **01-minimal-gemm**       | <li>Introduces CuTe fundamentals</li><li>Implements 16x8x8 GEMM kernel using single MMA instruction from scratch</li><li>Python kernel invocation, precision validation & performance benchmarking</li><li>Profiling with Nsight Compute (ncu)</li> | [minimal-gemm]()                      |
| **02-mixed-precision-gemm** | *Coming soon*    | *Stay tuned*              |


## License

This project is licensed under the MIT License - see the LICENSE file for details.