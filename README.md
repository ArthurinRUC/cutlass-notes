# CUTLASS Notes

The CUTLASS notes series will begin with a minimal GEMM implementation, gradually expand to incorporate CuTe and various CUTLASS components, as well as features of new architectures, e.g. Hopper and Blackwell, ultimately achieving a high-performance fused GEMM operator.


## Usage

```bash
git clone https://github.com/ArthurinRUC/cutlass-notes.git
# clone cutlass
cd cutlass-notes
git submodule update --init --recursive
```

## Run sample code

All example code in this GitHub repository can be compiled and run by simply executing the Python script. For example:

```bash
cd 01-minimal-gemm
python minimal_gemm.py
```

## Note list

| Notes                     | Summary                                                                                              | Links                                                                 |
|---------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| **00-Intro**              | Brief introduction to CUTLASS | [intro](https://zhuanlan.zhihu.com/p/1937220431728845963)                                   |
| **01-minimal-gemm**       | <li>Introduces CuTe fundamentals</li><li>Implements 16x8x8 GEMM kernel using single MMA instruction from scratch</li><li>Python kernel invocation, precision validation & performance benchmarking</li><li>Profiling with Nsight Compute (ncu)</li> | [minimal-gemm](https://zhuanlan.zhihu.com/p/1937517614084650073)                      |
| **02-mixed-precision-gemm** | <li>Implements mixed-precision GEMM supporting varying input/output/accumulation precisions</li><li>Explores technical details for numerical precision conversion within kernels</li><li>Demonstrates custom FP8 GEMM kernel implementation via PTX instructions (for CUTLASS-unsupported MMA ops)</li> | [mixed-precision-gemm](https://zhuanlan.zhihu.com/p/1940158874255602181) |
| **03-tiled-mma** | <li>Introduces the key conceptual model of GEMM operator: Three-Level Tiling</li><li>Details the implementation of Tiled MMA operations in CUTLASS CuTe</li><li>Explains the usage and semantics of various parameters in the Tiled MMA API</li><li>Extends the GEMM kernel from single instruction to single tile operation</li>   | [tiled-mma](https://zhuanlan.zhihu.com/p/1950555644814946318)     |
| **04-tiled-copy** | *Coming soon*    | *Stay tuned*              |


## License

This project is licensed under the MIT License - see the LICENSE file for details.