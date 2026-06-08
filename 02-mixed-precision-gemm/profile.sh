# CSV output
# ncu --csv --log-file 02.csv  --metrics gpu__time_duration.sum --kernel-name "mixed_precision_gemm" python mixed_precision_gemm.py

# ncu-rep output
# C++ / reference impl (was the default; commented out):
# ncu -o ncu_prof_2 --import-source 1 --set full --kernel-name "mixed_precision_gemm" -f python mixed_precision_gemm.py
# CuTe DSL impl:
ncu -o ncu_prof_2 --import-source 1 --set full --kernel-name "regex:.*mixed_precision_gemm.*" -f python cutedsl_mixed_precision_gemm.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_2 python mixed_precision_gemm.py
