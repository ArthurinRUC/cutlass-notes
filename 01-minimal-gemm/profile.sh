# CSV output
# ncu --csv --log-file 01.csv  --metrics gpu__time_duration.sum --kernel-name "minimal_gemm" python minimal_gemm.py

# ncu-rep output
# C++ / reference impl (was the default; commented out):
# ncu -o ncu_prof_1 --import-source 1 --set full --kernel-name "minimal_gemm" -f python minimal_gemm.py
# CuTe DSL impl:
ncu -o ncu_prof_1 --import-source 1 --set full --kernel-name "regex:.*minimal_gemm.*" -f python cutedsl_minimal_gemm.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_1 python minimal_gemm.py
