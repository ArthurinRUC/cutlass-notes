# CSV output
# ncu --csv --log-file dynamic_mma.csv  --metrics gpu__time_duration.sum --kernel-name "dynamic_mma" python dynamic_mma.py

# ncu-rep output
# C++ / reference impl (was the default; commented out):
# ncu -o ncu_prof_8 --import-source 1 --set full --kernel-name "dynamic_mma" -f python dynamic_mma.py
# CuTe DSL impl:
ncu -o ncu_prof_8 --import-source 1 --set full --kernel-name "regex:.*dynamic_mma.*" -f python cutedsl_dynamic_mma.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_8 python dynamic_mma.py
