# CSV output
# ncu --csv --log-file 03.csv  --metrics gpu__time_duration.sum --kernel-name "tiled_mma" python tiled_mma.py

# ncu-rep output
# C++ / reference impl (was the default; commented out):
# ncu -o ncu_prof_3 --import-source 1 --set full --kernel-name "tiled_mma" -f python tiled_mma.py
# CuTe DSL impl:
ncu -o ncu_prof_3 --import-source 1 --set full --kernel-name "regex:.*tiled_mma.*" -f python cutedsl_tiled_mma.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_3 python tiled_mma.py
