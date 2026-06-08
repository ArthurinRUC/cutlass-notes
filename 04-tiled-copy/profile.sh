# CSV output
# ncu --csv --log-file tiled_copy.csv  --metrics gpu__time_duration.sum --kernel-name "tiled_copy" python tiled_copy.py

# ncu-rep output
# C++ / reference impl (was the default; commented out):
# ncu -o ncu_prof_4 --import-source 1 --set full --kernel-name "tiled_copy" -f python tiled_copy.py
# CuTe DSL impl:
ncu -o ncu_prof_4 --import-source 1 --set full --kernel-name "regex:.*tiled_copy.*" -f python cutedsl_tiled_copy.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_4 python tiled_copy.py
