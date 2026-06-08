# CSV output
# ncu --csv --log-file tma_load_store.csv  --metrics gpu__time_duration.sum --kernel-name "tma_load_store" python tma_load_store.py

# ncu-rep output
# C++ / reference impl (was the default; commented out):
# ncu -o ncu_prof_11 --import-source 1 --set full --kernel-name "tma_load_store" -f python tma_load_store.py
# CuTe DSL impl:
ncu -o ncu_prof_11 --import-source 1 --set full --kernel-name "regex:.*tma_load_store.*" -f python cutedsl_tma_load_store.py

# nsys-rep output
# nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas,cublas-verbose,mpi,ucx,oshmem,python-gil,syscall --backtrace=dwarf --output=nsys_prof_11 python tma_load_store.py
