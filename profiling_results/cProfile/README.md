Using cProfile with MPI.  
Run `mpirun -n XXX run_optimization_cProfile.py`.  
Visualize by `snakeviz cpu_0.prof`.

`run_optimization_cProfile.py` does the same as `scripts/run_optimization.py`, but the script is slightly modified for cProfiling.