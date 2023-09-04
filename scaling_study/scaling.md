# Scaling study
MPI scaling study on `../scripts/run_optimization.py`.

### NOTE on problem setup
OpenAeroStruct mesh: `num_y=21, num_x=5`.  
Dymos grid setting: for memory usage study, `tx = dm.Radau(num_segments=40, order=3, solve_segments=False, compressed=True)`.
For wall time scaling, I measured timings with `num_segments=20` and `40`.

## Memory usage

The following table summarizes the maximum memory usage *per processor* for each part of the code: `setup`, `final_setup`, `run_model`, and `compute_totals`.  
The memory usage is shown in % of the total memory (64 GB).
These are manually monitored using the `top` command.

**Table: Memory usage per processor**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 9.7%    | 5.1%          | 5.1%        | 11.6%            |
| 2               | 9.1%    | 5.3%          | 5.3%        | 8.6%             |
| 4               | 8.9%    | 3.3%          | 3.3%        | 4.9%             |
| 8               | 8.7%    | 2.1%          | 2.1%        | 2.9%             |

The `setup` call requires nearly constant memory per processor, therefore the total memory usage of all processors is close to linear w.r.t. the number of processor.  
With `mpirun -n 16`, it runs out of the memory (64 GB) during `setup`.

## Wall time
The following tables summarizes the wall time scaling studies on 20 and 40 Dymos segments.
Timing is split into  `setup`, `final_setup`, `run_model`, and `compute_totals`.

**Table: Wall time scaling with `num_segments=20`, [s]**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 7.72    | 3.82          | 9.46        | 105.72           |
| 2               | 7.68    | 6.43          | 5.14        | 67.71            |
| 4               | 7.97    | 6.33          | 2.85        | 44.61            |
| 8               | 9.00    | 6.99          | 1.75        | 40.58            |
| 16              | 12.76   | 10.08         | 1.34        | 48.87            |

\
**Table: Wall time scaling with `num_segments=40`, [s]**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 26.08   | 7.94          | 18.83       | 595.14           |
| 2               | 24.91   | 13.27         | 10.41       | 383.60           |
| 4               | 26.41   | 12.65         | 5.82        | 251.87           |
| 8               | 29.73   | 15.04         | 3.69        | 233.61           |


Speedup plot (of strong scaling) for `num_segments = 20`:
![Speed up (num_segments = 20)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_Nseg20.jpg?raw=true)

\
Speedup plot for `num_segments = 40`:
![Speed up (num_segments = 40)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_Nseg40.jpg?raw=true)