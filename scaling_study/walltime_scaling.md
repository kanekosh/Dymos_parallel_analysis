# Wall time scaling

## NOTE on problem setup
OpenAeroStruct mesh: `num_y=21, num_x=5`.  
Dymos grid setting:  `tx = dm.Radau(num_segments=Nseg, order=3, solve_segments=False, compressed=True)`, where `Nseg=20` or `40`.
(Note that I used different number of segments from the memory scaling studies.)

All of the scaling studies below are the strong scaling study, i.e., the problem size remains the same for all numbers of processors.

## Scaling results on my workstation

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
| 16 - mem crash  |         |               |             |                  |

\
\
**Speedup plot (of strong scaling) for `num_segments = 20` on my workstation**
![Speed up (num_segments = 20) on workstation](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_Nseg20.jpg?raw=true)

\
**Speedup plot for `num_segments = 40` on my workstation**
![Speed up (num_segments = 40) on workstation](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_Nseg40.jpg?raw=true)

## Scaling results on a HPC cluster (Stampede2 SKX node)

I ran the same strong scaling study on a Stampede2 SKX node.
I only used single node, which has 48 processors and 192 GB RAM.

The scaling is much better on Stampede than on my workstation.
For the `num_segments=40` case with 40 processors, it ran out of memory during `prob.setup()`, therefore I could not measure the wall time.
\
\
**Speedup plot (of strong scaling) for `num_segments = 20` on Stampede**
![Speed up (num_segments = 20) on stampede](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_Nseg20_stampede.jpg?raw=true)

\
**Speedup plot for `num_segments = 40` on Stampede**
![Speed up (num_segments = 40) on stampede](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_Nseg40_stampede.jpg?raw=true)
