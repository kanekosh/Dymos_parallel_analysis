# Comparison of OAS-Dymos coupling approaches

**Monolithic**: add an OAS point (`AeroStructPoint`) at each node.
This results in one big OpenMDAO problem that includes everything, and the top-level OpenMDAO problem has access to all OAS intermediate variables.  

**Nested**: add an OAS points as a subproblem (wrapped by `AeroStructPoint_SubProblemWrapper` in `oas_subproblem.py`) at each node.
From the top-level OpenMDAO problem, each OAS analysis is treated as a black-box that returns CL and CD given the flight conditions and wing design.
The OAS intermediate variables are only stored at the subproblem level, and not available to the top-level OpenMDAO problem.

## How to switch 
Set `use_subproblem` to `True` in `aero_oas.OASAnalyses` to enable the nested approach.
Set it to False to do monolithic approach.

## Memory usage

The memory issue was not resolved by the nested approach (which makes sense).
The following table summarizes the memory usage *per processor* for `num_segments = 40`.
The `setup` call required a bit more memory than the monolithic approach, but other calls required a little less for parallel cases.

**Table: Memory usage per processor: nested approach**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 11.6%   | 7.6%          | 7.6%        | 14.0%            |
| 2               | 10.1    | 4.2%          | 4.2%        | 7.3%             |
| 4               | 9.3%    | 2.3%          | 2.3%        | 3.8%             |
| 8               | 8.9%    | 1.3%          | 1.3%        | 2.1%             |

## Wall time comparison

NOTE: `compute_totals()` does not exploits the total Jacobian coloring, therefore it is not fair comparison for `compute_totals`.
The monolithic approach will be much faster when using the coloring.
The nested approach will also become faster, but not as much as the factor for monolithic approach.
TODO: how can I exploit the coloring here?

Wall time `run_model` is the same for serial runs, but the nested approach might show slightly better scaling. (Or just a measurement error. Need to repeat or test on larger mesh to make sure.)
For derivatives computations, see the later section (of comparing time of 1 iterations of optimization.)

NOTE: for nested approach, `final_setup` of the subproblem is called within `setup` call, thus the time for `final_setup` appears in `setup` calls.

###  With `num_segments=20`
**Table: Monolithic approach wall times, [s]**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 7.72    | 3.82          | 9.46        | 105.72           |
| 2               | 7.68    | 6.43          | 5.14        | 67.71            |
| 4               | 7.97    | 6.33          | 2.85        | 44.61            |
| 8               | 9.00    | 6.99          | 1.75        | 40.58            |
| 16              | 12.76   | 10.08         | 1.34        | 48.87            |

\
**Table: Nested approach wall times, [s]**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 6.66    | 0.28          | 9.55        | 11.38            |
| 2               | 4.95    | 0.43          | 5.07        | 6.46             |
| 4               | 3.80    | 0.47          | 2.75        | 3.80             |
| 8               | 3.51    | 0.59          | 1.60        | 3.09             |
| 16              | 5.08    | 0.64          | 1.09        | 4.30             |

\
**Speedup comparison for `run_model`**
![Speed up (run_model)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_run_model_Nseg20.jpg?raw=true)

\
**Speedup plot for `compute_totals`**
![Speed up (compute_totals)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_compute_totals_Nseg20.jpg?raw=true)


###  With `num_segments=40`
**Table: Monolithic approach wall times, [s]**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 26.08   | 7.94          | 18.83       | 595.14           |
| 2               | 24.91   | 13.27         | 10.41       | 383.60           |
| 4               | 26.41   | 12.65         | 5.82        | 251.87           |
| 8               | 29.73   | 15.04         | 3.69        | 233.61           |

\
**Table: Nested approach wall times, [s]**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 15.97   | 0.49          | 18.86       | 24.43            |
| 2               | 12.88   | 0.98          | 10.15       | 14.63            |
| 4               | 11.12   | 0.83          | 5.53        | 9.33             |
| 8               | 11.52   | 0.99          | 3.21        | 9.47             |

\
**Speedup comparison for `run_model`**
![Speed up (run_model)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_run_model_Nseg40.jpg?raw=true)

\
**Speedup plot for `compute_totals`**
![Speed up (compute_totals)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_compute_totals_Nseg40.jpg?raw=true)


## Wall time comparison of 1 iteration of optimization
TODO: update
This timing is done in series.
I measured the wall time for SNOPT with only 1 iteration (analysis + derivatives computation).
This study reflects the total Jacobian coloring for derivatives computation.

NOTE: total coloring -> 11 FWD solves

###  With `num_segments=20`
**Table: Monolithic approach wall times, [s]**
| Number of procs | Analysis | Derivatives | 
| :-------------: | -------: | ----------: |
| 1               | 9.39     | 13.77       |
| 2               | 5.14     | 8.57        |
| 4               | 2.89     | 5.62        |
| 8               | 1.75     | 4.86        |
| 16              | 1.31     | 4.75        |

\
**Table: Nested approach wall times, [s]**
| Number of procs | Analysis | Derivatives | 
| :-------------: | -------: | ----------: |
| 1               | 9.60     | 10.83       |
| 2               | 5.10     | 6.02        |
| 4               | 2.77     | 3.42        |
| 8               | 1.61     | 2.40        |
| 16              | 1.06     | 2.31        |

\
**Speedup comparison for `run_model`**
![Speed up (run_model)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_run_model_Nseg20_SNOPT1.jpg?raw=true)

\
**Speedup plot for `compute_totals`**
![Speed up (compute_totals)](https://github.com/kanekosh/Dymos_parallel_analysis/blob/main/scaling_study/figs/speedup_compute_totals_Nseg20_SNOPT1.jpg?raw=true)



### Nested vs. Monolithic - in series
With wing twist and thickness variables (10 vars), total coloring yields 25 FWD solves.

NOTE: the followings are run on Mac - above scalability study is on workstation. That's why the time is different.

**Table: fixed-design trajectory optimization**
|                 | Analysis | Derivatives | 
| :-------------: | -------: | ----------: |
| Monolithic      | 6.26     | 13.27       |
| Nested          | 6.50     | 8.88        |


**Table: wing design & trajectory optimization**
|                 | Analysis | Derivatives | 
| :-------------: | -------: | ----------: |
| Monolithic      | 6.66     | 40.95       |
| Nested          | 6.46     | 9.15        |
