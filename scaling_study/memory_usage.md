# Memory Usage Scaling

## NOTE on problem setup
OpenAeroStruct mesh: `num_y=21, num_x=5`.  
Dymos grid setting: `tx = dm.Radau(num_segments=40, order=3, solve_segments=False, compressed=True)`.
This results in 160 nodes, each of them has an OAS analysis inside.
(Note that I used different number of segments from the wall time scaling studies.)

All of the scaling studies below are the strong scaling study, i.e., the problem size remains the same for all numbers of processors.

## Dymos+OAS monolithic problem

The following table summarizes the maximum memory usage *per processor* for each part of the code: `setup`, `final_setup`, `run_model`, and `compute_totals`.  
The memory usage is shown in % of the total memory of my machine (64 GB).
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

## Dymos+OAS problem using subproblem for OAS analyses - did not help!
Above, I directly put OAS analysis as a group under the dynamics group.
This results in a very big single OpenMDAO model that has access to all OAS components and variables.
I'd call this a *monolithic* approach.

Alternatively, I also tried a *subproblem* (or *nested*) approach.
This solves the identical problem to the monolithic approach, but instead of directly putting OAS group below dymos, I wrapped each OAS analysis as an OpenMDAO problem.
This subproblem takes wing design and flight conditions, and returns lift and drag to the top-level.
Therefore, the top-level OpenMDAO (Dymos) problem does not see OAS internal components and variables, hence it is much compact.
(The current code does not use `SubmodelComp` because I didn't know it exists, but I think it does essentially the same thing.)

I thought this might help the memory issue, but it turns out it does not help.
The memory usage trends was similar to the monolithic approach.

**Table: Memory usage per processor: subproblem approach**
| Number of procs | `setup` | `final_setup` | `run_model` | `compute_totals` | 
| :-------------: | ------: | ------------: | ----------: | ---------------: |
| 1               | 11.6%   | 7.6%          | 7.6%        | 14.0%            |
| 2               | 10.1%   | 4.2%          | 4.2%        | 7.3%             |
| 4               | 9.3%    | 2.3%          | 2.3%        | 3.8%             |
| 8               | 8.9%    | 1.3%          | 1.3%        | 2.1%             |

The current runscript is defaulted to the monolithic approach.
To run the subproblem approach, go to `aero_oas.py`, and in `OASAnalyses` group, set the option `use_subproblem` to `True`.

## Massive multipoint OAS problem (without Dymos)

Finally, I also run a massive multipoint OAS problem without Dymos, which has (kind of) a similar problem structure to the Dymos+OAS problem.
For this case, I used a different OAS mesh size (41x9) and the number of nodes (400 nodes) to see the trends clearly.

The following table summarizes the per-processor memory usage during `prob.setup()`.
We can see that memory usage decreases as we increase the number of processors, which is a good thing.
This implies that memory scaling for massive multipoint problem won't be problematic unlike the Dymos+OAS case.

**Table: Memory usage per processor: multipoint OAS without Dymos**
| Number of procs | `setup` |
| :-------------: | ------: |
| 1               | 30.4%   |
| 2               | 15.8%   |
| 4               | 8.4%    |
| 8               | 4.7%    |
| 16              | 2.8%    |
