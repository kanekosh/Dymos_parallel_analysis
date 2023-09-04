# Dymos trajectory optimization with parallel aerostructural analyses

## Problem setup
This repo implements a toy trajectory optimization problem that minimizes the energy consumption for a level, straight flight of aircraft.
At each node, the aircraft ODE model calls OpenAeroStruct (OAS) aerostructural analysis to compute the CL and CD given the flow speed and angle of attack.

With a collocation formulation, nodes are parallel-in-time, therefore the ODE function evaluations (which involve OAS analysis) can be parallelized.
Note that the parallelization is only done at Dymos node level - each OAS analysis is not parallelized.
The parallelization structure here is very similar to parallel multipoint analyses and derivative computation.

The N2 diagram showing the problem structure is available in `n2.html`.

## Environment and dependencies
- Ubuntu 20.04.6 LTS
- CPU: 16 cores / 32 threads (AMD Ryzen 5950X, 3.4 GHz)
- RAM: 64 GB
- Python 3.11.1

Python packages and versions are listed in `requirements.txt` (excluding `pyOptSparse`, which is not hosted on PyPI).
Major packages I used are:
- numpy 1.25.2
- scipy 1.10.1
- openmdao 3.27.0
- dymos 1.8.0
- openaerostruct 2.7.0
- petsc 3.19.4
- petsc4py 3.19.4
- mpi4py 3.1.4
- pyoptsparse 2.9.2

## Reproducing the issue
```
cd scripts
mpirun -n 16 python run_optimization.py
```

On my machine with 64 GB RAM, this ran out of memory, and the process was killed with signal 9.
If your machine has more RAM, you may need to run it with a larger problem size to reproduce the issue.
This can be done by increasing the OAS mesh size or dymos num_segments.

A summary of memory usage scaling can be found [here](scaling_study/README.md).

## Ref: Python files
All Python files are in `scripts`.
- `run_optimization.py`: trajectory optimization runscript. Currently, this does not call `run_driver` for debugging purposes. Instead, it only calls `run_model` and `compute_totals` once after setting up the optimization problem. 
- `get_oas_surface.py`: creates an OAS surface.
- `dynamics.py`: aircraft ODE Group. Inside the ODE Group, we have an aerodynamic Group `AeroForceOAS` (defined in `aero_oas.py`) which does the aerostructural analyses.
- `aero_oas.py`: aerodynamic model. The Group `OASAnalyses` implements the parallel-in-node OAS analyses.
- `utils.py`: utility functions.