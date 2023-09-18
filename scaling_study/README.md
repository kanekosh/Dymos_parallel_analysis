# Scaling studies

## [Memory usage scaling](./memory_usage.md)
The issue seems to be that the memory usage during `prob.setup()` increases nearly linearly when increasing the number of processors.
This happens for a Dymos+OAS problem but not for a massive multipoint OAS problem.

## [Wall time scaling](./walltime_scaling.md)
The speedup was primarily limited mainly because of the memory bound, and I don't think this is an issue with OpenMDAO.