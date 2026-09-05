# EcoFlowBenchSolve.jl

Julia package wrapping [Circuitscape.jl](https://github.com/Circuitscape/Circuitscape.jl) and
[Omniscape.jl](https://github.com/Circuitscape/Omniscape.jl) for EcoFlowBench dataset generation.

Activate with `JULIA_PROJECT` pointing at this directory (done by `scripts/env.sh`), then:

```julia
using EcoFlowBenchSolve
EcoFlowBenchSolve.solver_versions()
```
