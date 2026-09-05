"""
    EcoFlowBenchSolve

Thin wrapper around Circuitscape.jl and Omniscape.jl that exposes in-memory
solve functions for the EcoFlowBench dataset generator.

Planned public API (Phase 5):

* `solve_pairwise(R, focal; params)`   -> pairwise Circuitscape (T1/T2)
* `solve_advanced(R, S, G; params)`    -> advanced-mode Circuitscape (T3)
* `solve_omniscape(R, S; params)`      -> Omniscape (T4)
* `SolveStats`                         -> wall time, iterations, memory, versions, converged

This file is a skeleton created in Phase 1; the solver functions are
implemented in Phase 5 (see docs/TASK_BRIEF.md §7.1).
"""
module EcoFlowBenchSolve

using Circuitscape
using Omniscape

export solver_versions

"""
    solver_versions() -> NamedTuple

Return the exact versions of Julia, Circuitscape.jl and Omniscape.jl in the
active environment. Recorded in every sample's metadata.
"""
function solver_versions()
    return (
        julia = string(VERSION),
        circuitscape = string(pkgversion(Circuitscape)),
        omniscape = string(pkgversion(Omniscape)),
    )
end

end # module
