# Batch entry point used by the Python driver / Slurm jobs.
# julia --project=<pkg> scripts/solve_shard.jl <inputs.h5> <outputs.h5> [--solver cholmod] [--fallback cg+amg]
#        [--omniscape-solver cg+amg] [--tmp /tmp] [--max N]
using EcoFlowBenchSolve
args = copy(ARGS)
function popopt!(a, name, default)
    i = findfirst(==(name), a)
    i === nothing && return default
    v = a[i + 1]; deleteat!(a, i:i + 1); v
end
solver = popopt!(args, "--solver", "cholmod")
fallback = popopt!(args, "--fallback", "cg+amg")
osolver = popopt!(args, "--omniscape-solver", "cholmod")
tmp = popopt!(args, "--tmp", tempdir())
maxn = parse(Int, popopt!(args, "--max", string(typemax(Int))))
length(args) == 2 || error("usage: solve_shard.jl <inputs.h5> <outputs.h5> [options]")
fb = fallback == "none" ? nothing : fallback
r = solve_shard(args[1], args[2]; solver, fallback = fb, omniscape_solver = osolver, tmproot = tmp, max_samples = maxn)
println("SHARD_RESULT ", r)
