# Reference-solver validation (Phase 5 owner requirement):
#   * solve N samples of an inputs shard with CHOLMOD and with CG+AMG; report max relative
#     differences in cum_current and Reff (pairwise) and current (advanced) and Omniscape cum_current
#   * determinism: solve M samples twice per solver; report bitwise equality
# julia --project=. examples/solver_comparison.jl <inputs.h5> <out.json> [n_compare=10] [n_repeat=5]
using EcoFlowBenchSolve, HDF5, JSON, Statistics, LinearAlgebra
const E = EcoFlowBenchSolve
inputs, outjson = ARGS[1], ARGS[2]
n_cmp = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
n_rep = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5
relmax(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / max(maximum(abs.(Float64.(a))), eps())
function load(f, sid)
    g = f["samples"][sid]
    R = E.h5_hw(g["inputs"], "resistance"); nd = E.h5_hw(g["inputs"], "nodata_mask") .> 0
    focal = Int32.(E.h5_hw(g["configs"]["points"], "focal_mask"))
    S = E.h5_hw(g["configs"]["advanced"], "source_strength"); G = E.h5_hw(g["configs"]["advanced"], "ground")
    So = E.h5_hw(g["configs"]["omniscape"], "source_strength")
    rad = Int(attrs(g)["omni_radius"]); bs = Int(attrs(g)["omni_block_size"])
    return R, nd, focal, S, G, So, rad, bs
end
f = h5open(inputs, "r"); sids = collect(keys(f["samples"]))
res = Dict{String,Any}("n_compare" => 0, "n_repeat" => 0, "threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
                       "compare" => [], "determinism" => [])
# warm-up (JIT) on the first sample
let (R, nd, focal, S, G, So, rad, bs) = load(f, sids[1])
    E.solve_pairwise(R, nd, focal; solver = "cholmod", fallback = nothing, workdir = mktempdir())
    E.solve_pairwise(R, nd, focal; solver = "cg+amg", fallback = nothing, workdir = mktempdir())
end
for sid in sids[1:min(n_cmp, end)]
    R, nd, focal, S, G, So, rad, bs = load(f, sid)
    pc = E.solve_pairwise(R, nd, focal; solver = "cholmod", fallback = nothing, workdir = mktempdir())
    pg = E.solve_pairwise(R, nd, focal; solver = "cg+amg", fallback = nothing, workdir = mktempdir())
    ac = E.solve_advanced(R, nd, S, G; solver = "cholmod", fallback = nothing, workdir = mktempdir())
    ag = E.solve_advanced(R, nd, S, G; solver = "cg+amg", fallback = nothing, workdir = mktempdir())
    oc = E.solve_omniscape(R, nd, So; radius = rad, block_size = bs, solver = "cholmod")
    og = E.solve_omniscape(R, nd, So; radius = rad, block_size = bs, solver = "cg+amg")
    push!(res["compare"], Dict(
        "sample" => sid, "K" => length(pc.labels),
        "pairwise_converged" => (pc.stats.converged, pg.stats.converged),
        "cum_current_reldiff" => relmax(pc.cum_current, pg.cum_current),
        "reff_reldiff" => relmax(pc.reff, pg.reff),
        "reff_max" => maximum(pc.reff),
        "advanced_converged" => (ac.stats.converged, ag.stats.converged),
        "advanced_current_reldiff" => relmax(ac.current, ag.current),
        "advanced_voltage_reldiff" => relmax(ac.voltage, ag.voltage),
        "omniscape_converged" => (oc.stats.converged, og.stats.converged),
        "omniscape_cum_reldiff" => (oc.stats.converged && og.stats.converged) ? relmax(oc.cum_current, og.cum_current) : NaN,
        "t_pairwise" => (pc.stats.wall_s, pg.stats.wall_s), "t_advanced" => (ac.stats.wall_s, ag.stats.wall_s),
        "t_omniscape" => (oc.stats.wall_s, og.stats.wall_s),
        "residual_cholmod" => (get(pc.stats.solver_params, "residual_rel", NaN), get(ac.stats.solver_params, "residual_rel", NaN)),
        "residual_cgamg" => (get(pg.stats.solver_params, "residual_rel", NaN), get(ag.stats.solver_params, "residual_rel", NaN)),
        "cg_error" => pg.stats.error))
    res["n_compare"] += 1
    println("compare $(sid[1:8]) K=$(length(pc.labels)) cum Δ=$(res["compare"][end]["cum_current_reldiff"]) reff Δ=$(res["compare"][end]["reff_reldiff"]) omni Δ=$(res["compare"][end]["omniscape_cum_reldiff"])")
end
for sid in sids[1:min(n_rep, end)]
    R, nd, focal, S, G, So, rad, bs = load(f, sid)
    d = Dict{String,Any}("sample" => sid)
    for sol in ("cholmod", "cg+amg")
        a = E.solve_pairwise(R, nd, focal; solver = sol, fallback = nothing, workdir = mktempdir())
        b = E.solve_pairwise(R, nd, focal; solver = sol, fallback = nothing, workdir = mktempdir())
        d["pairwise_bitwise_$sol"] = (a.cum_current == b.cum_current) && (a.reff == b.reff)
        d["pairwise_maxdiff_$sol"] = maximum(abs.(a.cum_current .- b.cum_current))
        oa = E.solve_omniscape(R, nd, So; radius = rad, block_size = bs, solver = sol)
        ob = E.solve_omniscape(R, nd, So; radius = rad, block_size = bs, solver = sol)
        d["omniscape_bitwise_$sol"] = oa.cum_current == ob.cum_current
        d["omniscape_maxdiff_$sol"] = maximum(abs.(oa.cum_current .- ob.cum_current))
    end
    push!(res["determinism"], d); res["n_repeat"] += 1
    println("determinism $(sid[1:8]): ", d)
end
close(f)
open(outjson, "w") do io; JSON.print(io, sanitize_nan(res), 1); end
println("wrote ", outjson)
