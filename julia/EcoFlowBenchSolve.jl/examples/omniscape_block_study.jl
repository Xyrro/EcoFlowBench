# Omniscape block-size fidelity study: solve one sample with two block sizes (same radius) and
# compare cum_current / flow_potential / normalized maps.
# julia --project=. examples/omniscape_block_study.jl <inputs.h5> <sample_id|first> <block_a> <block_b> <out.json>
using EcoFlowBenchSolve, HDF5, JSON, Statistics, LinearAlgebra
const E = EcoFlowBenchSolve
inputs, sid_arg, ba, bb, outjson = ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4]), ARGS[5]
f = h5open(inputs, "r")
sid = sid_arg == "first" ? first(keys(f["samples"])) : sid_arg
g = f["samples"][sid]
R = E.h5_hw(g["inputs"], "resistance"); nd = E.h5_hw(g["inputs"], "nodata_mask") .> 0
S = E.h5_hw(g["configs"]["omniscape"], "source_strength")
rad = Int(attrs(g)["omni_radius"]); tier = String(attrs(g)["tier"])
close(f)
E.warmup()
function metrics(a, b, valid)
    x = Float64.(a[valid]); y = Float64.(b[valid])
    d = x .- y
    Dict("rel_l2" => norm(d) / norm(x), "max_abs_diff_over_max" => maximum(abs.(d)) / maximum(abs.(x)),
         "pearson" => cor(x, y), "spearman" => cor(sortperm(sortperm(x)), sortperm(sortperm(y))),
         "log_rel_l2" => norm(log1p.(max.(x, 0)) .- log1p.(max.(y, 0))) / norm(log1p.(max.(x, 0))))
end
res = Dict{String,Any}("sample" => sid, "tier" => tier, "radius" => rad, "blocks" => (ba, bb), "H" => size(R, 1))
ra = E.solve_omniscape(R, nd, S; radius = rad, block_size = ba, solver = "cholmod")
rb = E.solve_omniscape(R, nd, S; radius = rad, block_size = bb, solver = "cholmod")
valid = .!nd
res["t_a"] = ra.stats.wall_s; res["t_b"] = rb.stats.wall_s
res["converged"] = (ra.stats.converged, rb.stats.converged)
res["cum_current"] = metrics(ra.cum_current, rb.cum_current, valid)
res["flow_potential"] = metrics(ra.flow_potential, rb.flow_potential, valid)
res["normalized"] = metrics(ra.normalized, rb.normalized, valid)
println("block study $tier r=$rad b=$ba vs b=$bb: cum rel_l2=", res["cum_current"]["rel_l2"], " pearson=", res["cum_current"]["pearson"],
        " t=", round(res["t_a"]), "s vs ", round(res["t_b"]), "s")
open(outjson, "w") do io; JSON.print(io, E.sanitize_nan(res), 1); end
