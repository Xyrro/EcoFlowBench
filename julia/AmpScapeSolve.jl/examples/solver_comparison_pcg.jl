# Item 3 (2026-09-06): cross-check CHOLMOD outputs against (a) AmpScape's AMG-PCG (the fixed acceleration
# baseline solver) and (b) Circuitscape's own cg+amg path, on N mini samples with K <= 4 (first pair) and T3.
# julia --project=. examples/solver_comparison_pcg.jl <inputs.h5> <out.json> [N=10]
using AmpScapeSolve, HDF5, JSON, LinearAlgebra, Statistics
const E = AmpScapeSolve
inputs, outjson = ARGS[1], ARGS[2]; N = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 10
relmax(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / max(maximum(abs.(Float64.(a))), eps())
f = h5open(inputs, "r"); res = Any[]
E.warmup()
for sid in keys(f["samples"])
    length(res) >= N && break
    g = f["samples"][sid]
    focal = Int32.(E.h5_hw(g["configs"]["points"], "focal_mask"))
    K = length(unique(focal[focal .> 0])); K <= 4 || continue
    R = E.h5_hw(g["inputs"], "resistance"); nd = E.h5_hw(g["inputs"], "nodata_mask") .> 0
    S = E.h5_hw(g["configs"]["advanced"], "source_strength"); G = E.h5_hw(g["configs"]["advanced"], "ground")
    L, idx = E.graph_laplacian(R, nd)
    # reference: CHOLMOD (no refinement needed at S) — first pair
    pc = E.solve_pairwise(R, nd, focal; solver = "cholmod", fallback = nothing, workdir = mktempdir(), cg_baseline_enabled = false)
    i, j = pc.pair_index[1, 1], pc.pair_index[1, 2]; src = focal .== j; gnd = focal .== i
    inj = zeros(size(R)); inj[src] .= 1.0 / count(src)
    # (a) our AMG-PCG to the CHOLMOD residual level
    t_pcg = @elapsed cb = E.cg_baseline(L, idx, inj, gnd; rtol_targets = (1e-6, 1e-10))
    # rebuild the PCG solution vector by re-running to 1e-10 and mapping back (cg_baseline returns stats only)
    valid = idx .> 0; order = sortperm(vec(idx[valid])); free = .!(vec(gnd[valid])[order])
    Lf = L[free, free]; b = Float64.(vec(inj[valid]))[order][free]
    P = AmpScapeSolve.aspreconditioner(AmpScapeSolve.ruge_stuben(Lf))
    x = zeros(length(b)); r = b .- Lf * x; z = similar(r); AmpScapeSolve.ldiv!(z, P, r); p = copy(z); rz = dot(r, z); it = 0
    while norm(r) / norm(b) > 1e-10 && it < 20000
        Ap = Lf * p; a = rz / dot(p, Ap); x .+= a .* p; r .-= a .* Ap; AmpScapeSolve.ldiv!(z, P, r); rz2 = dot(r, z); p .= z .+ (rz2 / rz) .* p; rz = rz2; it += 1
    end
    vf = zeros(count(valid)); vf[free] .= x; vmap = zeros(size(R)); vmap[valid] .= vf[invperm(order)]
    cur_pcg = E.node_current_map(L, idx, vmap; region_pixels = src .| gnd, region_current = 1.0)
    reff_pcg = maximum(vmap[src])
    # (b) Circuitscape cg+amg
    pg = E.solve_pairwise(R, nd, focal; solver = "cg+amg", fallback = nothing, workdir = mktempdir(), cg_baseline_enabled = false, refine = false)
    ac = E.solve_advanced(R, nd, S, G; solver = "cholmod", fallback = nothing, workdir = mktempdir(), cg_baseline_enabled = false)
    ag = E.solve_advanced(R, nd, S, G; solver = "cg+amg", fallback = nothing, workdir = mktempdir(), cg_baseline_enabled = false, refine = false)
    d = Dict("sample" => sid, "K" => K,
             "pcg_iters_1e-06" => cb["iters_to_1e-06"], "pcg_iters_1e-10" => cb["iters_to_1e-10"], "pcg_time_s" => t_pcg,
             "pcg_residual" => norm(r) / norm(b), "cholmod_residual" => pc.stats.solver_params["residual_rel"],
             "pair_current_reldiff_pcg" => relmax(pc.pair_current[1], cur_pcg), "reff_reldiff_pcg" => abs(pc.reff[1, 2] - reff_pcg) / pc.reff[1, 2],
             "cum_current_reldiff_cscg" => relmax(pc.cum_current, pg.cum_current), "reff_reldiff_cscg" => relmax(pc.reff, pg.reff),
             "cscg_residual" => pg.stats.solver_params["residual_rel"], "t_cholmod" => pc.stats.wall_s, "t_cscg" => pg.stats.wall_s,
             "adv_current_reldiff_cscg" => relmax(ac.current, ag.current), "adv_residual_cholmod" => ac.stats.solver_params["residual_rel"],
             "adv_residual_cscg" => ag.stats.solver_params["residual_rel"])
    push!(res, d); println(JSON.json(d))
end
close(f)
open(outjson, "w") do io; JSON.print(io, E.sanitize_nan(res), 1); end
println("wrote ", outjson, " (", length(res), " samples)")
