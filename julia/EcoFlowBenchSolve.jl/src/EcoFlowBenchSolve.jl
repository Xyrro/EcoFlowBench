"""
    EcoFlowBenchSolve

Wrapper around Circuitscape.jl and Omniscape.jl for EcoFlowBench dataset generation.

Array convention: every raster is a Julia `Matrix` indexed `[row, col]` = `(H, W)`, matching the
Python (C-order) layout. HDF5 datasets written by h5py with shape (H, W) are read here as
(W, H) and transposed on the way in/out (`h5_hw` / `h5_hw!`).

Public API
* `solve_pairwise(R, nodata, focal; kw...)`  -> `PairwiseResult`  (T1 / T1W / T2)
* `solve_advanced(R, nodata, S, G; kw...)`   -> `AdvancedResult`  (T3)
* `solve_omniscape(R, nodata, S; kw...)`     -> `OmniscapeResult` (T4)
* `solve_shard(inputs_h5, outputs_h5; kw...)` batch mode over an inputs shard (resumable)
* `solver_versions()`, `SolveStats`

Circuitscape 5.17.1 has no in-memory raster API, so pairwise/advanced solves write ASCII grids
to a scratch directory (node-local `/tmp` in jobs) and call `Circuitscape.compute(::Dict)`.
Omniscape is called in memory. Every solve records `SolveStats` (wall time, max RSS, solver,
precision, thread counts, versions, converged flag, error text).
"""
module EcoFlowBenchSolve

using Circuitscape
using Omniscape
using DelimitedFiles
using LinearAlgebra
using Dates
using Printf
using HDF5
using JSON
using Statistics
using Logging

export solve_pairwise, solve_advanced, solve_omniscape, solve_shard, solver_versions, SolveStats, sanitize_nan

const NODATA = -9999.0

# ---------------------------------------------------------------------------
# Versions / stats
# ---------------------------------------------------------------------------
function solver_versions()
    return (julia = string(VERSION), circuitscape = string(pkgversion(Circuitscape)),
            omniscape = string(pkgversion(Omniscape)))
end

Base.@kwdef mutable struct SolveStats
    solver::String = ""
    precision::String = "double"
    wall_s::Float64 = NaN
    maxrss_mb::Float64 = NaN
    converged::Bool = false
    error::String = ""
    fallback_used::Bool = false
    n_threads::Int = Threads.nthreads()
    blas_threads::Int = BLAS.get_num_threads()
    hostname::String = gethostname()
    julia_version::String = string(VERSION)
    circuitscape_version::String = string(pkgversion(Circuitscape))
    omniscape_version::String = string(pkgversion(Omniscape))
    solver_params::Dict{String,Any} = Dict{String,Any}()
    started_utc::String = ""
end

"""Recursively replace NaN/Inf by `nothing` (JSON null) so stats always serialise."""
sanitize_nan(x::AbstractFloat) = isfinite(x) ? x : nothing
sanitize_nan(x::AbstractDict) = Dict{String,Any}(string(k) => sanitize_nan(v) for (k, v) in x)
sanitize_nan(x::Union{AbstractVector,Tuple}) = [sanitize_nan(v) for v in x]
sanitize_nan(x) = x

stats_json(s::SolveStats) = JSON.json(sanitize_nan(Dict(String(f) => getfield(s, f) for f in fieldnames(SolveStats))))

# ---------------------------------------------------------------------------
# ASCII grid I/O (Circuitscape file interface)
# ---------------------------------------------------------------------------
function write_asc(path::AbstractString, A::AbstractMatrix; nodata = NODATA, cellsize = 100.0)
    open(path, "w") do io
        @printf(io, "ncols %d\nnrows %d\nxllcorner 0\nyllcorner 0\ncellsize %g\nNODATA_value %g\n",
                size(A, 2), size(A, 1), cellsize, nodata)
        writedlm(io, A, ' ')
    end
end

read_asc(path::AbstractString) = Matrix{Float64}(readdlm(path; skipstart = 6))

"""Replace Circuitscape's -9999 (NoData) values by `fill` and return Float32."""
function clean_map(A::AbstractMatrix; fill = 0.0f0)
    B = Float32.(A)
    B[B .== Float32(NODATA)] .= fill
    return B
end

# ---------------------------------------------------------------------------
# Reference Circuitscape configuration (configs/solver/circuitscape_reference.yaml)
# ---------------------------------------------------------------------------
function cs_base_config(; solver = "cholmod", four_neighbors = false, avg_resistances = false,
                        precision = "double", parallelize = false)
    Dict{String,String}(
        "data_type" => "raster",
        "habitat_map_is_resistances" => "True",
        "connect_four_neighbors_only" => four_neighbors ? "True" : "False",
        "connect_using_avg_resistances" => avg_resistances ? "True" : "False",   # explicit: average conductance
        "use_polygons" => "False",
        "solver" => solver,
        "precision" => precision,
        "use_64bit_indexing" => "True",
        "cholmod_batch_size" => "1000",
        "low_memory_mode" => "False",
        "parallelize" => parallelize ? "True" : "False",
        "set_null_currents_to_nodata" => "False",
        "set_null_voltages_to_nodata" => "False",
        "set_focal_node_currents_to_zero" => "False",
        "log_transform_maps" => "False",
        "compress_grids" => "False",
        "write_as_tif" => "False",
        "suppress_messages" => "True",
        "log_level" => "INFO",
    )
end

"""Config keys that affect the solution and are recorded per sample (plus the hard-coded CG settings)."""
function solver_params(cfg::Dict{String,String}; extra = Dict{String,Any}())
    keys_ = ["scenario", "solver", "precision", "connect_four_neighbors_only", "connect_using_avg_resistances",
             "habitat_map_is_resistances", "use_polygons", "ground_file_is_resistances", "use_direct_grounds",
             "use_unit_currents", "remove_src_or_gnd", "use_64bit_indexing", "parallelize"]
    d = Dict{String,Any}(k => cfg[k] for k in keys_ if haskey(cfg, k))
    # Circuitscape 5.17.1 hard-codes the CG settings (src/core.jl); recorded here for provenance.
    d["cg_rtol_hardcoded"] = 1e-6
    d["cg_itmax_hardcoded"] = 100_000
    d["accept_residual_hardcoded"] = 1e-4
    merge!(d, extra)
    return d
end


# ---------------------------------------------------------------------------
# Exact solver graph in Julia (mirror of ecoflowbench/sources/graph.py) for residual checks
# ---------------------------------------------------------------------------
using SparseArrays

"""Laplacian of the Circuitscape graph (8-neighbour, NoData removed, average conductance) and node index map."""
function graph_laplacian(R::AbstractMatrix, nodata::AbstractMatrix{Bool})
    H, W = size(R)
    idx = fill(0, H, W); n = 0
    for j in 1:W, i in 1:H            # column-major fill is fine; ids only need to be a bijection
        if !nodata[i, j]; n += 1; idx[i, j] = n; end
    end
    g = [nodata[i, j] ? 0.0 : 1.0 / Float64(R[i, j]) for i in 1:H, j in 1:W]
    I = Int[]; J = Int[]; V = Float64[]
    function add(i, j, i2, j2, diag)
        (1 <= i2 <= H && 1 <= j2 <= W) || return
        (idx[i, j] > 0 && idx[i2, j2] > 0) || return
        w = (g[i, j] + g[i2, j2]) / 2
        diag && (w /= sqrt(2.0))
        push!(I, idx[i, j]); push!(J, idx[i2, j2]); push!(V, w)
    end
    for j in 1:W, i in 1:H
        idx[i, j] == 0 && continue
        add(i, j, i, j + 1, false); add(i, j, i + 1, j, false)
        add(i, j, i + 1, j + 1, true); add(i, j, i + 1, j - 1, true)
    end
    G = sparse(I, J, V, n, n); G = G + G'
    L = spdiagm(0 => vec(sum(G, dims = 2))) - G
    return L, idx
end

"""
    kirchhoff_residual(L, idx, voltage, injection, grounded; supernode=falses(...))

‖r‖₂/‖b‖₂ with r = L v − b on free nodes. Grounded nodes (0 V) are excluded (they absorb any
current). Pixels in `supernode` form one short-circuited focal region (Circuitscape treats
repeated labels this way): their rows are replaced by the single equation
Σ_region (L v) = Σ_region b, i.e. only the region's net injected current is checked.
"""
function kirchhoff_residual(L, idx, voltage::AbstractMatrix, injection::AbstractMatrix, grounded::AbstractMatrix{Bool};
                            supernode::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    valid = idx .> 0
    order = sortperm(vec(idx[valid]))
    v = Float64.(vec(voltage[valid]))[order]
    b = Float64.(vec(injection[valid]))[order]
    gnd = vec(grounded[valid])[order]
    sn = supernode === nothing ? falses(length(v)) : vec(supernode[valid])[order]
    r = L * v .- b
    free = .!gnd .& .!sn
    rr = sum(abs2, r[free])
    any(sn) && (rr += (sum(r[sn]))^2)
    nb = norm(b[.!gnd])
    return nb > 0 ? sqrt(rr) / nb : NaN
end

# ---------------------------------------------------------------------------
# Pairwise (T1 / T1W / T2)
# ---------------------------------------------------------------------------
struct PairwiseResult
    cum_current::Matrix{Float32}
    reff::Matrix{Float64}                # K×K, ordered by focal label (1..K)
    labels::Vector{Int}
    pair_index::Matrix{Int32}            # P×2 (i, j) labels, i < j
    pair_current::Vector{Matrix{Float32}}
    pair_voltage::Vector{Matrix{Float32}}
    stats::SolveStats
end

function _run_cs(cfg::Dict{String,String}, st::SolveStats)
    st.started_utc = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS")
    GC.gc()
    rss0 = Sys.maxrss()
    t = @elapsed r = try
        with_logger(NullLogger()) do
            Circuitscape.compute(cfg)
        end
    catch err
        st.error = sprint(showerror, err)
        nothing
    end
    st.wall_s = t
    st.maxrss_mb = Sys.maxrss() / 2^20
    st.converged = r !== nothing
    return r
end

"""
    solve_pairwise(R, nodata, focal; solver="cholmod", fallback="cg+amg", keep_pair_maps=true, workdir=mktempdir())

`R` Float32/64 (H,W) resistances (≥1), `nodata` Bool (H,W), `focal` Int32 (H,W) labels (0 = none,
repeated labels = one region). Returns `PairwiseResult`. If the reference solver fails (e.g. out
of memory) and `fallback` is set, the solve is repeated with the fallback and `stats.fallback_used`
is set.
"""
function solve_pairwise(R::AbstractMatrix, nodata::AbstractMatrix{Bool}, focal::AbstractMatrix{<:Integer};
                        solver = "cholmod", fallback = "cg+amg", keep_pair_maps = true,
                        four_neighbors = false, workdir = mktempdir())
    H, W = size(R)
    Rw = Float64.(R); Rw[nodata] .= NODATA
    write_asc(joinpath(workdir, "habitat.asc"), Rw)
    write_asc(joinpath(workdir, "points.asc"), Int.(focal); nodata = 0)
    labels = sort(unique(vec(focal[focal .> 0])))
    K = length(labels)
    function attempt(sol)
        cfg = cs_base_config(; solver = sol, four_neighbors = four_neighbors)
        cfg["scenario"] = "pairwise"
        cfg["habitat_file"] = joinpath(workdir, "habitat.asc")
        cfg["point_file"] = joinpath(workdir, "points.asc")
        cfg["output_file"] = joinpath(workdir, "out.out")
        cfg["write_cur_maps"] = keep_pair_maps ? "True" : "False"
        cfg["write_volt_maps"] = keep_pair_maps ? "True" : "False"
        cfg["write_cum_cur_map_only"] = keep_pair_maps ? "False" : "True"
        cfg["write_max_cur_maps"] = "False"
        st = SolveStats(solver = sol)
        st.solver_params = solver_params(cfg; extra = Dict("K" => K, "keep_pair_maps" => keep_pair_maps))
        r = _run_cs(cfg, st)
        return r, st
    end
    r, st = attempt(solver)
    if r === nothing && fallback !== nothing && fallback != solver
        r, st2 = attempt(fallback)
        st2.fallback_used = true
        st2.error = "reference solver ($solver) failed: " * st.error * (r === nothing ? " | fallback failed: " * st2.error : "")
        st = st2
    end
    if r === nothing
        return PairwiseResult(zeros(Float32, H, W), fill(NaN, K, K), labels, zeros(Int32, 0, 2),
                              Matrix{Float32}[], Matrix{Float32}[], st)
    end
    # resistances matrix: first row/col are labels
    reff = fill(NaN, K, K)
    rl = Int.(round.(r[1, 2:end])); cl = Int.(round.(r[2:end, 1]))
    pos = Dict(l => i for (i, l) in enumerate(labels))
    for (a, la) in enumerate(cl), (b, lb) in enumerate(rl)
        reff[pos[la], pos[lb]] = r[a + 1, b + 1]
    end
    cum = clean_map(read_asc(joinpath(workdir, "out_cum_curmap.asc")))
    pairs = [(labels[i], labels[j]) for i in 1:K for j in (i + 1):K]
    pair_index = Matrix{Int32}(undef, length(pairs), 2)
    pc = Matrix{Float32}[]; pv = Matrix{Float32}[]
    # Circuitscape convention (verified 2026-09-05): for pair (i, j) node i is grounded (v = 0) and
    # one unit of current is injected at node j (v_j = Reff_ij). Residual is computed on the
    # full-precision Float64 voltages before the float32 cast.
    resid = 0.0
    L = idx = nothing
    for (p, (i, j)) in enumerate(pairs)
        pair_index[p, 1] = i; pair_index[p, 2] = j
        if keep_pair_maps
            push!(pc, clean_map(read_asc(joinpath(workdir, "out_curmap_$(i)_$(j).asc"))))
            v64 = read_asc(joinpath(workdir, "out_voltmap_$(i)_$(j).asc")); v64[v64 .== NODATA] .= 0.0
            if L === nothing; L, idx = graph_laplacian(R, nodata); end
            src = focal .== j; gnd = focal .== i
            inj = zeros(Float64, size(R)); inj[src] .= 1.0 / count(src)
            resid = max(resid, kirchhoff_residual(L, idx, v64, inj, gnd; supernode = count(src) > 1 ? src : nothing))
            push!(pv, Float32.(v64))
        end
    end
    st.solver_params["pair_convention"] = "pair (i,j): i grounded (0 V), 1 A injected at j"
    st.solver_params["residual_rel"] = keep_pair_maps ? resid : NaN
    return PairwiseResult(cum, reff, labels, pair_index, pc, pv, st)
end

# ---------------------------------------------------------------------------
# Advanced (T3)
# ---------------------------------------------------------------------------
struct AdvancedResult
    current::Matrix{Float32}
    voltage::Matrix{Float32}
    stats::SolveStats
end

function solve_advanced(R::AbstractMatrix, nodata::AbstractMatrix{Bool}, S::AbstractMatrix, G::AbstractMatrix;
                        solver = "cholmod", fallback = "cg+amg", four_neighbors = false, workdir = mktempdir())
    H, W = size(R)
    Rw = Float64.(R); Rw[nodata] .= NODATA
    write_asc(joinpath(workdir, "habitat.asc"), Rw)
    write_asc(joinpath(workdir, "source.asc"), Float64.(S); nodata = NODATA)
    write_asc(joinpath(workdir, "ground.asc"), Float64.(G .> 0); nodata = NODATA)
    function attempt(sol)
        cfg = cs_base_config(; solver = sol, four_neighbors = four_neighbors)
        cfg["scenario"] = "advanced"
        cfg["habitat_file"] = joinpath(workdir, "habitat.asc")
        cfg["source_file"] = joinpath(workdir, "source.asc")
        cfg["ground_file"] = joinpath(workdir, "ground.asc")
        cfg["ground_file_is_resistances"] = "False"
        cfg["use_direct_grounds"] = "True"
        cfg["use_unit_currents"] = "False"
        cfg["remove_src_or_gnd"] = "keepall"
        cfg["output_file"] = joinpath(workdir, "out.out")
        cfg["write_cur_maps"] = "True"
        cfg["write_volt_maps"] = "True"
        st = SolveStats(solver = sol)
        st.solver_params = solver_params(cfg)
        r = _run_cs(cfg, st)
        return r, st
    end
    r, st = attempt(solver)
    if r === nothing && fallback !== nothing && fallback != solver
        r, st2 = attempt(fallback); st2.fallback_used = true
        st2.error = "reference solver ($solver) failed: " * st.error * (r === nothing ? " | fallback failed: " * st2.error : "")
        st = st2
    end
    r === nothing && return AdvancedResult(zeros(Float32, H, W), zeros(Float32, H, W), st)
    cur = clean_map(read_asc(joinpath(workdir, "out_curmap.asc")))
    v64 = read_asc(joinpath(workdir, "out_voltmap.asc")); v64[v64 .== NODATA] .= 0.0
    L, idx = graph_laplacian(R, nodata)
    st.solver_params["residual_rel"] = kirchhoff_residual(L, idx, v64, Float64.(S), G .> 0)
    return AdvancedResult(cur, Float32.(v64), st)
end

# ---------------------------------------------------------------------------
# Omniscape (T4)
# ---------------------------------------------------------------------------
struct OmniscapeResult
    cum_current::Matrix{Float32}
    flow_potential::Matrix{Float32}
    normalized::Matrix{Float32}
    stats::SolveStats
end

function solve_omniscape(R::AbstractMatrix, nodata::AbstractMatrix{Bool}, S::AbstractMatrix;
                         radius::Int, block_size::Int, solver = "cg+amg", precision = "double",
                         source_threshold = 0.0, four_neighbors = false, correct_artifacts = true)
    H, W = size(R)
    Rw = Float64.(R); Rw[nodata] .= NODATA
    Sw = Float64.(S); Sw[nodata] .= NODATA
    Rm = missingarray(Rw, Float64, NODATA)
    Sm = missingarray(Sw, Float64, NODATA)
    cfg = Dict{String,String}(
        "radius" => string(radius), "block_size" => string(block_size), "project_name" => "efb",
        "source_threshold" => string(source_threshold), "source_from_resistance" => "false",
        "resistance_is_conductance" => "false", "r_cutoff" => "Inf", "buffer" => "0",
        "calc_flow_potential" => "true", "calc_normalized_current" => "true",
        "correct_artifacts" => correct_artifacts ? "true" : "false", "mask_nodata" => "true",
        "connect_four_neighbors_only" => four_neighbors ? "true" : "false", "solver" => solver,
        "precision" => precision, "parallelize" => "false", "parallel_batch_size" => "10",
        "write_raw_currmap" => "false", "write_as_tif" => "false", "suppress_cs_messages" => "true",
    )
    st = SolveStats(solver = solver, precision = precision)
    st.solver_params = Dict{String,Any}(k => cfg[k] for k in ["radius", "block_size", "source_threshold", "solver",
        "precision", "connect_four_neighbors_only", "correct_artifacts", "calc_flow_potential", "calc_normalized_current"])
    st.started_utc = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS")
    GC.gc()
    t = @elapsed res = try
        with_logger(NullLogger()) do
            run_omniscape(cfg, Rm; source_strength = Sm, write_outputs = false)
        end
    catch err
        st.error = sprint(showerror, err)
        nothing
    end
    st.wall_s = t
    st.maxrss_mb = Sys.maxrss() / 2^20
    st.converged = res !== nothing
    res === nothing && return OmniscapeResult(zeros(Float32, H, W), zeros(Float32, H, W), zeros(Float32, H, W), st)
    tofloat(a) = clean_map(missingarray_to_array(a, NODATA))
    return OmniscapeResult(tofloat(res[1]), tofloat(res[2]), tofloat(res[3]), st)
end

# ---------------------------------------------------------------------------
# HDF5 helpers (h5py (H,W) C-order <-> Julia (H,W))
# ---------------------------------------------------------------------------
h5_hw(g, name) = permutedims(read(g[name]))                     # (W,H) -> (H,W)
h5_hw!(g, name, A::AbstractMatrix) = (g[name] = permutedims(A); nothing)
function h5_phw!(g, name, mats::Vector{<:AbstractMatrix{T}}) where {T}
    isempty(mats) && return nothing
    P = length(mats); H, W = size(mats[1])
    A = Array{T,3}(undef, W, H, P)                               # Julia (W,H,P) == C-order (P,H,W)
    for p in 1:P
        A[:, :, p] = permutedims(mats[p])
    end
    g[name] = A
    nothing
end

# ---------------------------------------------------------------------------
# Warm-up: trigger JIT compilation on a tiny problem so recorded solve times are steady-state
# ---------------------------------------------------------------------------
function warmup(; tmproot = tempdir())
    R = ones(Float32, 12, 12); R[:, 6] .= 5.0f0
    nd = falses(12, 12)
    focal = zeros(Int32, 12, 12); focal[3, 3] = 1; focal[10, 10] = 2
    S = zeros(Float32, 12, 12); S[2:4, 2:4] .= 1.0f0
    G = zeros(Int8, 12, 12); G[12, :] .= 1
    t = @elapsed begin
        for sol in ("cholmod", "cg+amg")
            solve_pairwise(R, nd, focal; solver = sol, fallback = nothing, workdir = mktempdir(tmproot))
            solve_advanced(R, nd, S, G; solver = sol, fallback = nothing, workdir = mktempdir(tmproot))
            solve_omniscape(R, nd, S; radius = 3, block_size = 1, solver = sol)
        end
    end
    @info @sprintf("warm-up (JIT) done in %.1f s", t)
    return t
end

# ---------------------------------------------------------------------------
# Batch mode over a shard
# ---------------------------------------------------------------------------
"""
    solve_shard(inputs_h5, outputs_h5; solver="cholmod", fallback="cg+amg", omniscape_solver="cholmod",
                tmproot=tempdir(), keep_pair_maps_max_k=4, omni=(radius, block_size), only=nothing)

Inputs layout (written by `ecoflowbench.solve.prepare`):
    /samples/<sid>/inputs/{resistance (H,W) f32, nodata_mask (H,W) u8}
    /samples/<sid>/configs/<cname>/  attrs kind ∈ {points, wall_to_wall, regions, advanced, omniscape}
                                     datasets focal_mask | source_strength [+ ground]
    /samples/<sid> attrs: tier, omni_radius, omni_block_size
Outputs layout:
    /samples/<sid>/outputs/<cname>/{cum_current, reff, pair_index, pairwise_current, voltage | current, voltage |
                                    cum_current, flow_potential, normalized} + attr stats (JSON)
Resumable: samples that already have a complete outputs group are skipped.
"""
function solve_shard(inputs_h5::AbstractString, outputs_h5::AbstractString; solver = "cholmod", fallback = "cg+amg",
                     omniscape_solver = "cholmod", tmproot = tempdir(), keep_pair_maps_max_k = 4, only = nothing,
                     max_samples = typemax(Int), log_every = 10, configs = nothing, force = false, do_warmup = true)
    do_warmup && warmup(; tmproot)
    t_start = time()
    fin = h5open(inputs_h5, "r")
    fout = h5open(outputs_h5, isfile(outputs_h5) ? "r+" : "w")
    haskey(fout, "samples") || create_group(fout, "samples")
    sids = keys(fin["samples"])
    only !== nothing && (sids = filter(s -> s in only, sids))
    n_done = 0; n_skip = 0
    for (n, sid) in enumerate(sids)
        n > max_samples && break
        gin = fin["samples"][sid]
        done_before = haskey(fout["samples"], sid) && haskey(fout["samples"][sid], "outputs") && haskey(attrs(fout["samples"][sid]), "complete")
        if done_before && !force
            n_skip += 1; continue
        end
        R = h5_hw(gin["inputs"], "resistance")
        nodata = h5_hw(gin["inputs"], "nodata_mask") .> 0
        gout = haskey(fout["samples"], sid) ? fout["samples"][sid] : create_group(fout["samples"], sid)
        if force && haskey(gout, "outputs")
            oo = gout["outputs"]                      # re-solve only the listed configs, keep the others
        else
            haskey(gout, "outputs") && delete_object(gout, "outputs")
            oo = create_group(gout, "outputs")
        end
        for cname in keys(gin["configs"])
            configs !== nothing && !(cname in configs) && continue
            gc = gin["configs"][cname]
            haskey(oo, cname) && delete_object(oo, cname)
            kind = attrs(gc)["kind"]
            og = create_group(oo, cname)
            wd = mktempdir(tmproot)
            try
                if kind in ("points", "wall_to_wall", "regions")
                    focal = Int32.(h5_hw(gc, "focal_mask"))
                    K = length(unique(focal[focal .> 0]))
                    keep = K <= keep_pair_maps_max_k
                    res = solve_pairwise(R, nodata, focal; solver, fallback, keep_pair_maps = keep, workdir = wd)
                    h5_hw!(og, "cum_current", res.cum_current)
                    og["reff"] = permutedims(res.reff)
                    og["labels"] = Int32.(res.labels)
                    og["pair_index"] = permutedims(res.pair_index)
                    if keep
                        h5_phw!(og, "pairwise_current", res.pair_current)
                        h5_phw!(og, "voltage", res.pair_voltage)
                    end
                    attrs(og)["stats"] = stats_json(res.stats)
                elseif kind == "advanced"
                    S = h5_hw(gc, "source_strength"); G = h5_hw(gc, "ground")
                    res = solve_advanced(R, nodata, S, G; solver, fallback, workdir = wd)
                    h5_hw!(og, "current", res.current)
                    h5_hw!(og, "voltage", res.voltage)
                    attrs(og)["stats"] = stats_json(res.stats)
                elseif kind == "omniscape"
                    S = h5_hw(gc, "source_strength")
                    radius = Int(attrs(gin)["omni_radius"]); bs = Int(attrs(gin)["omni_block_size"])
                    thr = haskey(attrs(gc), "source_threshold") ? Float64(attrs(gc)["source_threshold"]) : 0.0
                    res = solve_omniscape(R, nodata, S; radius, block_size = bs, solver = omniscape_solver, source_threshold = thr)
                    h5_hw!(og, "cum_current", res.cum_current)
                    h5_hw!(og, "flow_potential", res.flow_potential)
                    h5_hw!(og, "normalized", res.normalized)
                    attrs(og)["stats"] = stats_json(res.stats)
                else
                    attrs(og)["stats"] = stats_json(SolveStats(error = "unknown config kind $kind"))
                end
            finally
                rm(wd; recursive = true, force = true)
            end
        end
        attrs(gout)["complete"] = 1
        flush(fout)
        n_done += 1
        if n_done % log_every == 0
            @info @sprintf("solved %d samples (%d skipped) in %.1f s", n_done, n_skip, time() - t_start)
        end
    end
    close(fin); close(fout)
    @info @sprintf("shard done: %d solved, %d skipped, %.1f s", n_done, n_skip, time() - t_start)
    return (solved = n_done, skipped = n_skip, seconds = time() - t_start)
end

end # module
