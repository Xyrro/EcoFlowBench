# Solver smoke test: run the smallest built-in Circuitscape pairwise example and the
# built-in Omniscape test configuration, and compare against the reference outputs
# shipped with each package. Run from the package directory:
#   julia --project=. examples/smoke_test.jl <outdir>
using Circuitscape, Omniscape, DelimitedFiles, Pkg, Printf

outdir = length(ARGS) >= 1 ? ARGS[1] : mktempdir()
isdir(outdir) && rm(outdir; recursive = true)   # start clean so Omniscape does not suffix the project dir
mkpath(outdir)

pkgdir(name) = begin
    d = filter(p -> p.second.name == name, Pkg.dependencies())
    first(values(d)).source
end
cs_root = pkgdir("Circuitscape")
om_root = pkgdir("Omniscape")
println("Circuitscape source: ", cs_root)
println("Omniscape source:    ", om_root)

read_asc(p) = readdlm(p; skipstart = 6)

# ---------------------------------------------------------------------------
# 1. Circuitscape pairwise, test case 1 (5x5 raster with polygons, CHOLMOD)
# ---------------------------------------------------------------------------
cs_test = joinpath(cs_root, "test")
cs_out = joinpath(outdir, "circuitscape")
mkpath(cs_out)
cd(cs_test) do
    ini = "input/raster/pairwise/1/sgVerify1.ini"
    cfg = Dict{String,String}(Circuitscape.parse_config(ini))   # struct -> Dict (as in the package tests)
    cfg["output_file"] = joinpath(cs_out, "sgVerify1.out")
    tmp_ini = joinpath(cs_out, "sgVerify1.ini")
    open(tmp_ini, "w") do io
        for (k, v) in cfg
            println(io, k, " = ", v)
        end
    end
    t = @elapsed r = Circuitscape.compute(cfg)
    println(@sprintf("Circuitscape pairwise example 1 solved in %.2f s", t))
    println("resistances (first rows):")
    show(stdout, "text/plain", r[1:min(end, 4), :]); println()
end
ref = read_asc(joinpath(cs_test, "output_verify", "sgVerify1_cum_curmap.asc"))
got = read_asc(joinpath(cs_out, "sgVerify1_cum_curmap.asc"))
err = maximum(abs.(ref .- got))
println("cum_curmap size ", size(got), " max|ref-got| = ", err)
err < 1e-6 || error("Circuitscape smoke test mismatch: $err")

# ---------------------------------------------------------------------------
# 2. Omniscape built-in test (30x30 resistance, radius 5, block 2)
# ---------------------------------------------------------------------------
om_test = joinpath(om_root, "test")
om_out = joinpath(outdir, "omniscape")
mkpath(om_out)
cd(om_test) do
    cfg = Omniscape.parse_cfg("input/config.ini")
    cfg["solver"] = "cg+amg"            # the shipped ini deliberately sets an invalid name
    cfg["project_name"] = joinpath(om_out, "test1")
    cfg["write_as_tif"] = "false"
    cfg["parallelize"] = "false"
    ini = joinpath(om_out, "config.ini")
    open(ini, "w") do io
        for (k, v) in cfg
            println(io, k, " = ", v)
        end
    end
    t = @elapsed res = run_omniscape(ini)
    println(@sprintf("Omniscape example solved in %.2f s", t))
    println("outputs written: ", readdir(joinpath(om_out, "test1")))
end
ref = Omniscape.read_raster(joinpath(om_test, "output_verify", "test1", "cum_currmap.tif"), Float64)[1]
got = read_asc(joinpath(om_out, "test1", "cum_currmap.asc"))
ref = coalesce.(ref, -9999.0)
mask = .!(ref .== -9999) .& .!(got .== -9999)
err = maximum(abs.(ref[mask] .- got[mask]))
println("omniscape cum_currmap size ", size(got), " max|ref-got| = ", err)
err < 1e-4 || error("Omniscape smoke test mismatch: $err")

# ---------------------------------------------------------------------------
# 3. Versions + determinism (run Circuitscape example twice, compare bitwise)
# ---------------------------------------------------------------------------
cd(cs_test) do
    ini = joinpath(cs_out, "sgVerify1.ini")
    r1 = Circuitscape.compute(ini)
    r2 = Circuitscape.compute(ini)
    println("bitwise-identical resistances across two runs: ", r1 == r2)
end
println("Julia ", VERSION, "  threads=", Threads.nthreads())
for (_, d) in Pkg.dependencies()
    d.name in ("Circuitscape", "Omniscape", "AlgebraicMultigrid", "Krylov", "IterativeSolvers") && println("  ", d.name, " ", d.version)
end
println("SMOKE TEST OK")
