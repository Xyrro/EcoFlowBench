using Test
using AmpScapeSolve

@testset "AmpScapeSolve skeleton" begin
    v = AmpScapeSolve.solver_versions()
    @test v.julia == string(VERSION)
    @test v.circuitscape != "unknown"
    @test v.omniscape != "unknown"
end

@testset "cg_baseline on a tiny grid" begin
    R = ones(Float32, 10, 10); R[:, 5] .= 20.0f0
    nd = falses(10, 10)
    focal = zeros(Int32, 10, 10); focal[2, 2] = 1; focal[9, 9] = 2
    L, idx = AmpScapeSolve.graph_laplacian(R, nd)
    inj = zeros(10, 10); inj[9, 9] = 1.0
    gnd = focal .== 1
    b0 = AmpScapeSolve.cg_baseline(L, idx, inj, gnd; rtol_targets = (1e-6, 1e-10))
    @test b0["converged_1e-06"] && b0["iters_to_1e-06"] >= 1 && b0["residual_1e-10"] <= 1e-9
    # warm start from the converged solution: residual_start tiny and (near) zero iterations
    valid = idx .> 0
    v = zeros(10, 10)
    # rebuild the solution vector as a map to test the warm-start path
    order = sortperm(vec(idx[valid])); free = .!(vec(gnd[valid])[order])
    x = L[free, free] \ Float64.(vec(inj[valid]))[order][free]
    vf = zeros(count(valid)); vf[free] .= x
    vmap = zeros(10, 10); vmap[valid] .= vf[invperm(order)]
    b1 = AmpScapeSolve.cg_baseline(L, idx, inj, gnd; rtol_targets = (1e-6,), x0 = vmap)
    @test b1["warm_start"] && b1["residual_start"] < 1e-8 && b1["iters_to_1e-06"] <= 1
end
