using Test
using EcoFlowBenchSolve

@testset "EcoFlowBenchSolve skeleton" begin
    v = EcoFlowBenchSolve.solver_versions()
    @test v.julia == string(VERSION)
    @test v.circuitscape != "unknown"
    @test v.omniscape != "unknown"
end
