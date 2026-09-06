using Test
using AmpScapeSolve

@testset "AmpScapeSolve skeleton" begin
    v = AmpScapeSolve.solver_versions()
    @test v.julia == string(VERSION)
    @test v.circuitscape != "unknown"
    @test v.omniscape != "unknown"
end
