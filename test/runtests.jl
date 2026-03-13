module EmbeddedBenchmarkTests
using Test

# Test custom embedding method types and dispatch
@testset "DomainConstructor" begin include("test_DomainConstructor.jl") end
@testset "EmbeddedGeometry" begin include("test_EmbeddedGeometry.jl") end
@testset "ManufacturedSolutions" begin include("test_ManufacturedSolutions.jl") end
# @testset "helpers" begin include("test_helpers.jl") end
# @testset "manufactured_functions" begin include("test_manufacturedfunctions.jl") end

end # module