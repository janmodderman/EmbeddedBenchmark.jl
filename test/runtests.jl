module EmbeddedBenchmarkTests
using Test

# Test custom embedding method types and dispatch
@testset "DomainConstructor" begin include("test_DomainConstructor.jl") end
@testset "EmbeddedGeometry" begin include("test_EmbeddedGeometry.jl") end
@testset "ManufacturedSolutions" begin include("test_ManufacturedSolutions.jl") end

# TO DO: Add tests for the following:
# - PostProcessing
# - Parameters
# - BenchmarkRunner
# - SolutionConstructor
# - WeakForms
# - FESpaces

end # module