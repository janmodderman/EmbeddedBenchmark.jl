export build_solution

"""
    build_solution(::Type{S}, p::ManufacturedParams, N::Int) where {S<:ManufacturedSolution}

Construct a ManufacturedSolution of type S in N dimensions from a ManufacturedParams.
"""
function build_solution(::Type{AirySolution}, p::ManufacturedParams, ::Val{N}) where {N}
    AirySolution{N}(p.g, p.k, p.η₀, p.d)
end

# function build_solution(::Type{TrigSolution}, p::ManufacturedParams, ::Val{N}) where {N}
#     TrigSolution{N}(p.k, p.ω)
# end

# function build_solution(::Type{PolynomialSolution}, p::ManufacturedParams, ::Val{N}) where {N}
#     PolynomialSolution{N}()
# end

# Convenience: infer N from SimulationParams
function build_solution(::Type{S}, p::ManufacturedParams, params::SimulationParams{N}) where {S<:ManufacturedSolution, N}
    build_solution(S, p, Val{N}())
end