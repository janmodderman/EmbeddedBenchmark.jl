using Gridap
using ForwardDiff
using StaticArrays

export ManufacturedSolution, AirySolution
export manufactured_functions
export AirySolution2D, AirySolution3D

"""
    abstract type ManufacturedSolution end

Abstract type for manufactured solutions in N-dimensional space.
N must be 2 or 3.
"""
abstract type ManufacturedSolution{N} end

function _check_dim(N)
    N isa Int && N in (2, 3) || throw(ArgumentError("Dimension N must be 2 or 3, got $N"))
end

# const PointLike{N,T} = Union{AbstractVector{T}, VectorValue{N,T}}

# TO DO: Implement a general function to compute f₁ and f₂ from any ManufacturedSolution using ForwardDiff
# function manufactured_functions(s::ManufacturedSolution{N}) where {N}
#     u_fd(x::SVector{N,<:Real},t) = s(x,t)

#     u(x,t)  = u_fd(SVector(Tuple(x)), t)

#     ∇u(x,t) = VectorValue(Tuple(ForwardDiff.gradient(u_fd, SVector(Tuple(x)))))

#     function ∇u_fd(y::SVector{N,<:Real})
#         ForwardDiff.gradient(u_fd, y)
#     end

#     Δu(x) = -tr(ForwardDiff.jacobian(∇u_fd, SVector(Tuple(x))))

#     return u, ∇u, Δu
# end


"""
    struct AirySolution <: ManufacturedSolution end

Airy wave solution
"""
struct AirySolution{N} <: ManufacturedSolution{N}
    g::Float64          # gravitational constant [m/s²]
    k::Float64          # wave number [rad/m]
    η₀::Float64         # wave amplitude [m]
    d::Float64          # vertical domain length
    ω::Float64          # angular frequency [rad/s]
    function AirySolution{N}(g, k, η₀, d) where {N}
        _check_dim(N)
        ω = sqrt(g * k * tanh(k * d))   # Dispersion relation for linear gravity waves
        new{N}(g, k, η₀, d, ω)
    end
end

function manufactured_functions(s::AirySolution{2})
    # Analytical solution
    u(x::VectorValue{2},t::Real) = s.η₀*s.g/s.ω * (cosh(s.k*(x[2]+s.d))/cosh(s.k*s.d)) * sin(s.k*x[1] - s.ω*t)

    # Analytical gradient — derived by hand
    # ∂u/∂x₁ = η₀g/ω * cosh(k(x₂+d))/cosh(kd) * k*cos(kx₁ - ωt)
    # ∂u/∂x₂ = η₀g/ω * k*sinh(k(x₂+d))/cosh(kd) * sin(kx₁ - ωt)
    ∇u(x::VectorValue{2},t::Real) = VectorValue(
        s.η₀*s.g/s.ω * (cosh(s.k*(x[2]+s.d))/cosh(s.k*s.d)) * s.k*cos(s.k*x[1] - s.ω*t),
        s.η₀*s.g/s.ω * (s.k*sinh(s.k*(x[2]+s.d))/cosh(s.k*s.d)) * sin(s.k*x[1] - s.ω*t)
    )

    # Analytical Laplacian — Airy 2D satisfies ∇²u = 0 exactly
    # ∂²u/∂x₁² = -k²u,  ∂²u/∂x₂² = k²u  →  sum = 0
    Δu(x::VectorValue{2},t::Real) = 0.0

    return u, ∇u, Δu
end

function manufactured_functions(s::AirySolution{3})
    # Analytical solution
    u(x::VectorValue{3},t::Real) = s.η₀*s.g/s.ω * (cosh(s.k*(x[3]+s.d))/cosh(s.k*s.d)) * sin(s.k*x[1] + s.k*x[2] - s.ω*t)

    # Analytical gradient
    # ∂u/∂x₁ = η₀g/ω * cosh(k(x₃+d))/cosh(kd) * k*cos(kx₁ + kx₂ - ωt)
    # ∂u/∂x₂ = η₀g/ω * cosh(k(x₃+d))/cosh(kd) * k*cos(kx₁ + kx₂ - ωt)
    # ∂u/∂x₃ = η₀g/ω * k*sinh(k(x₃+d))/cosh(kd) * sin(kx₁ + kx₂ - ωt)
    ∇u(x::VectorValue{3},t::Real) = VectorValue(
        s.η₀*s.g/s.ω * (cosh(s.k*(x[3]+s.d))/cosh(s.k*s.d)) * s.k*cos(s.k*x[1] + s.k*x[2] - s.ω*t),
        s.η₀*s.g/s.ω * (cosh(s.k*(x[3]+s.d))/cosh(s.k*s.d)) * s.k*cos(s.k*x[1] + s.k*x[2] - s.ω*t),
        s.η₀*s.g/s.ω * (s.k*sinh(s.k*(x[3]+s.d))/cosh(s.k*s.d)) * sin(s.k*x[1] + s.k*x[2] - s.ω*t)
    )

    # Analytical Laplacian — 3D Airy with equal k in x₁ and x₂
    # ∂²u/∂x₁² = -k²u,  ∂²u/∂x₂² = -k²u,  ∂²u/∂x₃² = 2k²u  →  sum = 0... 
    # Wait: ∂²u/∂x₃² = k²u (vertical), horizontal = -k²u - k²u = -2k²u
    # So Δu = -2k²u + k²u = -k²u ≠ 0 for 3D with k in both horizontal dirs
    Δu(x::VectorValue{3},t::Real) = -s.k^2 * u(x,t)

    return u, ∇u, Δu
end

# Convenience constructors for 2D and 3D Airy solutions
AirySolution2D(g, k, η₀, d) = AirySolution{2}(g, k, η₀, d)
AirySolution3D(g, k, η₀, d) = AirySolution{3}(g, k, η₀, d)

# Define the solution function for Airy waves in 2D and 3D
(s::AirySolution{2})(x::VectorValue{2}, t::Real) = s.η₀*s.g/s.ω*(cosh(s.k*(x[2]+s.d))/cosh(s.k*s.d))*sin(s.k*x[1] - s.ω*t)
(s::AirySolution{3})(x::VectorValue{3}, t::Real) = s.η₀*s.g/s.ω*(cosh(s.k*(x[3]+s.d))/cosh(s.k*s.d))*sin(s.k*x[1] + s.k*x[2] - s.ω*t)

# TO DO: Implement TrigSolution and PolynomialSolution types and their corresponding solution functions
# TO DO: Add more manufactured solutions: x, y, z, separate dependencies, etc.