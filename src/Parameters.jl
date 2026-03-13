export EmbeddingMethod, AGFEM, CUTFEM, SBM, WSBM
export GeometryParams, PhysicsParams, SolverParams, SimulationParams
export setup_model

abstract type EmbeddingMethod end
struct AGFEM  <: EmbeddingMethod end
struct CUTFEM <: EmbeddingMethod end
struct SBM    <: EmbeddingMethod end
struct WSBM   <: EmbeddingMethod end

"""
    struct GeometryParams{N}

Geometric parameters for the simulation domain in N dimensions.
"""
struct GeometryParams{N}
    L₁::Float64         # horizontal domain length (x-direction)
    L₂::Float64         # horizontal domain length (y-direction, 3D only)
    L₃::Float64         # vertical domain length
    R::Float64          # radius (cylinder/sphere)

    function GeometryParams{N}(L₁, L₂, L₃, R) where {N}
        _check_dim(N)
        new{N}(L₁, L₂, L₃, R)
    end
end

# Convenience constructors
GeometryParams2D(L₁, L₃, R)     = GeometryParams{2}(L₁, 0.0, L₃, R)
GeometryParams3D(L₁, L₂, L₃, R) = GeometryParams{3}(L₁, L₂, L₃, R)

"""
    struct ManufacturedParams{P}

Flexible parameter container for Method of Manufactured Solutions.
Holds arbitrary named parameters required to construct a ManufacturedSolution.
"""
struct ManufacturedParams{P<:NamedTuple}
    params::P
end

# Keyword constructor: accepts any named parameters
ManufacturedParams(; kwargs...) = ManufacturedParams(NamedTuple(kwargs))

# Access fields directly via dot syntax
Base.getproperty(p::ManufacturedParams, s::Symbol) = s === :params ? getfield(p, :params) : getfield(p.params, s)
Base.propertynames(p::ManufacturedParams) = propertynames(p.params)

"""
    struct SolverParams

Numerical solver parameters.
"""
struct SolverParams
    n::Int64            # number of elements
    order::Int64        # polynomial order
    γg::Float64         # ghost penalty coefficient
    folder::String      # output folder
end

"""
    struct SimulationParams{N}

Top-level simulation struct composing all parameter groups.
N is the spatial dimension (2 or 3).
"""
struct SimulationParams{N}
    geometry::GeometryParams{N}
    manufactured::ManufacturedParams
    solver::SolverParams

    function SimulationParams{N}(geometry::GeometryParams{N}, manufactured::ManufacturedParams, solver::SolverParams) where {N}
        _check_dim(N)
        new{N}(geometry, manufactured, solver)
    end
end

function SimulationParams(geometry::GeometryParams{N},
                           manufactured::ManufacturedParams,
                           solver::SolverParams) where {N}
    SimulationParams{N}(geometry, manufactured, solver)
end

"""
    setup_model(params::SimulationParams) -> (model, labels)

Build a CartesianDiscreteModel from SimulationParams with correct boundary tags.
Tags:
  2D — "top": [6], "DT": [1,2,3,4,7,8]
  3D — "top": [22], "DT": [1..20,23,24,25,26]
"""
function setup_model(params::SimulationParams{N}) where {N}
    return _setup_model(Val(N), params)
end

function _setup_model(::Val{2}, params::SimulationParams)
    n, L₁, L₃ = params.solver.n, params.geometry.L₁, params.geometry.L₃
    domain     = (-L₁/2, L₁/2, -L₃, 0.0)
    model      = CartesianDiscreteModel(domain, (n, n))
    labels     = get_face_labeling(model)
    add_tag_from_tags!(labels, "top", [6])
    add_tag_from_tags!(labels, "DT",  [1,2,3,4,7,8])
    return model, labels
end

function _setup_model(::Val{3}, params::SimulationParams)
    n, L₁, L₂, L₃ = params.solver.n, params.geometry.L₁, params.geometry.L₂, params.geometry.L₃
    domain         = (-L₁/2, L₁/2, -L₂/2, L₂/2, -L₃, 0.0)
    model          = CartesianDiscreteModel(domain, (n, n, n))
    labels         = get_face_labeling(model)
    add_tag_from_tags!(labels, "top", [22])
    add_tag_from_tags!(labels, "DT",  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23,24,25,26])
    return model, labels
end