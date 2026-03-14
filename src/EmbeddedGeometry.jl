using GridapEmbedded
using STLCutters

using GridapEmbedded.LevelSetCutters

"""
    TO DO: implement transient dependency on distances function
"""

# ===================================================
# Abstract type
# ===================================================
abstract type EmbeddedGeometry{N} end

# ===================================================
# Concrete geometry types
# ===================================================

"""
    struct CylinderGeometry <: EmbeddedGeometry{2}

Circular/cylindrical embedded geometry in 2D.
"""
struct CylinderGeometry <: EmbeddedGeometry{2}
    R::Float64                      # radius
    x₀::VectorValue{2,Float64}      # vertical offset from domain midpoint
    # Lₓ::Float64                     # horizontal domain length
    # L₃::Float64                     # vertical domain length
end

"""
    struct SphereGeometry <: EmbeddedGeometry{3}

Spherical embedded geometry in 3D.
"""
struct SphereGeometry <: EmbeddedGeometry{3}
    R::Float64                      # radius
    x₀::VectorValue{3,Float64}      # domain midpoint
    # Lₓ::Float64                     # horizontal domain length
    # L₃::Float64                     # vertical domain length
end

# ===================================================
# Geometry builders — return GridapEmbedded geo objects
# ===================================================
function build_geometry(g::CylinderGeometry)
    return disk(g.R, x0=g.x₀)
end

function build_geometry(g::SphereGeometry)
    return sphere(g.R, x0=g.x₀)
end

# ===================================================
# Cutting functions
# ===================================================

"""
    geometry_cut(model::DiscreteModel, g::EmbeddedGeometry)

Cut a discrete model with an analytical embedded geometry.
Returns `(cutgeo, cutgeo_facets)`.
"""
function geometry_cut(model::DiscreteModel, g::EmbeddedGeometry)
    geo = build_geometry(g)
    return cut(model, geo), cut_facets(model, geo)
end

"""
    geometry_cut(model::DiscreteModel, g::STLGeometry)

Cut a discrete model with an STL geometry.
Returns `(cutgeo, cutgeo_facets)`.
"""
function geometry_cut(model::DiscreteModel, g::STLGeometry)
    cutgeo = cut(model, g)
    return cutgeo, cutgeo.cutfacets
end

# ===================================================
# Distances — levelset function
# ===================================================
# TO DO: assert δ correctness in general scenarios
function distances(bgmodel::DiscreteModel,
                    Γ::Union{BoundaryTriangulation, SkeletonTriangulation},
                    geo::EmbeddedGeometry{N}, degree::Int) where {N}

    Dspace   = num_point_dims(Γ)
    pmid     = geo.x₀
    QΓ       = CellQuadrature(Γ, degree)
    qcp      = get_cell_points(QΓ)
    z        = zero(VectorValue{Dspace, Float64})
    d_vec_cs = CellState(z, QΓ)
    n_vec_cs = CellState(z, QΓ)
    Xd_cs    = CellState(z, QΓ)

    for (icell, cell) in enumerate(qcp.cell_phys_point)
        for (ipoint, point) in enumerate(cell)
            δ    = pmid - point                     # is order important here when we change INSIDE or OUTSIDE?
            absδ = sqrt(δ ⋅ δ)
            dist = absδ - geo.R
            n    = dist >= 0.0 ? δ / absδ : -δ / absδ
            d_vec_cs.values[icell][ipoint] = dist * n
            n_vec_cs.values[icell][ipoint] = n
            Xd_cs.values[icell][ipoint] = point + d_vec_cs.values[icell][ipoint]
        end
    end

    return d_vec_cs, n_vec_cs, Xd_cs
end

# ===================================================
# Distances — STL
# ===================================================
# TO DO: verify and test STL cases
function distances(bgmodel::DiscreteModel,
                    Γ::Union{BoundaryTriangulation, SkeletonTriangulation},
                    geo::STLGeometry, degree::Int)

    topo           = Gridap.Geometry.get_grid_topology(bgmodel)
    D              = num_dims(topo)
    Dspace         = num_point_dims(Γ)
    QΓ             = CellQuadrature(Γ, degree)
    qcp            = get_cell_points(QΓ)

    cell_to_facets  = STLCutters.compute_cell_to_facets(bgmodel, STLCutters.get_stl(geo))
    face_to_cells   = Gridap.Geometry.get_faces(topo, D-1, D)
    face_to_facets  = STLCutters.compose_index_map(face_to_cells, cell_to_facets)
    face_to_facets  = face_to_facets[_face_to_bgface(Γ)]

    z        = zero(VectorValue{Dspace, Float64})
    d_vec_cs = CellState(z, QΓ)
    n_vec_cs = CellState(z, QΓ)
    Xd_cs = CellState(z, QΓ)

    for (icell, cell) in enumerate(qcp.cell_phys_point)
        Xc = STLCutters.closest_point(cell, geo,
                                        repeat(face_to_facets[icell], outer=length(cell)))
        for (ipoint, point) in enumerate(cell)
            δ = Xc[ipoint] - point
            d_vec_cs.values[icell][ipoint] = δ
            n_vec_cs.values[icell][ipoint] = δ / √(δ ⋅ δ)
            Xd_cs.values[icell][ipoint] = point + d_vec_cs.values[icell][ipoint]
        end
    end

    return d_vec_cs, n_vec_cs, Xd_cs
end

# Function to shift analytical solutions, required for MMS with SBM & WSBM
# TO DO: currently only tested for returning VectorValue, verify scalars or higher order tensors work as well
function fshifted(Xd::CellState, fun::Function, 
                    Γ::Union{BoundaryTriangulation, SkeletonTriangulation}, 
                    degree::Int)

    Dspace   = num_point_dims(Γ)
    QΓ       = CellQuadrature(Γ, degree)
    z        = zero(VectorValue{Dspace, Float64})
    vals     = Xd.values                                          # values of coordinate of quadrature points + distance function at corresponding quadrature point
    fshift   = CellState(z,QΓ)                                   # empty CellState
    phys_pts = get_cell_points(QΓ).cell_phys_point
    for icell in 1:length(phys_pts)
        for ipoint in 1:length(phys_pts[icell])
            fshift.values[icell][ipoint] = fun(vals[icell][ipoint])  # evaluate analytical function at Xd
        end
    end
    return fshift                                                     # return CellState of shifted analytical function at each quadrature point
end # function

# ===================================================
# Distance result container
# ===================================================
struct DistanceData
    d::CellState    # distance vector
    n::CellState    # normal vector
    Xd::CellState   # shifted point
    fsbm::CellState # shifted analytical function
end

# ===================================================
# Wrappers for each unfitted method: SBM or WSBM
# ===================================================
function compute_distances(::SBM, bgmodel::DiscreteModel,
                            geo::EmbeddedGeometry, Γ₁::BoundaryTriangulation,
                            fun::Function, degree::Int, t::Real)
    d, n, Xd = distances(bgmodel, Γ₁, geo, degree)
    fsbm     = fshifted(Xd, fun, Γ₁, degree)
    return DistanceData(d, n, Xd, fsbm)
end

function compute_distances(::SBM, bgmodel::DiscreteModel,
                            geo::STLGeometry, Γ₁::BoundaryTriangulation,
                            fun::Function, degree::Int, t::Real)
    d, n, Xd = distances(bgmodel, Γ₁, geo, degree)
    fsbm     = fshifted(Xd, fun, Γ₁, degree)
    return DistanceData(d, n, Xd, fsbm)
end

function compute_distances(::WSBM, bgmodel::DiscreteModel,
                            geo::EmbeddedGeometry, Γ₁::BoundaryTriangulation,
                            E⁰::SkeletonTriangulation,
                            fun::Function, degree::Int, t::Real)
    d₁, n₁, Xd₁ = distances(bgmodel, Γ₁, geo, degree)
    dₑ, nₑ, Xdₑ = distances(bgmodel, E⁰, geo, degree)
    fsbm₁        = fshifted(Xd₁, fun, Γ₁, degree)
    fsbmₑ        = fshifted(Xdₑ, fun, E⁰, degree)
    return (
        boundary = DistanceData(d₁, n₁, Xd₁, fsbm₁),
        edges    = DistanceData(dₑ, nₑ, Xdₑ, fsbmₑ)
    )
end

function compute_distances(::WSBM, bgmodel::DiscreteModel,
                            geo::STLGeometry, Γ₁::BoundaryTriangulation,
                            E⁰::SkeletonTriangulation,
                            fun::Function, degree::Int, t::Real)
    d₁, n₁, Xd₁ = distances(bgmodel, Γ₁, geo, degree)
    dₑ, nₑ, Xdₑ = distances(bgmodel, E⁰, geo, degree)
    fsbm₁        = fshifted(Xd₁, fun, Γ₁, degree)
    fsbmₑ        = fshifted(Xdₑ, fun, E⁰, degree)
    return (
        boundary = DistanceData(d₁, n₁, Xd₁, fsbm₁),
        edges    = DistanceData(dₑ, nₑ, Xdₑ, fsbmₑ)
    )
end