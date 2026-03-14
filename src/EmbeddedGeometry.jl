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
end # struct

"""
    struct SphereGeometry <: EmbeddedGeometry{3}

Spherical embedded geometry in 3D.
"""
struct SphereGeometry <: EmbeddedGeometry{3}
    R::Float64                      # radius
    x₀::VectorValue{3,Float64}      # domain midpoint
end # struct

# ===================================================
# Geometry builders — return GridapEmbedded geo objects
# ===================================================
function build_geometry(g::CylinderGeometry)
    return disk(g.R, x0=g.x₀)
end # function

function build_geometry(g::SphereGeometry)
    return sphere(g.R, x0=g.x₀)
end # function

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
end # function

"""
    geometry_cut(model::DiscreteModel, g::STLGeometry)

Cut a discrete model with an STL geometry.
Returns `(cutgeo, cutgeo_facets)`.
"""
function geometry_cut(model::DiscreteModel, g::STLGeometry)
    cutgeo = cut(model, g)
    return cutgeo, cutgeo.cutfacets
end # function

# ===================================================
# Distances — levelset function
# ===================================================
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
            δ    = pmid - point          # vector from point to center
            absδ = sqrt(δ ⋅ δ)
            δ̂    = δ / absδ
            Xd   = pmid - geo.R * δ̂     # point on boundary closest to x
            d    = Xd - point
            absd = sqrt(d ⋅ d)
            n    = absd > 0 ? d / absd : δ̂

            d_vec_cs.values[icell][ipoint] = d
            n_vec_cs.values[icell][ipoint] = n
            Xd_cs.values[icell][ipoint]    = Xd
        end
    end

    return d_vec_cs, n_vec_cs, Xd_cs
end # function

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
end # function

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
end # struct

# ===================================================
# Wrappers for each unfitted method: SBM or WSBM
# ===================================================
function compute_distances(::SBM, bgmodel::DiscreteModel,
                            geo::EmbeddedGeometry, Γ₁::BoundaryTriangulation,
                            fun::Function, degree::Int, t::Real)
    d, n, Xd = distances(bgmodel, Γ₁, geo, degree)
    fsbm     = fshifted(Xd, fun, Γ₁, degree)
    return DistanceData(d, n, Xd, fsbm)
end # function

function compute_distances(::SBM, bgmodel::DiscreteModel,
                            geo::STLGeometry, Γ₁::BoundaryTriangulation,
                            fun::Function, degree::Int, t::Real)
    d, n, Xd = distances(bgmodel, Γ₁, geo, degree)
    fsbm     = fshifted(Xd, fun, Γ₁, degree)
    return DistanceData(d, n, Xd, fsbm)
end # function

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
end # function

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
end # function