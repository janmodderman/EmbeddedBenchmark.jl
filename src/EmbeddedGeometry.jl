using GridapEmbedded
using STLCutters

using GridapEmbedded.LevelSetCutters

export EmbeddedGeometry, AnalyticalGeometry
export CylinderGeometry, SphereGeometry
export build_geometry, geometry_cut

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
    R::Float64       # radius
    x₀::Float64      # vertical offset from domain midpoint
    Lₓ::Float64      # horizontal domain length
    L₃::Float64      # vertical domain length
end

"""
    struct SphereGeometry <: EmbeddedGeometry{3}

Spherical embedded geometry in 3D.
"""
struct SphereGeometry <: EmbeddedGeometry{3}
    R::Float64       # radius
    x₀::Float64      # vertical offset from domain midpoint
    Lₓ::Float64      # horizontal domain length
    L₃::Float64      # vertical domain length
end

# ===================================================
# Geometry builders — return GridapEmbedded geo objects
# ===================================================
function build_geometry(g::CylinderGeometry)
    pmin = Point(-g.Lₓ/2, -g.L₃)
    pmax = Point( g.Lₓ/2,  0.0)
    pmid = 0.5*(pmax + pmin) + VectorValue(0.0, g.x₀)
    return disk(g.R, x0=pmid)
end

function build_geometry(g::SphereGeometry)
    pmin = Point(-g.Lₓ/2, -g.Lₓ/2, -g.L₃)
    pmax = Point( g.Lₓ/2,  g.Lₓ/2,  0.0)
    pmid = 0.5*(pmax + pmin) + VectorValue(0.0, 0.0, g.x₀)
    return sphere(g.R, x0=pmid)
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
# Analytical distance functions for SBM & WSBM
# ===================================================
function analytical_distance(geo::CylinderGeometry, fun::Function)
    pmid = VectorValue(0.0, -geo.L₃ + geo.x₀)

    D(x)      = pmid - x
    absD(x)   = sqrt(D(x) ⋅ D(x))
    dist(x)   = absD(x) - geo.R
    n(x)      = dist(x) >= 0.0 ? D(x) / absD(x) : -D(x) / absD(x)
    d(x)      = abs(dist(x)) * n(x)

    n(t)      = x -> n(x)
    d(t)      = x -> d(x)
    funsbm(t) = x -> fun(x + d(x), t)

    return d, n, funsbm
end

function analytical_distance(geo::SphereGeometry, fun::Function)
    pmid = VectorValue(0.0, 0.0, -geo.L₃ + geo.x₀)

    D(x)      = pmid - x
    absD(x)   = sqrt(D(x) ⋅ D(x))
    dist(x)   = absD(x) - geo.R
    n(x)      = dist(x) >= 0.0 ? D(x) / absD(x) : -D(x) / absD(x)
    d(x)      = abs(dist(x)) * n(x)

    n(t)      = x -> n(x)
    d(t)      = x -> d(x)
    funsbm(t) = x -> fun(x + d(x), t)

    return d, n, funsbm
end

# ===================================================
# STL distance functions for SBM & WSBM
# ===================================================
function stl_distance(model::DiscreteModel, geo::STLGeometry,
                      Γ₁::BoundaryTriangulation, dΓ₁::Measure,
                      fun::Function)
    _, n, d, xd, _ = DistanceSTL.STLdistance(model, geo, Γ₁, dΓ₁)
    return (d,), (n,), (DistanceSTL.fshifted(xd, fun, dΓ₁),)
end

function stl_distance(model::DiscreteModel, geo::STLGeometry,
                      Γ₁::BoundaryTriangulation, dΓ₁::Measure,
                      E⁰::SkeletonTriangulation, dE⁰::Measure,
                      fun::Function)
    _, n₁, d₁, xd₁, _ = DistanceSTL.STLdistance(model, geo, Γ₁, dΓ₁)
    _, nₑ, dₑ, xdₑ, _ = DistanceSTL.STLdistance(model, geo, E⁰, dE⁰)
    return (d₁, dₑ), (n₁, nₑ),
           (DistanceSTL.fshifted(xd₁, fun, dΓ₁), DistanceSTL.fshifted(xdₑ, fun, dE⁰))
end