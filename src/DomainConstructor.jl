using Gridap
using GridapEmbedded
using STLCutters

using GridapEmbedded.Interfaces
using STLCutters: STLEmbeddedDiscretization

export Domain, DomainConfig, DomainSide, INSIDE, OUTSIDE
export build_domain, build_reference_domain

export Measures, build_measures

# ===================================================
# Domain Side
# ===================================================
@enum DomainSide INSIDE OUTSIDE

# ===================================================
# Domain Configuration
# ===================================================
"""
    struct DomainConfig

Configuration for domain construction derived from two physical choices:
- `side`:         is the physical domain INSIDE or OUTSIDE the embedded boundary?
- `intersected`:  does the embedded boundary intersect a domain boundary?
- `Γ₂_tags`:      boundary tags for the external boundary Γ₂ (default: `["top"]`)

All GridapEmbedded flags and normal sign conventions are derived automatically.
"""
struct DomainConfig
    side::DomainSide
    intersected::Bool
    Γ₂_tags::Vector{String}
end

DomainConfig(side::DomainSide, intersected::Bool) = DomainConfig(side, intersected, ["top"])
DomainConfig() = DomainConfig(OUTSIDE, true, ["top"])

# ===================================================
# Flag Derivation
# ===================================================
"""
    _get_flags(config::DomainConfig)

Derive all GridapEmbedded flags and normal conventions from a DomainConfig.
Returns a NamedTuple with fields:
- `physical_flag`:  flag for the physical domain
- `active_flag`:    flag for the active domain
- `inactive_flag`:  flag for the inactive domain
- `ghost_flag`:     flag for the ghost skeleton
- `sbm_inner`:      flag for the SBM inner domain
- `sbm_cut`:        flag for the SBM cut domain
- `flip_normal`:    whether to flip the embedded boundary normal
"""
function _get_flags(config::DomainConfig)
    if config.side == OUTSIDE
        return (
            physical_flag  = PHYSICAL_OUT,
            active_flag    = ACTIVE_OUT,
            inactive_flag  = config.intersected ? ACTIVE_IN : IN,
            ghost_flag     = ACTIVE_OUT,
            sbm_inner      = OUT,
            sbm_cut        = CUT,
            flip_normal    = true
        )
    else  # INSIDE
        return (
            physical_flag  = PHYSICAL_IN,
            active_flag    = ACTIVE_IN,
            inactive_flag  = config.intersected ? ACTIVE_OUT : OUT,
            ghost_flag     = ACTIVE_IN,
            sbm_inner      = IN,
            sbm_cut        = CUT,
            flip_normal    = false
        )
    end
end

# ===================================================
# Domain Struct
# ===================================================
"""
    struct Domain{T1,T2,T3,T4,T5,T6,T7,T8,T9}

Container for all triangulations and normal vectors of a domain.

# Fields
- `Ω⁻`:     Physical interior domain
- `Ω⁻act`:  Active domain (AGFEM/CUTFEM only, else Nothing)
- `Γ₁`:     Embedded or surrogate boundary
- `nΓ₁`:    Normal of Γ₁
- `Γ₂`:     External boundary
- `nΓ₂`:    Normal of Γ₂
- `E⁰`:     Ghost skeleton (CUTFEM/WSBM only, else Nothing)
- `nE⁰`:    Normal of ghost skeleton (else Nothing)
- `Ωsbm`:   SBM interior domains (WSBM only, else Nothing)
"""
struct Domain{T1,T2,T3,T4,T5,T6,T7,T8,T9}
    Ω⁻::T1
    Ω⁻act::T2
    Γ₁::T3
    nΓ₁::T4
    Γ₂::T5
    nΓ₂::T6
    E⁰::T7
    nE⁰::T8
    Ωwsbm::T9
end

# ===================================================
# Base Builders (private)
# ===================================================
function _build_agfem_base(cutgeo, cutgeo_facets, config::DomainConfig)
    f     = _get_flags(config)
    Ω⁻    = Interior(cutgeo, f.physical_flag)
    Ω⁻act = Interior(cutgeo, f.active_flag)
    Γ₁    = EmbeddedBoundary(cutgeo)
    nΓ₁   = f.flip_normal ? -get_normal_vector(Γ₁) : get_normal_vector(Γ₁)
    Γ₂    = BoundaryTriangulation(cutgeo_facets, f.physical_flag, tags=config.Γ₂_tags)
    nΓ₂   = get_normal_vector(Γ₂)
    return Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂
end

function _build_sbm_base(cutgeo, config::DomainConfig)
    f     = _get_flags(config)
    Ω⁻act = Interior(cutgeo, f.active_flag)
    Ω⁻pas = Interior(cutgeo, f.inactive_flag)
    Γ₁    = Interface(Ω⁻pas, Ω⁻act).⁻                   # TO DO: verify that we do not need to flip to .⁺ if we flip from OUTSIDE to INSIDE
    nΓ₁   = get_normal_vector(Γ₁)
    Γ₂    = BoundaryTriangulation(Ω⁻act, tags=config.Γ₂_tags)
    nΓ₂   = get_normal_vector(Γ₂)
    return Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂
end

# ===================================================
# Public Build Functions
# ===================================================
function build_domain(method::AGFEM, cutgeo, cutgeo_facets, config::DomainConfig=DomainConfig())
    Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂ = _build_agfem_base(cutgeo, cutgeo_facets, config)
    Domain(Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂, nothing, nothing, nothing)
end

function build_domain(method::CUTFEM, cutgeo, cutgeo_facets, config::DomainConfig=DomainConfig())
    f     = _get_flags(config)
    Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂ = _build_agfem_base(cutgeo, cutgeo_facets, config)
    E⁰    = GhostSkeleton(cutgeo, f.ghost_flag)
    nE⁰   = get_normal_vector(E⁰)
    Domain(Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂, E⁰, nE⁰, nothing)
end

function build_domain(method::SBM, cutgeo, cutgeo_facets, config::DomainConfig=DomainConfig())
    Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂ = _build_sbm_base(cutgeo, config)
    Domain(Ω⁻act, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂, nothing, nothing, nothing)
end

function build_domain(method::WSBM, cutgeo, cutgeo_facets, config::DomainConfig=DomainConfig())
    f     = _get_flags(config)
    Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂ = _build_sbm_base(cutgeo, config)
    E⁰    = GhostSkeleton(cutgeo, f.ghost_flag)
    nE⁰   = get_normal_vector(E⁰)
    Ωwsbm  = (Interior(cutgeo, f.sbm_inner), Interior(cutgeo, f.sbm_cut))
    Domain(Ω⁻act, nothing, Γ₁, nΓ₁, Γ₂, nΓ₂, E⁰, nE⁰, Ωwsbm)
end

# STL wrapper — delegates to embedded discretization versions
function build_domain(method::EmbeddingMethod, cutgeo::STLEmbeddedDiscretization, cutgeo_facets, config::DomainConfig=DomainConfig())
    build_domain(method, cutgeo.cut, cutgeo.cutfacets, config)
end

# Reference domain uses SBM surrogate — cutgeo_facets unused
build_reference_domain(cutgeo, config::DomainConfig=DomainConfig()) = build_domain(SBM(), cutgeo, nothing, config)
build_reference_domain(cutgeo::STLEmbeddedDiscretization, config::DomainConfig=DomainConfig()) = build_domain(SBM(), cutgeo.cut, nothing, config)

# ===================================================
# Measures Struct
# ===================================================
"""
    struct Measures

Container for all quadrature measures derived from a Domain.

# Fields
- `dΩ⁻`:  measure on the physical/active interior domain
- `dΓ₁`:  measure on the embedded or surrogate boundary
- `dΓ₂`:  measure on the external boundary (Nothing if not needed)
- `dE⁰`:  measure on the ghost skeleton (Nothing for AGFEM/SBM)
"""
struct Measures{T1,T2,T3,T4}
    dΩ⁻::T1
    dΓ₁::T2
    dΓ₂::T3
    dE⁰::T4
end

# ===================================================
# Build Measures from Domain
# ===================================================
"""
    build_measures(domain::Domain, degree::Int)

Construct all quadrature measures from a Domain.
Measures are only built for fields that are not Nothing.
"""
function build_measures(domain::Domain, degree::Int)
    dΩ⁻ = Measure(domain.Ω⁻,  degree)
    dΓ₁ = Measure(domain.Γ₁,  degree)
    dΓ₂ = domain.Γ₂  !== nothing ? Measure(domain.Γ₂,  degree) : nothing
    dE⁰ = domain.E⁰  !== nothing ? Measure(domain.E⁰,  degree) : nothing
    return Measures(dΩ⁻, dΓ₁, dΓ₂, dE⁰)
end

# Helper — extract WSBM interior measures from domain.Ωsbm
function _get_wsbm_measures(domain::Domain, degree::Int64)
    @assert domain.Ωwsbm !== nothing "WSBM requires domain.Ωwsbm to be set"
    dΩᵢ = Measure(domain.Ωwsbm[1], degree)   
    dΩₒ = Measure(domain.Ωwsbm[2], degree)
    return dΩᵢ, dΩₒ
end

# ===================================================
# Volume fraction for WSBM
# ===================================================
function volume_fraction(cutgeo::EmbeddedDiscretization, Ω⁻act::Triangulation)
    Ω⁻    = Interior(cutgeo, CUT_OUT)
    Ω⁻cut = Interior(cutgeo, CUT)

    vol⁻    = get_cell_measure(Ω⁻, Ω⁻cut)
    vol⁻act = get_cell_measure(Ω⁻cut)
    γvol    = vol⁻ ./ vol⁻act

    bg_to_ioc    = compute_bgcell_to_inoutcut(cutgeo, cutgeo.geo)
    cell_to_mask = collect(Bool, bg_to_ioc .!= -1)
    bg_to_ioc2   = bg_to_ioc[cell_to_mask]
    inds         = findall(x -> x == 0, bg_to_ioc2)
    A            = float(bg_to_ioc2)
    A[inds]      = γvol

    CellField(A, Ω⁻act)
end

function volume_fraction(cutgeo::STLEmbeddedDiscretization, Ω⁻act::Triangulation)
    volume_fraction(cutgeo.cut, Ω⁻act)
end