using Gridap
using GridapEmbedded

export FESpaceConfig, FESpaces, build_spaces

# ===================================================
# FESpace Configuration
# ===================================================
"""
    struct FESpaceConfig

Configuration for finite element space construction.

# Fields
- `order`:           polynomial order
- `dirichlet_tags`:  tags for Dirichlet boundary conditions (default: `["DT"]`)
- `t`:               time at which Dirichlet data is evaluated (default: `0.0`)
"""
struct FESpaceConfig
    order::Int
    dirichlet_tags::Vector{String}
    t::Float64
end

FESpaceConfig(order::Int)                              = FESpaceConfig(order, ["DT"], 0.0)
FESpaceConfig(order::Int, dirichlet_tags)              = FESpaceConfig(order, dirichlet_tags, 0.0)

# ===================================================
# FESpaces Struct
# ===================================================
"""
    struct FESpaces{TV, TU}

Container for test and trial finite element spaces.

# Fields
- `V`:  test space
- `U`:  trial space
"""
struct FESpaces{TV, TU}
    V::TV
    U::TU
end # struct

# ===================================================
# Build Spaces — generic fallback (CUTFEM, SBM, WSBM)
# ===================================================
"""
    build_spaces(::EmbeddingMethod, domain::Domain, config::FESpaceConfig, u::ManufacturedSolution)

Construct standard test and trial FE spaces.
Used for CUTFEM, SBM, and WSBM — no aggregation required.
"""
function build_spaces(::EmbeddingMethod, domain::Domain, config::FESpaceConfig, u::ManufacturedSolution)
    reffe = ReferenceFE(lagrangian, Float64, config.order)
    V     = TestFESpace(domain.Ω⁻act, reffe; dirichlet_tags=config.dirichlet_tags)
    U     = TrialFESpace(V, x -> u(x, config.t))
    return FESpaces(V, U)
end # function

# ===================================================
# Build Spaces — AGFEM (requires aggregation)
# ===================================================
"""
    build_spaces(::AGFEM, domain::Domain, config::FESpaceConfig, u::ManufacturedSolution,
                    cutgeo, g::EmbeddedGeometry, domain_config::DomainConfig)

Construct aggregated FE spaces for AGFEM.
Aggregation flag is derived from DomainConfig — consistent with _get_flags.
"""
function build_spaces(::AGFEM, domain::Domain, config::FESpaceConfig, u::ManufacturedSolution,
                        cutgeo, g::EmbeddedGeometry, domain_config::DomainConfig)
    f          = _get_flags(domain_config)
    geo        = build_geometry(g)
    reffe      = ReferenceFE(lagrangian, Float64, config.order)
    Vstd       = TestFESpace(domain.Ω⁻act, reffe; dirichlet_tags=config.dirichlet_tags)
    aggregates = aggregate(AggregateCutCellsByThreshold(1.0), cutgeo, geo, f.sbm_inner)
    V          = AgFEMSpace(Vstd, aggregates)
    U          = TrialFESpace(V, x -> u(x, config.t))
    return FESpaces(V, U)
end # function