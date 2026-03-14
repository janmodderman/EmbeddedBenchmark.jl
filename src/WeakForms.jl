using Gridap
using GridapEmbedded

using Gridap.Geometry

export WeakForm, build_weak_form

# ===================================================
# WeakForm Struct
# ===================================================
"""
    struct WeakForm{Ta, Tl}

Container for bilinear and linear form functions for a given method.

# Fields
- `a`:   NamedTuple of bilinear forms — keys depend on method
            AGFEM:  (interior=a₀)
            CUTFEM: (interior=a₀, ghost=aₑ)
            SBM:    (interior=a₀, boundary=aᵧ)
            WSBM:   (interior=a₀, boundary=aᵧ, ghost=aₑ, shift_edge=aₛ)
- `l`:   linear form (right hand side)
"""
struct WeakForm{Ta, Tl}
    a::Ta   # NamedTuple of bilinear forms
    l::Tl   # linear form
end

# ===================================================
# Operator Definitions
# ===================================================

# Shifting operator
# _s(∇ϕ, ∇∇ϕ, d, n) = ((∇∇ϕ⋅d + ∇ϕ)⋅n)*n - ∇ϕ

# Weighted test function
_w_α(α, w)    = α*w
_w_α(α, w, v) = α*(w⋅v)

# ===================================================
# Bilinear Forms
# ===================================================

# --- Interior ---
function _a_interior(dΩ::Measure)
    (ϕ, v) -> ∫(∇(ϕ)⋅∇(v))dΩ
end

function _a_interior(dΩᵢ::Measure, dΩₒ::Measure, α::CellField)
    (ϕ, v) -> ∫(∇(ϕ)⋅∇(v))dΩᵢ + ∫((_w_α∘(α, ∇(ϕ), ∇(v))))dΩₒ
end

# --- Ghost penalty ---
function _a_ghost(dE⁰::Measure, nE⁰::SkeletonPair, h::Float64, γg::Float64, ::Val{1})
    (ϕ, v) -> ∫((γg*(h^3))*jump(nE⁰⋅∇(v))⊙jump(nE⁰⋅∇(ϕ)))dE⁰
end

function _a_ghost(dE⁰::Measure, nE⁰::SkeletonPair, h::Float64, γg::Float64, ::Val{2})
    (ϕ, v) -> ∫((γg*(h^3))*jump(nE⁰⋅∇(v))⊙jump(nE⁰⋅∇(ϕ)) +
                (γg*(h^5))*jump(nE⁰⋅∇∇(v))⊙jump(nE⁰⋅∇∇(ϕ)))dE⁰
end

# --- Shift on edges ---
function _a_shift_edge(dE⁰::Measure, nE⁰::SkeletonPair,
                        dist_edg::DistanceData, α::CellField)
    d = dist_edg.d
    n = dist_edg.n
    (ϕ, v) -> ∫(jump(nE⁰ * (_w_α ∘ (α, v))) ⋅
                ((((mean(∇∇(ϕ)) ⋅ d) + mean(∇(ϕ))) ⋅ n) * n - mean(∇(ϕ))))dE⁰ +
                ∫(mean((_w_α ∘ (α, v))) ⋅ 
                ((((jump(∇∇(ϕ)) ⋅ d) + jump(∇(ϕ))) ⋅ n) * n - jump(∇(ϕ))))dE⁰
end

# --- Shift on boundary ---
function _a_shift_boundary(dΓ₁::Measure, nΓ₁::CellField, dist::DistanceData)
    d = dist.d
    n = dist.n
    (ϕ, v) -> ∫((nΓ₁ ⋅ ( ( ((∇∇(ϕ) ⋅ d) + ∇(ϕ)) ⋅ n) * n - ∇(ϕ))) * v)dΓ₁
end

function _a_shift_boundary(dΓ₁::Measure, nΓ₁::CellField, dist::DistanceData, α::CellField)
    d = dist.d
    n = dist.n
    (ϕ, v) -> ∫((nΓ₁ ⋅ ((((∇∇(ϕ) ⋅ d) + ∇(ϕ)) ⋅ n) * n - ∇(ϕ))) * (_w_α ∘ (α, v)))dΓ₁
end

# ===================================================
# Linear Forms (RHS)
# ===================================================

# AGFEM / CUTFEM — no shifting
function _l_standard(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
                        dΓ₂::Measure, nΓ₂::CellField, f₁::Function, f₂::Function)
    v -> ∫(f₁ * v)dΩ + ∫((nΓ₁ ⋅ f₂) * v)dΓ₁ + ∫((nΓ₂ ⋅ f₂) * v)dΓ₂
end

# SBM - shifting on boundary
function _l_sbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
                dΓ₂::Measure, nΓ₂::CellField,
                dist::DistanceData, f₁::Function, f₂::Function)
    v -> ∫(f₁ * v)dΩ +
         ∫((nΓ₁ * v) ⋅ dist.n * (dist.fsbm ⋅ dist.n))dΓ₁ +
         ∫((nΓ₂ ⋅ f₂) * v)dΓ₂
end

# WSBM - shifting on boundary and edges
# TO DO: verify correct righthandside! + investigate possible optimizations
function _l_wsbm(dΩᵢ::Measure, dΩₒ::Measure, dΓ₁::Measure, nΓ₁::CellField,
                    dE⁰::Measure, nE⁰::SkeletonPair,
                    dΓ₂::Measure, nΓ₂::CellField,
                    dist::NamedTuple, α::CellField,
                    f₁::Function, f₂::Function)
    v -> ∫(f₁ * (_w_α ∘ (α, v)))dΩₒ + ∫(f₁ * v)dΩᵢ +
         ∫((nΓ₁ * (_w_α ∘ (α, v))) ⋅ dist.boundary.n * (dist.boundary.fsbm ⋅ dist.boundary.n))dΓ₁ +
         ∫(jump(nE⁰ * (_w_α ∘ (α, v))) ⋅ dist.edges.n * (dist.edges.fsbm ⋅ dist.edges.n))dE⁰ +
         ∫( mean((_w_α ∘ (α, v))) * jump(nE⁰) ⋅ dist.edges.n * (dist.edges.fsbm ⋅ dist.edges.n))dE⁰ + 
         ∫((nΓ₂ ⋅ f₂) * (_w_α ∘ (α, v)))dΓ₂
end

# ===================================================
# Public Interface — build_weak_form dispatches on method
# ===================================================

"""
    build_weak_form(::AGFEM, measures, domain, params, f₁, f₂) -> WeakForm

Build weak form for AGFEM. Returns interior bilinear form and standard RHS.
"""
function build_weak_form(::AGFEM, measures::Measures, domain::Domain,
                            f₁::Function, f₂::Function)
    a = (interior = _a_interior(measures.dΩ⁻),)
    l = _l_standard(measures.dΩ⁻, measures.dΓ₁, domain.nΓ₁,
                    measures.dΓ₂, domain.nΓ₂, f₁, f₂)
    WeakForm(a, l)
end

"""
    build_weak_form(::CUTFEM, measures, domain, h, γg, order, f₁, f₂) -> WeakForm

Build weak form for CUTFEM. Returns interior + ghost penalty bilinear forms and standard RHS.
"""
function build_weak_form(::CUTFEM, measures::Measures, domain::Domain,
                            h::Float64, γg::Float64, order::Int64,
                            f₁::Function, f₂::Function)
    a = (interior = _a_interior(measures.dΩ⁻),
            ghost    = _a_ghost(measures.dE⁰, domain.nE⁰, h, γg, Val(order)))
    l = _l_standard(measures.dΩ⁻, measures.dΓ₁, domain.nΓ₁,
                    measures.dΓ₂, domain.nΓ₂, f₁, f₂)
    WeakForm(a, l)
end

"""
    build_weak_form(::SBM, measures, domain, n, d, f₁, f₂, f₂sbm) -> WeakForm

Build weak form for SBM. Returns interior + boundary shift bilinear forms and shifted RHS.
n and d can be Function (analytical) or Tuple (STL) — dispatch handles both.
"""
function build_weak_form(::SBM, measures::Measures, domain::Domain,
                            dist::DistanceData, f₁::Function, f₂::Function)
    a = (interior = _a_interior(measures.dΩ⁻),
            boundary = _a_shift_boundary(measures.dΓ₁, domain.nΓ₁, dist))
    l = _l_sbm(measures.dΩ⁻, measures.dΓ₁, domain.nΓ₁,
                measures.dΓ₂, domain.nΓ₂, dist, f₁, f₂)
    WeakForm(a, l)
end

"""
    build_weak_form(::WSBM, measures, domain, n, d, α, h, γg, order, f₁, f₂, f₂sbm) -> WeakForm

Build weak form for WSBM. Returns interior + boundary shift + ghost bilinear forms and weighted shifted RHS.
"""
function build_weak_form(::WSBM, measures::Measures, domain::Domain,
                            dist::NamedTuple, α::CellField,
                            h::Float64, γg::Float64, order::Int64,
                            f₁::Function, f₂::Function)
    a = (interior = _a_interior(measures.dΩ⁻[1], measures.dΩ⁻[2], α),
            boundary = _a_shift_boundary(measures.dΓ₁, domain.nΓ₁, dist.boundary, α),
            ghost    = _a_ghost(measures.dE⁰, domain.nE⁰, h, γg, Val(order)),
            shift_edge = _a_shift_edge(measures.dE⁰, domain.nE⁰, dist.edges, α))
    l = _l_wsbm(measures.dΩ⁻[1], measures.dΩ⁻[2], measures.dΓ₁, domain.nΓ₁,
                measures.dE⁰, domain.nE⁰,
                measures.dΓ₂, domain.nΓ₂,
                dist, α, f₁, f₂)
    WeakForm(a, l)
end
