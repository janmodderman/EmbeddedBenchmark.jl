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
_s(∇ϕ, ∇∇ϕ, d, n) = ((∇∇ϕ⋅d + ∇ϕ)⋅n)*n - ∇ϕ

# RHS shifting operator
_sᵣ(fun::CellField, n::CellField) = n*(n⋅fun)

# Weighted test function
_w_α(α, w)    = α*w
_w_α(α, w, v) = α*(w⋅v)

# CellField helpers
function _make_cellfield(trian::Triangulation, fun::Function)
    D   = num_point_dims(trian)
    x₀  = zero(VectorValue{D, Float64})
    fun_val = fun(x₀)
    fun_typed(x::VectorValue{D, Float64}) where D = fun(x)
    CellField(fun_typed, trian)
end

# function _make_cellfields(trian::Triangulation, d::Function, n::Function)
#     D   = num_point_dims(trian)
#     x₀  = zero(VectorValue{D, Float64})
#     # Probe to get concrete return types
#     d_val = d(x₀)
#     n_val = n(x₀)
#     # Wrap as typed functions so Gridap can infer return type
#     d_typed(x::VectorValue{D, Float64}) where D = d(x)
#     n_typed(x::VectorValue{D, Float64}) where D = n(x)
#     dcf = CellField(d_typed, trian)
#     ncf = CellField(n_typed, trian)
#     return dcf, ncf
# end

function _make_cellfields(trian::Triangulation, f1::Function, f2::Function)
    _make_cellfield(trian, f1), _make_cellfield(trian, f2)
end

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

# --- Shift on edges
function _a_shift_edge(dE⁰::Measure, nE⁰::SkeletonPair,
                  dist_edg::DistanceData, α::CellField)
    d = dist_edg.d
    n = dist_edg.n
    (ϕ, v) -> ∫(jump(nE⁰ * (_w_α ∘ (α, v))) ⋅
                ((((mean(∇∇(ϕ)) ⋅ d) + mean(∇(ϕ))) ⋅ n) * n - mean(∇(ϕ))))dE⁰
end


# function _a_ghost(dE⁰::Measure, nE⁰::SkeletonPair, n::Function, d::Function, α::CellField)
#     dcf, ncf = _make_cellfields(dE⁰.quad.trian, d, n)
#     (ϕ, v) -> ∫(jump(nE⁰*(_w_α∘(α, v)))⋅((_s∘(∇(ϕ).⁺, ∇∇(ϕ).⁺, dcf, ncf)) +
#                                             (_s∘(∇(ϕ).⁻, ∇∇(ϕ).⁻, dcf, ncf)))*0.5)dE⁰
# end

# function _a_ghost(dE⁰::Measure, nE⁰::SkeletonPair, n::Tuple, d::Tuple, α::CellField)
#     (ϕ, v) -> ∫(jump(nE⁰*(_w_α∘(α, v)))⋅
#                 ((((mean(∇∇(ϕ))⋅d[2]) + mean(∇(ϕ)))⋅n[2])*n[2] - mean(∇(ϕ))))dE⁰
# end

# --- Boundary (SBM/WSBM) ---
# function _a_boundary(dΓ₁::Measure, nΓ₁::CellField, n::Function, d::Function)
#     dcf, ncf = _make_cellfields(dΓ₁.quad.trian, d, n)
#     (ϕ, v) -> ∫(nΓ₁⋅(_s∘(∇(ϕ), ∇∇(ϕ), dcf, ncf))*v)dΓ₁
# end

# function _a_boundary(dΓ₁::Measure, nΓ₁::CellField, n::Tuple, d::Tuple)
#     (ϕ, v) -> ∫((nΓ₁⋅((((d[1]⋅∇∇(ϕ)) + ∇(ϕ))⋅n[1])*n[1] - ∇(ϕ)))*v)dΓ₁
# end

# # function _a_boundary(dΓ₁::Measure, nΓ₁::CellField, n::Function, d::Function, α::CellField)
# #     dcf, ncf = _make_cellfields(dΓ₁.quad.trian, d, n)
# #     (ϕ, v) -> ∫(nΓ₁⋅(_s∘(∇(ϕ), ∇∇(ϕ), dcf, ncf))*(_w_α∘(α, v)))dΓ₁
# # end

# function _a_boundary(dΓ₁::Measure, nΓ₁::CellField, n::Tuple, d::Tuple, α::CellField)
#     (ϕ, v) -> ∫((nΓ₁⋅((((d[1]⋅∇∇(ϕ)) + ∇(ϕ))⋅n[1])*n[1] - ∇(ϕ)))*(_w_α∘(α, v)))dΓ₁
# end

function _a_boundary(dΓ₁::Measure, nΓ₁::CellField, dist::DistanceData)
    d = dist.d
    n = dist.n
    (ϕ, v) -> ∫((nΓ₁ ⋅ ((((d ⋅ ∇∇(ϕ)) + ∇(ϕ)) ⋅ n) * n - ∇(ϕ))) * v)dΓ₁
end

function _a_boundary(dΓ₁::Measure, nΓ₁::CellField, dist::DistanceData, α::CellField)
    d = dist.d
    n = dist.n
    (ϕ, v) -> ∫((nΓ₁ ⋅ ((((d ⋅ ∇∇(ϕ)) + ∇(ϕ)) ⋅ n) * n - ∇(ϕ))) * (_w_α ∘ (α, v)))dΓ₁
end

# ===================================================
# Linear Forms (RHS)
# ===================================================

# AGFEM / CUTFEM — no shifting
function _l_standard(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
                     dΓ₂::Measure, nΓ₂::CellField, f₁::Function, f₂::Function)
    v -> ∫(f₁ * v)dΩ + ∫((nΓ₁ ⋅ f₂) * v)dΓ₁ + ∫((nΓ₂ ⋅ f₂) * v)dΓ₂
end

# SBM — analytical
# function _l_sbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
#                 dΓ₂::Measure, nΓ₂::CellField,
#                 n::Function, f₁::Function, f₂::Function, f₂sbm::Function)
#     f1cf      = _make_cellfield(dΩ.quad.trian,  f₁)
#     f2cf      = _make_cellfield(dΓ₂.quad.trian, f₂)
#     ncf₁, fsbmcf₁ = _make_cellfields(dΓ₁.quad.trian, n, f₂sbm)
#     v -> ∫(f1cf*v)dΩ + ∫((nΓ₁*v)⋅_sᵣ(fsbmcf₁, ncf₁))dΓ₁ + ∫((nΓ₂⋅f2cf)*v)dΓ₂
# end

# function _l_sbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
#                 dΓ₂::Measure, nΓ₂::CellField,
#                 n::Function, f₁::Function, f₂::Function, f₂sbm::Function)
#     v -> ∫(f₁ * v)dΩ + ∫((nΓ₁ * v) ⋅ _sᵣ(f₂sbm, n))dΓ₁ + ∫((nΓ₂ ⋅ f₂) * v)dΓ₂
# end

# SBM — STL
# function _l_sbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
#                 dΓ₂::Measure, nΓ₂::CellField,
#                 n::Tuple, f₁::Function, f₂::Function, f₂sbm::Tuple)
#     # f1cf = _make_cellfield(dΩ.quad.trian,  f₁)
#     # f2cf = _make_cellfield(dΓ₂.quad.trian, f₂)
#     v -> ∫(f₁*v)dΩ + ∫((nΓ₁⋅((f₂sbm[1]⋅n[1])*n[1]))*v)dΓ₁ + ∫((nΓ₂⋅f₂)*v)dΓ₂
# end

# # WSBM — analytical
# function _l_wsbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
#                  dE⁰::Measure, nE⁰::SkeletonPair,
#                  dΓ₂::Measure, nΓ₂::CellField,
#                  n::Function, α::CellField, f₁::Function, f₂::Function, f₂sbm::Function)
#     f1cf           = _make_cellfield(dΩ.quad.trian,  f₁)
#     f2cf           = _make_cellfield(dΓ₂.quad.trian, f₂)
#     ncf₁, fsbmcf₁  = _make_cellfields(dΓ₁.quad.trian, n, f₂sbm)
#     ncfₑ, fsbmcfₑ  = _make_cellfields(dE⁰.quad.trian, n, f₂sbm)
#     v -> ∫(f1cf*(_w_α∘(α, v)))dΩ +
#          ∫((nΓ₁*(_w_α∘(α, v)))⋅_sᵣ(fsbmcf₁, ncf₁))dΓ₁ +
#          ∫(jump(nE⁰*(_w_α∘(α, v)))⋅_sᵣ(fsbmcfₑ, ncfₑ))dE⁰ +
#          ∫((nΓ₂⋅f2cf)*(_w_α∘(α, v)))dΓ₂
# end

# WSBM — STL
# function _l_wsbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
#                  dE⁰::Measure, nE⁰::SkeletonPair,
#                  dΓ₂::Measure, nΓ₂::CellField,
#                  n::Tuple, α::CellField, f₁::Function, f₂::Function, f₂sbm::Tuple)
#     # f1cf = _make_cellfield(dΩ.quad.trian,  f₁)
#     # f2cf = _make_cellfield(dΓ₂.quad.trian, f₂)
#     v -> ∫(f₁*(_w_α∘(α, v)))dΩ +
#          ∫((nΓ₁*(_w_α∘(α, v)))⋅((f₂sbm[1]⋅n[1])*n[1]))dΓ₁ +
#          ∫(jump(nE⁰*(_w_α∘(α, v)))⋅((f₂sbm[2]⋅n[2])*n[2]))dE⁰ +
#          ∫((nΓ₂⋅f₂)*(_w_α∘(α, v)))dΓ₂
# end

function _l_sbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
                dΓ₂::Measure, nΓ₂::CellField,
                dist::DistanceData, f₁::Function, f₂::Function)
    v -> ∫(f₁ * v)dΩ +
         ∫((nΓ₁ * v) ⋅ dist.fsbm)dΓ₁ +
         ∫((nΓ₂ ⋅ f₂) * v)dΓ₂
end

function _l_wsbm(dΩ::Measure, dΓ₁::Measure, nΓ₁::CellField,
                 dE⁰::Measure, nE⁰::SkeletonPair,
                 dΓ₂::Measure, nΓ₂::CellField,
                 dist::DistanceData, α::CellField,
                 f₁::Function, f₂::Function)
    v -> ∫(f₁ * (_w_α ∘ (α, v)))dΩ +
         ∫((nΓ₁ * (_w_α ∘ (α, v))) ⋅ dist.boundary.fsbm)dΓ₁ +
         ∫(jump(nE⁰ * (_w_α ∘ (α, v))) ⋅ dist.edges.fsbm)dE⁰ +
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
# function build_weak_form(::SBM, measures::Measures, domain::Domain,
#                          n, d, f₁::Function, f₂::Function, f₂sbm)
#     a = (interior = _a_interior(measures.dΩ⁻),
#          boundary = _a_boundary(measures.dΓ₁, domain.nΓ₁, n, d))
#     l = _l_sbm(measures.dΩ⁻, measures.dΓ₁, domain.nΓ₁,
#                measures.dΓ₂, domain.nΓ₂, n, f₁, f₂, f₂sbm)
#     WeakForm(a, l)
# end

function build_weak_form(::SBM, measures::Measures, domain::Domain,
                         dist::DistanceData, f₁::Function, f₂::Function)
    a = (interior = _a_interior(measures.dΩ⁻),
         boundary = _a_boundary(measures.dΓ₁, domain.nΓ₁, dist))
    l = _l_sbm(measures.dΩ⁻, measures.dΓ₁, domain.nΓ₁,
               measures.dΓ₂, domain.nΓ₂, dist, f₁, f₂)
    WeakForm(a, l)
end

"""
    build_weak_form(::WSBM, measures, domain, n, d, α, h, γg, order, f₁, f₂, f₂sbm) -> WeakForm

Build weak form for WSBM. Returns interior + boundary shift + ghost bilinear forms and weighted shifted RHS.
"""
# function build_weak_form(::WSBM, measures::Measures, domain::Domain,
#                          n, d, α::CellField, h::Float64, γg::Float64, order::Int64,
#                          f₁::Function, f₂::Function, f₂sbm)
#     a = (interior = _a_interior(measures.dΩ⁻, _get_wsbm_measures(domain)..., α),
#          boundary = _a_boundary(measures.dΓ₁, domain.nΓ₁, n, d, α),
#          ghost    = _a_ghost(measures.dE⁰, domain.nE⁰, n, d, α))
#     l = _l_wsbm(measures.dΩ⁻, measures.dΓ₁, domain.nΓ₁,
#                 measures.dE⁰, domain.nE⁰,
#                 measures.dΓ₂, domain.nΓ₂, n, α, f₁, f₂, f₂sbm)
#     WeakForm(a, l)
# end

function build_weak_form(::WSBM, measures::Measures, domain::Domain,
                         dist::NamedTuple, α::CellField,
                         h::Float64, γg::Float64, order::Int64,
                         f₁::Function, f₂::Function)
    a = (interior = _a_interior(measures.dΩ⁻, _get_wsbm_measures(domain)..., α),
         boundary = _a_boundary(measures.dΓ₁, domain.nΓ₁, dist.boundary, α),
         ghost    = _a_ghost(measures.dE⁰, domain.nE⁰, h, γg, Val(order)),
         shift_edge = _a_shift_edge(measures.dE⁰, domain.nE⁰, dist.edges, α))
    l = _l_wsbm(measures.dΩ⁻, measures.dΓ₁, domain.nΓ₁,
                measures.dE⁰, domain.nE⁰,
                measures.dΓ₂, domain.nΓ₂,
                dist, α, f₁, f₂)
    WeakForm(a, l)
end
