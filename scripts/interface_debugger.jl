module debug_interface
using Gridap
using GridapEmbedded
using LinearAlgebra
include("../src/CaseSetup.jl")

(nₓ_vec, orders), (Lₓ, L₃, R), (g, k, ω, η₀), _, (ls, to), folder = CaseSetup.parameters(method, case)
nₓ = 8
order = 1
method = "sbm"
model, _, ϕ₀, f₁, f₂ = CaseSetup.setup_model_2d(nₓ;Lₓ=Lₓ,L₃=L₃,func_args=[g,k,η₀,ω])
cutgeo, geo, cutgeo_facets = CaseSetup.geometry_cut(model;Lₓ=Lₓ, L₃=L₃, R=R)
Ωsbm, Γ₁, nΓ₁, Γ₂, nΓ₂ = CaseSetup.build_domain(method, cutgeo, cutgeo_facets, geo, model)
d, n, f₂sbm = CaseSetup.analytical_distance(model,Lₓ,L₃,R,f₂)




CaseSetup.write_results_omg(nₓ, order, ϕₕ ,ϕ₀ , Ωwsbm, Ωsbm;folder=folder)
writevtk(Γ₁,"test",cellfields=["n"=>CellField(n(0),Γ₁),"d"=>CellField(d(0),Γ₁),"f2sbm"=>CellField(f₂sbm(0),Γ₁)])
writevtk(E⁰,"teste",cellfields=["n"=>CellField(n(0),E⁰),"d"=>CellField(d(0),E⁰),"f2sbm"=>CellField(f₂sbm(0),E⁰),"normsur+"=>CellField(nE⁰.⁺,E⁰),"normsur-"=>CellField(nE⁰.⁻,E⁰)])
CaseSetup.write_results_gam(nₓ, order, Γ₁, nΓ₁, Γ₂, nΓ₂;folder=folder)

end # module