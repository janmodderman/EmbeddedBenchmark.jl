module Manufactured2Dagfem
using Gridap
using Plots
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra
using CSV
include("casesetup.jl")

# function MMSagfem()
folder = "data/sims/manufactured/agfem/"
to = TimerOutput()


# Parameters
Lₓ = 1.0
d = 1.0
g = 9.81
k = 2π/d
R = 0.25
ω = sqrt(g*k*tanh(k*d))
η₀ = 0.1
method = "agfem"
# order = 1
ls = LUSolver()
# nₓ = 8
nₓ_vec = [8,16,32,64]
orders = [1,2]
l2s = []
cns = []

for order in orders
  degree = 2*order
  l2norms = Float64[]
  cnlist = Float64[]

  for nₓ in nₓ_vec

    # Setting up the model domain and MMS
    @timeit to "model $order, $nₓ" begin
    model, labels, ϕ₀, f₁, f₂ = CaseSetup.setup_model_2d(nₓ;Lₓ=Lₓ,L₃=d,func_args=[g,k,η₀,ω])
    end

    # Cutting the model domain
    @timeit to "cutting $order, $nₓ" begin
    cutgeo, geo, cutgeo_facets = CaseSetup.geometry_cut(model;Lₓ=Lₓ, L₃=d, R=R)
    end

    # Constructing the Interior and Boundaries
    @timeit to "domain $order, $nₓ" begin
    Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂ = CaseSetup.build_domain(method, cutgeo, cutgeo_facets, geo, model)
    end

    # Constructing a reference sbm Interior (present in all 4 methods)
    Ωsbm, _, _, _, _ = CaseSetup._build_domain_sbm(cutgeo, geo, model)
    dΩsbm = Measure(Ωsbm, degree)

    # Constructing quadratures
    @timeit to "quadratures $order, $nₓ" begin
    dΩ⁻, dΓ₁, dΓ₂ = CaseSetup.set_measures(degree, Ω⁻, Γ₁, Γ₂)
    end

    # Constructing FE Spaces
    @timeit to "spaces $order, $nₓ" begin
    V, U = CaseSetup.set_spaces(order, Ω⁻act, ϕ₀, cutgeo, geo)
    end

    # Constructing weak form
    @timeit to "weak_form $order, $nₓ" begin
    a, l = CaseSetup.weak_form(method)
    end

    # Constructing the matrices
    @timeit to "affine $order, $nₓ" begin
    op = CaseSetup.build_operator(a(dΩ⁻),l(dΩ⁻,dΓ₁,nΓ₁,dΓ₂,nΓ₂,f₁,f₂),U,V)
    end

    # Calculating L1 norm condition number 
    push!(cnlist, CaseSetup.get_cond(op))

    # Solve the problem
    @timeit to "solving $order, $nₓ" begin
    ϕₕ = solve(ls, op)
    end

    # Calculate L2 norm on the method domain and on the sbm domain
    # l2norm = CaseSetup.L2norm(dΩ⁻,ϕₕ,ϕ₀(0.0))
    l2norm_sbm = CaseSetup.L2norm(dΩsbm,ϕₕ,ϕ₀(0.0))
    push!(l2norms,l2norm_sbm)

    # Writing results to vtk
    CaseSetup.write_results_omg(nₓ, order, ϕₕ ,ϕ₀ , Ω⁻, Ωsbm;folder=folder)

  end # for
  push!(l2s, l2norms)
  push!(cns, cnlist)
end # for

plt = plot(legend=:bottomleft)
CaseSetup.plot_L2(nₓ_vec, orders, l2s; title="AgFEM 2D Cylinder L2 norm error")
display(plt)

plt2 = plot(legend=:bottomleft)
CaseSetup.plot_cond(nₓ_vec, orders, cns; title="AgFEM 2D Cylinder condition number")
# display(plt2)
show(to)

end
