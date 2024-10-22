module ModSbm
using Gridap
using Plots
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra
using STLCutters
include("../src/case_setup.jl")

function sbm(case::String, nₓ_vec::Vector;plot_flag=false, time_flag=false, vtk_flag=false)
method = "sbm"
orders, (Lₓ, L₃, R), (g, k, ω, η₀), _, (ls, to), folder = CaseSetup.parameters(method, case)

# start loops
l2s = []
cns = []
for order in orders
  degree = 2*order
  l2norms = Float64[]
  cnlist = Float64[]
  for nₓ in nₓ_vec
    # Setting up the model domain and MMS
    if case == "cylinder"
      @timeit to "model $order, $nₓ" begin
        model, _, ϕ₀, f₁, f₂ = CaseSetup.setup_model_2d(nₓ;Lₓ=Lₓ,L₃=L₃,func_args=[g,k,η₀,ω])
      end
    else
      @timeit to "model $order, $nₓ" begin
        model, _, ϕ₀, f₁, f₂ = CaseSetup.setup_model_3d(nₓ;Lₓ=Lₓ,L₃=L₃,func_args=[g,k,η₀,ω])
      end
    end # if

    # Cutting the model domain
    if case == "cylinder" || case == "sphere"
      @timeit to "cutting $order, $nₓ" begin
        cutgeo, cutgeo_facets = CaseSetup.geometry_cut(model;Lₓ=Lₓ, L₃=L₃, R=R)
      end
      geo = get_geometry(cutgeo)
    else
      if case == "sphere_stl"
        geo = STLGeometry("data/meshes/sphere.stl")
      else
        geo = STLGeometry("data/meshes/bunnylow.stl")
      end # if
      @timeit to "cutting $order, $nₓ" begin
        cutgeo, cutgeo_facets = CaseSetup.geometry_cut(model, geo)
      end
    end # if

    # Constructing the Interior and Boundaries
    @timeit to "domain $order, $nₓ" begin
      Ωsbm, Γ₁, nΓ₁, Γ₂, nΓ₂ = CaseSetup.build_domain(method, cutgeo, cutgeo_facets)
    end

    # Constructing quadratures
    @timeit to "quadratures $order, $nₓ" begin
      dΩsbm, dΓ₁, dΓ₂ = CaseSetup.set_measures(degree, Ωsbm, Γ₁, Γ₂)
    end

    # Constructing FE Spaces
    @timeit to "spaces $order, $nₓ" begin
      V, U = CaseSetup.set_spaces(order, Ωsbm, ϕ₀)
    end

    # Constructing weak form
    if case == "cylinder" || case == "sphere"
      @timeit to "weak_form $order, $nₓ" begin
        a, l = CaseSetup.weak_form(method)
      end
    else
      @timeit to "weak_form $order, $nₓ" begin
        a, l = CaseSetup.weak_form(method;stl_flag=true)
      end
    end # if

    # Constructing normal + distance + shifted functions
    if case == "cylinder" || case == "sphere"
      @timeit to "distances $order, $nₓ" begin
        d, n, f₂sbm = CaseSetup.analytical_distance(model,Lₓ,L₃,R,f₂)
      end
    else
      @timeit to "distances $order, $nₓ" begin
        d, n, f₂sbm = CaseSetup.stl_distance(model,geo,Γ₁,dΓ₁,f₂)
      end
    end # if

    # Constructing the matrices
    @timeit to "affine $order, $nₓ" begin
      op = CaseSetup.build_operator(a(dΩsbm,dΓ₁,nΓ₁,n,d),l(dΩsbm,dΓ₁,nΓ₁,dΓ₂,nΓ₂,n,f₁,f₂,f₂sbm),U,V)
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
    if vtk_flag
      CaseSetup.write_results_omg(nₓ, order, ϕₕ, ϕ₀, Ωsbm;folder=folder)
      CaseSetup.write_results_gam(nₓ, order, Γ₁, nΓ₁, Γ₂, nΓ₂;folder=folder)
    end
    # writevtk(Γ₁,"test",cellfields=["n"=>CellField(n(0),Γ₁),"d"=>CellField(d(0),Γ₁),"f2sbm"=>CellField(f₂sbm(0),Γ₁)])
  end # for
  push!(l2s, l2norms)
  push!(cns, cnlist)
end # for

if plot_flag
plt = plot(legend=:bottomleft)
CaseSetup.plot_L2(nₓ_vec, orders, l2s;marker=:xcross, title=method*" "*case*" L2 norm error")
display(plt)

plt2 = plot(legend=:bottomleft)
CaseSetup.plot_cond(nₓ_vec, orders, cns;marker=:xcross, title=method*" "*case*" condition number")
display(plt2)
end # if

if time_flag
  show(to)
end # if
  return nₓ_vec, orders, l2s, cns, to
end # function

end # module
