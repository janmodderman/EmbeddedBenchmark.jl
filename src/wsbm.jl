module ModWsbm
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

function wsbm(case::String, nₓ_vec::Vector;plot_flag=false, time_flag=false, vtk_flag=false)
method = "wsbm"
orders, (Lₓ, L₃, R), (g, k, ω, η₀), γg, (ls, to), folder = CaseSetup.parameters(method, case)

# start loops
l2s = []
cns = []
for order in orders
  degree = 2*order
  l2norms = Float64[]
  cnlist = Float64[]
  for nₓ in nₓ_vec
    h = Lₓ/nₓ
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
      Ωwsbm, Γ₁, nΓ₁, Γ₂, nΓ₂, E⁰, nE⁰ = CaseSetup.build_domain(method, cutgeo, cutgeo_facets)
    end

    # Constructing a reference sbm Interior (present in all 4 methods)
    Ωsbm, _, _, _, _ = CaseSetup.build_reference_domain(cutgeo)
    dΩsbm = Measure(Ωsbm, degree)

    # Constructing quadratures
    @timeit to "quadratures $order, $nₓ" begin
      dΩwsbm, dΓ₁, dΓ₂, dE⁰ = CaseSetup.set_measures(degree, Ωwsbm, Γ₁, Γ₂, E⁰)
    end

    # Constructing FE Spaces
    @timeit to "spaces $order, $nₓ" begin
      V, U = CaseSetup.set_spaces(order, Ωwsbm, ϕ₀)
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
        d, n, f₂sbm =CaseSetup.stl_distance(model,geo,Γ₁,dΓ₁,E⁰,dE⁰,f₂)
      end
    end # if

    @timeit to "volume_fraction $order, $nₓ" begin
      w_α = CaseSetup.volume_fraction(cutgeo, Ωwsbm)
    end

    # Constructing the matrices
    if case == "cylinder" || case == "sphere"
      @timeit to "affine $order, $nₓ" begin
        op = CaseSetup.build_operator(a(dΩwsbm,dΓ₁,nΓ₁,dE⁰,nE⁰,n,d,w_α,h,γg,order),l(dΩwsbm,dΓ₁,nΓ₁,dE⁰,nE⁰,dΓ₂,nΓ₂,n,w_α,f₁,f₂,f₂sbm,Γ₁,E⁰),U,V)
      end
    else
      @timeit to "affine $order, $nₓ" begin
        op = CaseSetup.build_operator(a(dΩwsbm,dΓ₁,nΓ₁,dE⁰,nE⁰,n,d,w_α,h,γg,order),l(dΩwsbm,dΓ₁,nΓ₁,dE⁰,nE⁰,dΓ₂,nΓ₂,n,w_α,f₁,f₂,f₂sbm),U,V)
      end
    end # if

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
      CaseSetup.write_results_omg(nₓ, order, ϕₕ ,ϕ₀ , Ωwsbm, Ωsbm;folder=folder)
      CaseSetup.write_results_gam(nₓ, order, Γ₁, nΓ₁, Γ₂, nΓ₂;folder=folder)
    end

  end # for
  push!(l2s, l2norms)
  push!(cns, cnlist)
end # for

if plot_flag
  plt = plot(legend=:bottomleft)
  CaseSetup.plot_L2(nₓ_vec, orders, l2s;marker=:heptagon, title=method*" "*case*" L2 norm error")
  display(plt)

  plt2 = plot(legend=:bottomleft)
  CaseSetup.plot_cond(nₓ_vec, orders, cns;marker=:heptagon, title=method*" "*case*" condition number")
  display(plt2)
end # if

if time_flag
  show(to)
end # if
  return nₓ_vec, orders, l2s, cns, to
end # function

end # module
