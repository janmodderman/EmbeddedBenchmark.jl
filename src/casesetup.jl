module CaseSetup
using Gridap
using Plots
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra
using STLCutters

using GridapEmbedded.Interfaces
using GridapEmbedded.CSG

function parameters(method::String, case::String)
    # Number of elements & element orders
    nₓ = [8,16,32,64,128,256,512]
    orders = [1,2]

    # Geometrical parameters
    Lₓ = 1.0    # [m]: horizontal domain length(s)
    L₃ = 1.0    # [m]: vertical domain length
    R = 0.25    # [m]: cylinder/sphere radius

    # MMS parameters
    g = 9.81                    # [m/s²]: gravitational constant
    k = 2π/L₃                   # [rad/m]: wave number
    ω = sqrt(g*k*tanh(k*L₃))    # [rad/s]: dispersion relation ocean waves
    η₀ = 0.1                    # [m]: wave amplitude

    # Ghost Penalty parameters per order 
    γg = (0.1,0.1)

    # solving & timing variables
    ls = LUSolver()     # Lienar Solver
    to = TimerOutput()  # TimerOutputs variable
    
    # output folder
    folder = "data/sims/"*method*"/"*case*"/"

    return (nₓ, orders), (Lₓ, L₃, R), (g, k, ω, η₀), γg, (ls, to), folder
end # function

# FUNCTION THAT RETURNS THE CORRECT WEAK FORM FOR AGFEM, CUTFEM, SBM & WSBM FOR BOTH ANALYTICAL & STL GEOMETRIES
function weak_form(method::String; stl_flag=false)
    # =============================AGFEM=============================
    # Bilinear form
    a1(dΩ) = (ϕ,v) -> ∫(∇(ϕ)⋅∇(v))dΩ
    # Righthand side
    l1(dΩ,dΓ₁,nΓ₁,dΓ₂,nΓ₂,f₁,f₂) = v -> ∫(f₁(0)*v)dΩ + ∫((nΓ₁⋅f₂(0))*v)dΓ₁ + ∫((nΓ₂⋅f₂(0))*v)dΓ₂
    # =============================CUTFEM=============================
    # Bilinear form
    a2(dΩ,dE⁰,nE⁰,h,γg,order) = (ϕ,v) -> ∫(∇(ϕ)⋅∇(v))dΩ + 
                                ∫((γg[1]*(h^3))*jump(nE⁰⋅∇(v))⊙jump(nE⁰⋅∇(ϕ)))dE⁰ +  # GP stabilization on gradients first order
                                ∫((order>1)*(γg[2]*(h^5))*jump(nE⁰⋅∇∇(v))⊙jump(nE⁰⋅∇∇(ϕ)))dE⁰ # GP stabilization on gradients second order
    # Righthand side
    l2(dΩ,dΓ₁,nΓ₁,dΓ₂,nΓ₂,f₁,f₂) = v -> ∫(f₁(0)*v)dΩ + ∫((nΓ₁⋅f₂(0))*v)dΓ₁ + ∫((nΓ₂⋅f₂(0))*v)dΓ₂
    # =============================SBM=============================
    # Bilinear form (ANALYTICAL)
    a31(dΩ,dΓ₁,nΓ₁,n,d) = (ϕ,v) -> ∫(∇(ϕ)⋅∇(v))dΩ + 
                                ∫((nΓ₁⋅((((d(0)⋅∇∇(ϕ)) + ∇(ϕ))⋅n(0))*n(0) - ∇(ϕ)))*v)dΓ₁
    # Righthand side (ANALYTICAL)
    l31(dΩ,dΓ₁,nΓ₁,dΓ₂,nΓ₂,n,f₁,f₂,f₂sbm) = v -> ∫(f₁(0)*v)dΩ + ∫((nΓ₁⋅(((v*f₂sbm(0))⋅n(0))*n(0))))dΓ₁ + ∫((nΓ₂⋅f₂(0))*v)dΓ₂ 
    # =============================================================
    # Bilinear form (STL)
    a32(dΩ,dΓ₁,nΓ₁,n,d) = (ϕ,v) -> ∫(∇(ϕ)⋅∇(v))dΩ + 
                                ∫((nΓ₁⋅((((d[1]⋅∇∇(ϕ)) + ∇(ϕ))⋅n[1])*n[1] - ∇(ϕ)))*v)dΓ₁
    # Righthand side (STL)
    l32(dΩ,dΓ₁,nΓ₁,dΓ₂,nΓ₂,n,f₁,f₂,f₂sbm) = v -> ∫(f₁(0)*v)dΩ + ∫((nΓ₁⋅((f₂sbm[1]⋅n[1])*n[1]))*v)dΓ₁ + ∫((nΓ₂⋅f₂(0))*v)dΓ₂ 
    # =============================WSBM=============================
    # Bilinear form (ANALYTICAL)
    a41(dΩ,dΓ₁,nΓ₁,dE⁰,nE⁰,n,d,w_α,h,γg,order) = (ϕ,v) -> ∫(∇(ϕ)⋅w_α(∇(v)))dΩ + 
                                            ∫(((nΓ₁*w_α(v))⊙(((∇∇(ϕ)⋅d(0) + ∇(ϕ))⋅n(0)*n(0)) - ∇(ϕ))))dΓ₁ +
                                            ∫(jump(nE⁰*w_α(v))⋅mean(((∇∇(ϕ)⋅d(0) + ∇(ϕ))⋅n(0))*n(0) - ∇(ϕ)))dE⁰ +
                                            ∫((γg[1]*(h^3))*jump(nE⁰⋅∇(v))⊙jump(nE⁰⋅∇(ϕ)))dE⁰ +  # GP stabilization on gradients first order
                                            ∫((order>1)*(γg[2]*(h^5))*jump(nE⁰⋅∇∇(v))⊙jump(nE⁰⋅∇∇(ϕ)))dE⁰ # GP stabilization on gradients second order
    # Righthand side (ANALYTICAL)
    l41(dΩ,dΓ₁,nΓ₁,dE⁰,nE⁰,dΓ₂,nΓ₂,n,w_α,f₁,f₂,f₂sbm,Γ₁,E⁰) = v -> ∫(f₁(0)*w_α(v))dΩ + ∫( ((nΓ₁*w_α(v))⋅n(0))*((CellField(f₂sbm(0),Γ₁)⋅n(0))) )dΓ₁ + ∫( (jump(nE⁰*w_α(v))⋅n(0))*(CellField(f₂sbm(0),E⁰)⋅n(0)))dE⁰ + ∫((nΓ₂⋅f₂(0))*w_α(v))dΓ₂  
    # ==============================================================
    # Bilinear form (STL)
    a42(dΩ,dΓ₁,nΓ₁,dE⁰,nE⁰,n,d,w_α,h,γg,order) = (ϕ,v) -> ∫(∇(ϕ)⋅w_α(∇(v)))dΩ + 
                                            ∫(((nΓ₁*w_α(v))⊙(((∇∇(ϕ)⋅d[1] + ∇(ϕ))⋅n[1]*[1]) - ∇(ϕ))))dΓ₁ +
                                            ∫(jump(nE⁰*w_α(v))⋅mean(((∇∇(ϕ)⋅d[2] + ∇(ϕ))⋅n[2])*n[2] - ∇(ϕ)))dE⁰ +
                                            ∫((γg[1]*(h^3))*jump(nE⁰⋅∇(v))⊙jump(nE⁰⋅∇(ϕ)))dE⁰ +  # GP stabilization on gradients first order
                                            ∫((order>1)*(γg[2]*(h^5))*jump(nE⁰⋅∇∇(v))⊙jump(nE⁰⋅∇∇(ϕ)))dE⁰ # GP stabilization on gradients second order
    # Righthand side (STL)
    l42(dΩ,dΓ₁,nΓ₁,dE⁰,nE⁰,dΓ₂,nΓ₂,n,w_α,f₁,f₂,f₂sbm) = v -> ∫(f₁(0)*w_α(v))dΩ + ∫( (nΓ₁*w_α(v))⋅((f₂sbm[1]⋅n[1])*n[1]) )dΓ₁ + ∫( jump(nE⁰*w_α(v))⋅(f₂sbm[2]⋅n[2])*n[2])dE⁰ + ∫((nΓ₂⋅f₂(0))*w_α(v))dΓ₂ 
    # ==============================END==============================
    if method == "agfem"
        return a1, l1
    elseif method == "cutfem"
        return a2, l2
    elseif method == "sbm"
        if !stl_flag
            return a31, l31
        else
            return a32, l32
        end # if
    elseif method == "wsbm"
        if !stl_flag
            return a41, l41
        else
            return a42, l42
        end # if
    else
        println("Method unsupported")
        return
    end # if

end # function

# FUNCTIONS TO SET UP THE DOMAIN AND METHOD OF MANUFACTURED SOLUTIONS IN 2D & 3D
function setup_model_3d(nₓ::Int64;Lₓ=0.5, L₃=0.25, func_args=[])
    # MODEL
    domain = (-Lₓ/2,Lₓ/2,-Lₓ/2,Lₓ/2,-L₃,0.0)
    model = CartesianDiscreteModel(domain,(nₓ,nₓ,nₓ/2))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "top", [22])
    add_tag_from_tags!(labels, "DC", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26])

    # ANALYTICAL FUNCTION DEFINITIONS
    g,k,η₀,ω = func_args
    ϕ₀(x,t) = η₀*g/ω*(cosh(k*(x[3]+L₃))/cosh(k*L₃))*sin(k*x[1] + k*x[2] -ω*t)
    ϕ₀(t) = x -> ϕ₀(x,t)
    f₁(t) = x->-tr(∇∇(ϕ₀(t))(x))
    f₂(x,t) = VectorValue(ω*η₀*(cosh(k*(x[3]+L₃))/sinh(k*L₃))*cos(k*x[1] + k*x[2] -ω*t),
                       ω*η₀*(cosh(k*(x[3]+L₃))/sinh(k*L₃))*cos(k*x[1] + k*x[2] -ω*t),
                        ω*η₀*(sinh(k*(x[3]+L₃))/sinh(k*L₃))*sin(k*x[1] + k*x[2] -ω*t))
    f₂(t) = x -> f₂(x,t)

    return model, labels, ϕ₀, f₁, f₂
end # function

function setup_model_2d(nₓ::Int64;Lₓ=0.5, L₃=0.25, func_args=[])
    # MODEL
    domain = (-Lₓ/2,Lₓ/2,-L₃,0.0)
    model = CartesianDiscreteModel(domain,(nₓ,nₓ))
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "top", [6])
    add_tag_from_tags!(labels, "DT", [1,2,3,4,5,7,8]) # Dirichlet Tags

    # ANALYTICAL FUNCTION DEFINITIONS
    g,k,η₀,ω = func_args
    ϕ₀(x,t) = η₀*g/ω*(cosh(k*(x[2]+L₃))/cosh(k*L₃))*sin(k*x[1] -ω*t)
    ϕ₀(t) = x -> ϕ₀(x,t)
    f₁(t) = x->-tr(∇∇(ϕ₀(t))(x))
    f₂(x,t) = VectorValue(ω*η₀*(cosh(k*(x[2]+L₃))/sinh(k*L₃))*cos(k*x[1] -ω*t),
                        ω*η₀*(sinh(k*(x[2]+L₃))/sinh(k*L₃))*sin(k*x[1] -ω*t))
    f₂(t) = x -> f₂(x,t)

    return model, labels, ϕ₀, f₁, f₂
end # function

# DOMAIN BUILDERS FOR AGFEM, CUTFEM, SBM & WSBM
function build_domain(method::String, cutgeo::EmbeddedDiscretization, cutgeo_facets::EmbeddedFacetDiscretization, geo::Geometry, model::DiscreteModel)
    if method == "agfem"
        return _build_domain_agfem(cutgeo, cutgeo_facets)
    elseif method == "cutfem"
        return _build_domain_cutfem(cutgeo, cutgeo_facets)
    elseif method == "sbm"
        return _build_domain_sbm(cutgeo, geo, model)
    elseif method == "wsbm"
        return _build_domain_wsbm(cutgeo, geo, model)
    else
        println("Method unsupported")
        return
    end # if
end # function

function _build_domain_agfem(cutgeo::EmbeddedDiscretization, cutgeo_facets::EmbeddedFacetDiscretization)
    Ω⁻ = Interior(cutgeo, PHYSICAL_OUT)
    Ω⁻act = Interior(cutgeo, ACTIVE_OUT)
    Γ₁ = EmbeddedBoundary(cutgeo)
    nΓ₁ = -get_normal_vector(Γ₁)
    Γ₂ = BoundaryTriangulation(cutgeo_facets, PHYSICAL_OUT, tags=["top"])
    nΓ₂ = get_normal_vector(Γ₂)
    return Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂
end # function

function _build_domain_cutfem(cutgeo::EmbeddedDiscretization, cutgeo_facets::EmbeddedFacetDiscretization)
    Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂ = _build_domain_agfem(cutgeo, cutgeo_facets)
    E⁰ = GhostSkeleton(cutgeo, ACTIVE_OUT)
    nE⁰ = get_normal_vector(E⁰)
    return Ω⁻, Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂, E⁰, nE⁰
end # function

# DEV NOTE: FOR SOME REASON THE PROBLEM DOESNT SOLVE CORRECTLY IF WE DO AN INTERFACE BETWEEN 2 SEPARATE DOMAINS, ALTHOUGH IT RETURNS A CORRECT BOUNDARYTRIANGULATION, INTERFACE WITH INTERIOR(MODEL) DOES WORK THOUGH
function _build_domain_sbm(cutgeo::EmbeddedDiscretization, geo::Geometry, model::DiscreteModel; var=[1])
    Ω⁻act = Interior(cutgeo, OUT)
    Ω⁻pas = Interior(cutgeo, ACTIVE_IN)
    Γ₁ = Interface(Ω⁻pas,Ω⁻act).⁻     # surrogate boundary (interior tags on the boundary)
    nΓ₁ = get_normal_vector(Γ₁)
    Γ₂ = BoundaryTriangulation(Ω⁻act, tags=["top"])
    nΓ₂ = get_normal_vector(Γ₂)
    return Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂
end # function

function _build_domain_wsbm(cutgeo::EmbeddedDiscretization, geo::Geometry, model::DiscreteModel)
    Ω⁻act = Interior(cutgeo, ACTIVE_OUT)
    Ω⁻pas = Interior(cutgeo, IN)
    Γ₁ = Interface(Ω⁻pas, Ω⁻act).⁻
    nΓ₁ = get_normal_vector(Γ₁)
    Γ₂ = BoundaryTriangulation(Ω⁻act, tags=["top"])
    nΓ₂ = get_normal_vector(Γ₂)
    E⁰ = GhostSkeleton(cutgeo, ACTIVE_OUT)
    nE⁰ = get_normal_vector(E⁰)
    return Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂, E⁰, nE⁰
end # function

# ANALYTICAL DISTANCE FUNCTIONS FOR SBM & WSBM
function analytical_distance(model::DiscreteModel,Lₓ::Float64,L₃::Float64,R::Float64,fun::Function)
    ncd = num_cell_dims(model)
    if ncd < 3  # 2D: horizontal cylinder
        pmin = Point(-Lₓ/2, -L₃)
        pmax = Point(Lₓ/2, 0.0)
        pmid = 0.5*(pmax + pmin) + VectorValue(0.0, L₃/2) 
    else        # 3D: sphere
        pmin = Point(-Lₓ/2, -Lₓ/2, -L₃)
        pmax = Point(Lₓ/2, Lₓ/2, 0.0)
        pmid = 0.5*(pmax + pmin) + VectorValue(0.0, 0.0, L₃/2) 
    end # if
    D(x,t) = pmid - x
    absD(x,t) = sqrt(D(x,t)⋅D(x,t))
    dist(x,t) = absD(x,t) - R
    n(x,t) = (dist(x,t)>=0.0)*(D(x,t)./absD(x,t)) - (dist(x,t)<0.0)*(D(x,t)./absD(x,t))
    d(x,t) = abs(dist(x,t))*n(x,t)
    d(t) = x -> d(x,t)
    n(t) = x -> n(x,t)
    funsbm(t) = x -> fun(x + d(x,t),t)
    return d, n, funsbm
end # function

# STL DISTANCE FUNCTIONS FOR SBM & WSBM
function stl_distance(model::DiscreteModel, geo::STLGeometry, Γ₁::BoundaryTriangulation, dΓ₁::Measure)
    _, n, d, xd, _ = DistanceSTL.STLdistance(model, geo, Γ₁, dΓ₁)
    return (d), (n), (DistanceSTL.fshifted(xd, fun, dΓ₁ₒ))
end # function

function stl_distance(model::DiscreteModel, geo::STLGeometry, Γ₁::BoundaryTriangulation, dΓ₁::Measure, E⁰::SkeletonTriangulation, dE⁰::Measure, fun::Function)
    _, n₁, d₁, xd₁, _ = DistanceSTL.STLdistance(model, geo, Γ₁, dΓ₁)
    _, nₑ, dₑ, xdₑ, _ = DistanceSTL.STLdistance(model, geo, E⁰, dE⁰)
    n = (n₁, nₑ)
    d = (d₁, dₑ)
    funsbm = (DistanceSTL.fshifted(xd₁, fun, dΓ₁), DistanceSTL.fshifted(xdₑ, fun, dE⁰))
    return d, n, funsbm
end # function

# CUTTING FUNCTIONS FOR ANALYTICAL AND STL
function geometry_cut(model::DiscreteModel;Lₓ=0.5, L₃=0.5, R=0.25)
    ncd = num_cell_dims(model)
    if ncd < 3
        pmin = Point(-Lₓ/2, -L₃)
        pmax = Point(Lₓ/2, 0.0)
        pmid = 0.5*(pmax + pmin) + VectorValue(0.0, L₃/2) 
        geo = disk(R, x0=pmid)
    else            
        pmin = Point(-Lₓ/2, -Lₓ/2, -L₃)
        pmax = Point(Lₓ/2, Lₓ/2, 0.0)
        pmid = 0.5*(pmax + pmin) + VectorValue(0.0, 0.0, L₃/2) 
        geo = sphere(R, x0=pmid)
    end # if
    cutgeo = cut(model, geo)
    cutgeo_facets = cut_facets(model, geo)
    return cutgeo, geo, cutgeo_facets
end # function

function geometry_cut(model::DiscreteModel, geo::STLGeometry)
    cut(model, geo)
end

# SETTING UP THE MEASURES FOR AGFEM, CUTFEM, SBM & WSBM
function set_measures(degree::Int64, Ω::Triangulation, Γ₁::Triangulation, Γ₂::Triangulation)
    return Measure(Ω,degree), Measure(Γ₁,degree), Measure(Γ₂,degree)
end # function

function set_measures(degree::Int64, Ω::Triangulation, Γ₁::Triangulation, Γ₂::Triangulation, E⁰::SkeletonTriangulation)
    return Measure(Ω,degree), Measure(Γ₁,degree), Measure(Γ₂,degree), Measure(E⁰, degree)
end # function

# VOLUME FRACTION FOR WSBM
function volume_fraction(cutgeo::EmbeddedDiscretization, Ω⁻act::Triangulation)
    Ω⁻ = Interior(cutgeo, PHYSICAL_OUT)
    vol⁻ = get_cell_measure(Ω⁻,Ω⁻act)
    vol⁻act = get_cell_measure(Ω⁻act)
    γvol = vol⁻ ./ vol⁻act
    w_α(f) = γvol*f
    return w_α
end # function

# SETTING UP THE FINITE ELEMENT SPACES FOR AGFEM, CUTFEM, SBM & WSBM
function set_spaces(order::Int64, Ω⁻act::Triangulation, fun::Function)
    reffe = ReferenceFE(lagrangian, Float64, order)
    V = TestFESpace(Ω⁻act, reffe, dirichlet_tags=["DT"])
    U = TrialFESpace(V, fun(0.0))
    return V, U
end # function

function set_spaces(order::Int64, Ω⁻act::Triangulation, fun::Function, cutgeo::EmbeddedDiscretization, geo::Geometry)
    Vstd, _ = set_spaces(order::Int64, Ω⁻act::Triangulation, fun::Function)
    strategy = AggregateCutCellsByThreshold(1.0)
    aggregates = aggregate(strategy, cutgeo, geo, OUT)
    V = AgFEMSpace(Vstd, aggregates)
    U = TrialFESpace(V, fun(0.0))
    return V, U
end # function

# BUILDING THE OPERATORS
function build_operator(a::Function, l::Function, U::FESpace, V::FESpace)
    AffineFEOperator(a,l,U,V)
end # function

# GET THE CONDITION NUMBER IN L1 NORM
function get_cond(op::FEOperator)
    cond(get_matrix(op),1)
end # function

# CALCULATE L2 ERROR NORM
function L2norm(dΩ::Measure, ϕₕ::FEFunction, ϕ₀::Function)
    √(∑(∫((ϕₕ-ϕ₀)*(ϕₕ-ϕ₀))dΩ))
end # function

# PLOTTING FUNCTIONS
function plot_L2(nₓ::Vector, orders::Vector, l2::Vector;marker=:circle, title="")
    for order in orders
    plot!(nₓ,l2[order],xaxis=:log,yaxis=:log,marker=marker,label="order=$order",xlabel="nₓ",ylabel="L2 norm",title=title)
    end # for
    plot!(nₓ,0.05*nₓ.^(-1), xaxis=:log, yaxis=:log, labels="1st order", linestyle=:solid, color=:black)
    plot!(nₓ,0.05*nₓ.^(-2), xaxis=:log, yaxis=:log, labels="2nd order", linestyle=:dash, color=:black)
    plot!(nₓ,0.05*nₓ.^(-3), xaxis=:log, yaxis=:log, labels="3rd order", linestyle=:dot, color=:black)
end # function

function plot_cond(nₓ::Vector, orders::Vector, cond::Vector;marker=:circle, title="")
    for order in orders
    plot!(nₓ,cond[order],xaxis=:log,yaxis=:log,marker=marker,label="order=$order",xlabel="nₓ",ylabel="Cond L1",title=title)
    end # for
end # function

# WRITING TO VTK
function write_results_omg(nₓ::Int64, order::Int64, ϕₕ::FEFunction ,ϕ₀::Function, Ω::Triangulation, Ωsbm::Triangulation;folder="")
    writevtk(Ω,folder*"omg_$(nₓ)_$order",cellfields=["ϕ"=>ϕ₀(0.0),"ϕₕ"=>ϕₕ, "error"=>(ϕₕ-ϕ₀(0.0))])
    writevtk(Ωsbm,folder*"omgsbm_$(nₓ)_$order",cellfields=["ϕ"=>ϕ₀(0.0),"ϕₕ"=>ϕₕ, "error"=>(ϕₕ-ϕ₀(0.0))])
end # function

function write_results_omg(nₓ::Int64, order::Int64, ϕₕ::FEFunction ,ϕ₀::Function, Ω::Triangulation;folder="")
    writevtk(Ω,folder*"omg_$(nₓ)_$order",cellfields=["ϕ"=>ϕ₀(0.0),"ϕₕ"=>ϕₕ, "error"=>(ϕₕ-ϕ₀(0.0))])
end # function

function write_results_gam(nₓ::Int64, order::Int64, Γ₁::Triangulation, nΓ₁::CellField, Γ₂::Triangulation, nΓ₂::CellField;folder="")
    writevtk(Γ₁,folder*"gam1_$(nₓ)_$order",cellfields=["normal"=>CellField(nΓ₁, Γ₁)])
    writevtk(Γ₂,folder*"gam2_$(nₓ)_$order",cellfields=["normal"=>CellField(nΓ₂, Γ₂)])
end # function

# SAVING DATA IN .JLD2
function save_data(nₓ::Int64, order::Int64, V::FESpace, ϕₕ::FEFunction, ϕ₀::Function, L2norm ;folder="")
    sol0 = interpolate_everywhere(ϕ₀,V)
    wsave(folder*"sol_"*"$(nₓ)_$order"*".jld2",Dict("df" => vcat(DataFrame(Phi0=[sol0], Phi=[sol0], Error=[sol0-sol0],L2Norm = [L2norm]),
    DataFrame(Phi0=[sol0], Phi=[ϕₕ], Error=[ϕₕ-sol0],L2Norm = [L2norm]))))
end # function

# SHOW PRINT STATEMENTS
function print_steps()

end # function

end # module