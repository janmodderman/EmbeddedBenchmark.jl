module debug_interface
using Gridap
using GridapEmbedded
using LinearAlgebra
include("../src/CaseSetup.jl")

using GridapEmbedded.Interfaces
using GridapEmbedded.CSG

# function SBMSkeleton(cut::EmbeddedDiscretization,in_or_out,geo::CSG.Geometry)

#     # @notimplementedif in_or_out in (PHYSICAL_IN,PHYSICAL_OUT) "Not implemented but not needed in practice. Ghost stabilization can be integrated in full facets."
#     # @assert in_or_out in (ACTIVE_IN,ACTIVE_OUT) || in_or_out.in_or_out in (IN,OUT,CUT)
#     cell_to_inoutcut = compute_bgcell_to_inoutcut(cut,geo)
#     model = cut.bgmodel
#     topo = get_grid_topology(model)
#     D = num_cell_dims(model)
#     facet_to_cells = Table(get_faces(topo,D-1,D))
#     facet_to_mask = fill(false,length(facet_to_cells))
#     _fill_SBM_skeleton_mask!(facet_to_mask,facet_to_cells,cell_to_inoutcut,in_or_out.in_or_out)
  
#     SkeletonTriangulation(model,facet_to_mask)
#   end

# function _fill_SBM_skeleton_mask!(facet_to_mask,facet_to_cells::Table,cell_to_inoutcut,in_or_out)

#     nfacets = length(facet_to_cells)
#     for facet in 1:nfacets
#       a = facet_to_cells.ptrs[facet]
#       b = facet_to_cells.ptrs[facet+1]
#       ncells_around = b-a
#       ncells_around_cut = 0
#       ncells_around_active = 0
#       for cell_around in 1:ncells_around
#         cell = facet_to_cells.data[a-1+cell_around]
#         inoutcut = cell_to_inoutcut[cell]
#         # if (inoutcut == CUT)
#         #   ncells_around_cut += 1
#         # end
#         if (inoutcut == CUT) || (inoutcut == in_or_out)
#           ncells_around_active += 1
#         end
#       end
#       if (ncells_around_cut >0) && (ncells_around_active == 2)
#         facet_to_mask[facet] = true
#       end
#     end
# end

function _build_domain_sbm(cutgeo::EmbeddedDiscretization, cutgeo_facets::EmbeddedFacetDiscretization, geo::Geometry, model::DiscreteModel; var=[1])
    # Ω⁻act, Ω⁻pas = _build_helper(cutgeo, geo, model, var)
        Ω⁻act = Interior(cutgeo, OUT)
    Ω⁻pas = Interior(cutgeo, ACTIVE_IN)
    @show Γ₁ = Interface(Ω⁻pas,Ω⁻act).⁻     # surrogate boundary (interior tags on the boundary)
    @show Γ₁.trian
    @show Γ₁.glue.face_to_bgface 
    @show Γ₁.glue.bgface_to_lcell
    @show Γ₁.glue.face_to_cell
    @show Γ₁.glue.face_to_lface
    @show Γ₁.glue.face_to_lcell
    @show Γ₁.glue.face_to_ftype
    @show Γ₁.glue.cell_to_ctype
    @show Γ₁.glue.cell_to_lface_to_pindex
    @show Γ₁.glue.ctype_to_lface_to_ftype
    @show Γ2 = Interface(Interior(model), Ω⁻act).⁻
    # Γ₁ = Interface(Interior(model), Ω⁻act).⁻
    @show Γ2.trian
    @show Γ2.glue.face_to_bgface  
    @show Γ2.glue.bgface_to_lcell
    @show Γ2.glue.face_to_cell
    @show Γ2.glue.face_to_lface
    @show Γ2.glue.face_to_lcell
    @show Γ2.glue.face_to_ftype
    @show Γ2.glue.cell_to_ctype
    @show Γ2.glue.cell_to_lface_to_pindex
    @show Γ2.glue.ctype_to_lface_to_ftype
    # @show new_ind = setdiff(BoundaryTriangulation(Ω⁻act).dtrian.glue.face_to_bgface, BoundaryTriangulation(Ω⁻act,tags=["top","DT"]).dtrian.glue.face_to_bgface)
    # @show Γ3 = BoundaryTriangulation(Ω⁻act, new_ind) #Interface(Interior(model), Ω⁻act).⁻
    # @show Γ3.dtrian.trian
    # @show Γ3.dtrian.glue.face_to_bgface  
    # @show Γ3.dtrian.glue.bgface_to_lcell
    # @show Γ3.dtrian.glue.face_to_cell
    # @show Γ3.dtrian.glue.face_to_lface
    # @show Γ3.dtrian.glue.face_to_lcell
    # @show Γ3.dtrian.glue.face_to_ftype
    # @show Γ3.dtrian.glue.cell_to_ctype
    # @show Γ3.dtrian.glue.cell_to_lface_to_pindex
    # @show Γ3.dtrian.glue.ctype_to_lface_to_ftype

    # @show isequal(Γ₁.glue.face_to_bgface ,Γ2.glue.face_to_bgface)
    # @show isequal(Γ2.glue.face_to_bgface, Γ3.dtrian.glue.face_to_bgface )

    # Ω⁻act = Interior(cutgeo, OUT)
    # Ω⁻pas = Interior(cutgeo, ACTIVE_IN)

    # Γ₁ = BoundaryTriangulation(cutgeo_facets, ACTIVE_IN)
    #SkeletonTriangulation(cutgeo_facets, ).⁺#Interface(Interior(model), Ω⁻act).⁻#Interface(Ω⁻pas,Ω⁻act).⁺

    nΓ₁ = get_normal_vector(Γ₁)
    Γ₂ = BoundaryTriangulation(Ω⁻act, tags=["top"])
    nΓ₂ = get_normal_vector(Γ₂)
    writevtk(Ω⁻pas,"pas")
    writevtk(Ω⁻act,"act")
    writevtk(Γ₁, "gam1", cellfields=["normal"=>CellField(nΓ₁,Γ₁)])
    writevtk(Γ₂, "gam2", cellfields=["normal"=>CellField(nΓ₂,Γ₂)])
    # writevtk(SBMSkeleton(cutgeo,OUT,get_geometery(cutgeo.geo)),"gam3")
    dΓ = Measure(Γ₁,1)
    @show nΓ₁(get_cell_points(dΓ.quad))
    @show get_normal_vector(Γ2)(get_cell_points(Measure(Γ2,1).quad))
    return Ω⁻act, Γ₁, nΓ₁, Γ₂, nΓ₂ 
end # function

function _build_helper(cutgeo::EmbeddedDiscretization, geo::Geometry, model::DiscreteModel, var::Vector)
    ioc = compute_bgcell_to_inoutcut(cutgeo,geo)
    pas_var = setdiff([-1,0,1],var)
    function _helper_loop(var,ioc)
        arr = []
        for i in var
            tmp=findall(ioc.==i)
            push!(arr,tmp)
        end
        return reduce(vcat,arr)
    end # function
    Ω⁻act = Interior(model, _helper_loop(var,ioc))
    Ω⁻pas = Interior(model, _helper_loop(pas_var,ioc))
    return Ω⁻act, Ω⁻pas
end # function

method="sbm"
case="cylinder"
(nₓ_vec, orders), (Lₓ, L₃, R), (g, k, ω, η₀), _, (ls, to), folder = CaseSetup.parameters(method, case)
nₓ = 6
order = 1
model, _, ϕ₀, f₁, f₂ = CaseSetup.setup_model_2d(nₓ;Lₓ=Lₓ,L₃=L₃,func_args=[g,k,η₀,ω])
cutgeo, geo, cutgeo_facets = CaseSetup.geometry_cut(model;Lₓ=Lₓ, L₃=L₃, R=R)
Ωsbm, Γ₁, nΓ₁, Γ₂, nΓ₂ = _build_domain_sbm(cutgeo,cutgeo_facets, geo, model)
d, n, f₂sbm = CaseSetup.analytical_distance(model,Lₓ,L₃,R,f₂)
degree=2*order
dΩsbm, dΓ₁, dΓ₂ = CaseSetup.set_measures(degree, Ωsbm, Γ₁, Γ₂)
V, U = CaseSetup.set_spaces(order, Ωsbm, ϕ₀)
a, l = CaseSetup.weak_form(method)
op = CaseSetup.build_operator(a(dΩsbm,dΓ₁,nΓ₁,n,d),l(dΩsbm,dΓ₁,nΓ₁,dΓ₂,nΓ₂,n,f₁,f₂,f₂sbm),U,V)
ϕₕ = solve(ls, op)


CaseSetup.write_results_omg(nₓ, order, ϕₕ ,ϕ₀, Ωsbm;folder="")
writevtk(Γ₁,"test",cellfields=["n"=>CellField(n(0),Γ₁),"d"=>CellField(d(0),Γ₁),"f2sbm"=>CellField(f₂sbm(0),Γ₁)])
# writevtk(E⁰,"teste",cellfields=["n"=>CellField(n(0),E⁰),"d"=>CellField(d(0),E⁰),"f2sbm"=>CellField(f₂sbm(0),E⁰),"normsur+"=>CellField(nE⁰.⁺,E⁰),"normsur-"=>CellField(nE⁰.⁻,E⁰)])
# CaseSetup.write_results_gam(nₓ, order, Γ₁, nΓ₁, Γ₂, nΓ₂;folder="")


end # module