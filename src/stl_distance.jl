module DistanceSTL
using Gridap
using GridapEmbedded
using LinearAlgebra
using STLCutters
using Gridap.CellData
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Geometry: CompositeTriangulation

using STLCutters: closest_point
using Gridap.CellData: get_cell_quadrature
using STLCutters: compute_cell_to_facets
using STLCutters: compose_index_map
using STLCutters: get_stl

function STLdistance(model::DiscreteModel, geo::STLGeometry, Γ::BoundaryTriangulation, dΓ::Measure)
    # This function calculates the closest distance from a point in the domain to the STL
    # For the application in SBM and WSBM, we want the closest distance of the quadrature points.
    # These points all lie in the faces of the boundary Γ. 
    # To speed up calculation, we pass along the STL facets closest to a face on Γ for each quadrature point.

    topo = get_grid_topology(model)
    D = num_dims(topo)
    nc = num_cells(Γ)
    qcp = get_cell_points(dΓ)
    Xₚ=Vector{VectorValue{D,Float64}}()
    for j in 1:nc
        for (i,ni) in enumerate(qcp.cell_phys_point[j])
            push!(Xₚ,ni)
        end
    end

    # Compute face_to_facets meaning face on Γ to facets of the STL file
    cell_to_facets = compute_cell_to_facets(model,get_stl(geo))
    face_to_cells = get_faces(topo,D-1,D)
    face_to_facets = compose_index_map(face_to_cells,cell_to_facets)
    newfaces = findall(!isempty,face_to_facets)
    face_to_facets = face_to_facets[newfaces]
    localfaces = findall(in(Γ.glue.face_to_bgface), newfaces)
    face_to_facets = face_to_facets[localfaces]

    npoints = Int64(length(Xₚ)/nc) 
    cq_facets = []            
    for j in 1:nc
        for i in 1:npoints
            push!(cq_facets,face_to_facets[j])
        end
    end
    Xc = closest_point(Xₚ,geo,cq_facets)
    distₘ0 = map(-,Xc,Xₚ)
    dₘ0 = map(norm,distₘ0)
    nₘ0 = map(/,distₘ0,dₘ0)

    distₘ = _quads_per_cell(nc, npoints, distₘ0)   #[distₘ0[1+(i-1)*npoints:npoints+(i-1)*npoints] for i in 1:nc]
    dₘ = _quads_per_cell(nc, npoints, dₘ0)   #[dₘ0[1+(i-1)*npoints:npoints+(i-1)*npoints] for i in 1:nc]
    nₘ = _quads_per_cell(nc, npoints, nₘ0)   # [nₘ0[1+(i-1)*npoints:npoints+(i-1)*npoints] for i in 1:nc]
    Xₘ = _quads_per_cell(nc, npoints, Xₚ)    #[Xₚ[1+(i-1)*npoints:npoints+(i-1)*npoints] for i in 1:nc]

    dsca = CellState(0.0,dΓ)
    nΓₜ_cs = CellState(VectorValue(0.0,0.0,0.0),dΓ)
    d_vec_cs = CellState(VectorValue(0.0,0.0,0.0),dΓ)
    Xd = CellState(VectorValue(0.0,0.0,0.0),dΓ)
    for icell in 1:length(dsca.values)
        for iqp in 1:length(dsca.values[icell])
            dsca.values[icell][iqp] = dₘ[icell][iqp]
            nΓₜ_cs.values[icell][iqp] = nₘ[icell][iqp]
            d_vec_cs.values[icell][iqp] = distₘ[icell][iqp]
            Xd.values[icell][iqp] = Xₘ[icell][iqp] + distₘ[icell][iqp]
        end
    end
    return dsca, nΓₜ_cs, d_vec_cs, Xd, Xₚ
end # function

function STLdistance(model::DiscreteModel, geo::STLGeometry, Γ::SkeletonTriangulation, dΓ::Measure)
    STLdistance(model, geo, Γ.⁺, dΓ)
end # function


# Function to shift analytical solutions, required for MMS with SBM & WSBM
function fshifted(Xd, fun, dΓ)
    vals = Xd.values                                          # values of coordinate of quadrature points + distance function at corresponding quadrature point
    outs = CellState(VectorValue(0.0,0.0,0.0),dΓ)             # empty CellState
    for icell in 1:length(Xd.values)                          # loop over each cell
      for iqp in 1:length(Xd.values[icell])                   # loop over each quadrature point in cell
        outs.values[icell][iqp] = fun(vals[icell][iqp], 0.0)  # evaluate analytical function at Xd
      end
    end
    outs                                                      # return CellState of shifted analytical function at each quadrature point
end # function

function _quads_per_cell(nc::Int64,nq::Int64,val::Vector)
    [val[1+(i-1)*nq:nq+(i-1)*nq] for i in 1:nc]
end # function

end # module