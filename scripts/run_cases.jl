module run_cases
using Gridap
using Plots
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra

include("../src/case_setup.jl")
include("agfem.jl")
include("cutfem.jl")
include("sbm.jl")
include("wsbm.jl")

function execute(cases;plot_flag=false)

l2s_agfem = []
cns_agfem = []
l2s_cutfem = []
cns_cutfem = []
l2s_sbm = []
cns_sbm = []
l2s_wsbm = []
cns_wsbm = []
nxlist = []
orderslist = []

# cases =  ["cylinder","sphere","sphere_stl","stanford"]
# cases =  ["cylinder","sphere","sphere_stl","stanford"]
# cases =  ["sphere_stl","stanford"]
for case in cases
    _, _, l2s1, cns1, _ = ModAgfem.agfem(case,nx; vtk_flag=true)
    _, _, l2s2, cns2, _ = ModCutfem.cutfem(case,nx; vtk_flag=true)
    _, _, l2s3, cns3, _ = ModSbm.sbm(case,nx; vtk_flag=true)
    nₓ, orders, l2s4, cns4, _ = ModWsbm.wsbm(case,nx; vtk_flag=true)
    push!(l2s_agfem,l2s1)
    push!(l2s_cutfem,l2s2)
    push!(l2s_sbm,l2s3)
    push!(l2s_wsbm,l2s4)

    push!(cns_agfem,cns1)
    push!(cns_cutfem,cns2)
    push!(cns_sbm,cns3)
    push!(cns_wsbm,cns4)

    push!(nxlist,nₓ)
    push!(orderslist,orders)

end # for

if plot_flag
    for order in orderslist[1]
        for (i,case) in enumerate(cases)
            plt = plot(legend=:bottomleft)
            nₓ = nxlist[i]
            plot!(nₓ,l2s_agfem[i][order],xaxis=:log,yaxis=:log,marker=:diamond,label="AgFEM",xlabel="nₓ [-]",ylabel="ε [-]",title="L2 norm error for "*case*" at order $order")
            plot!(nₓ,l2s_cutfem[i][order],xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
            plot!(nₓ,l2s_sbm[i][order],xaxis=:log,yaxis=:log,marker=:xcross,label="SBM")
            plot!(nₓ,l2s_wsbm[i][order],xaxis=:log,yaxis=:log,marker=:heptagon,label="WSBM")
            plot!(nₓ,0.05*nₓ.^(-1), xaxis=:log, yaxis=:log, labels="1st order", linestyle=:solid, color=:black)
            plot!(nₓ,0.05*nₓ.^(-2), xaxis=:log, yaxis=:log, labels="2nd order", linestyle=:dash, color=:black)
            plot!(nₓ,0.05*nₓ.^(-3), xaxis=:log, yaxis=:log, labels="3rd order", linestyle=:dot, color=:black)
            display(plt)
        end
    end # for

    for order in orderslist[1]
        for (i,case) in enumerate(cases)
            plt2 = plot(legend=:bottomleft)
            plot!(nₓ,cns_agfem[i][order],xaxis=:log,yaxis=:log,marker=:diamond,label="AgFEM",xlabel="nₓ [-]",ylabel="κ [-]",title="L1 norm condition number for "*case*" at order $order")
            plot!(nₓ,cns_cutfem[i][order],xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
            plot!(nₓ,cns_sbm[i][order],xaxis=:log,yaxis=:log,marker=:xcross,label="SBM")
            plot!(nₓ,cns_wsbm[i][order],xaxis=:log,yaxis=:log,marker=:heptagon,label="WSBM")
            display(plt2)
        end # for
    end # for

end # if

end # function
execute(["sphere_stl","stanford"];plot_flag=true)
end # module