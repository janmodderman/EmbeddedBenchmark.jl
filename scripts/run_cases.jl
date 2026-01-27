module run_cases
using Gridap
using Plots
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra
using CSV
include("../src/case_setup.jl")
include("../src/agfem.jl")
include("../src/cutfem.jl")
include("../src/sbm.jl")
include("../src/wsbm.jl")
# include("time_cases.jl")

function execute(cases::Vector,nx::Vector;plot_flag=false,vtk_flag=false)

l2s_agfem = []
cns_agfem = []
l2s_cutfem = []
cns_cutfem = []
l2s_sbm = []
cns_sbm = []
l2s_wsbm = []
cns_wsbm = []
to_agfem = []
to_cutfem = []
to_sbm = []
to_wsbm = []
nxlist = []
orderslist = [[1,2]]

for case in cases
    _, _, l2s1, cns1, tos1 = ModAgfem.agfem(case,nx; vtk_flag=vtk_flag)
    _, _, l2s2, cns2, tos2 = ModCutfem.cutfem(case,nx; vtk_flag=vtk_flag)
    _, _, l2s3, cns3, tos3 = ModSbm.sbm(case,nx; vtk_flag=vtk_flag)
    _, _, l2s4, cns4, tos4 = ModWsbm.wsbm(case,nx; vtk_flag=vtk_flag)
    push!(l2s_agfem,l2s1)
    push!(l2s_cutfem,l2s2)
    push!(l2s_sbm,l2s3)
    push!(l2s_wsbm,l2s4)

    push!(cns_agfem,cns1)
    push!(cns_cutfem,cns2)
    push!(cns_sbm,cns3)
    push!(cns_wsbm,cns4)
    push!(nxlist,nx)

    push!(to_agfem,tos1)
    push!(to_cutfem,tos2)
    push!(to_sbm,tos3)
    push!(to_wsbm,tos4)
end # for

for order in orderslist[1]
    for (i,case) in enumerate(cases)
        touch("data/exp_pro/convergence/$(case)_$(order).csv")
        mn = DataFrame(N = nxlist[i], L2_agfem = l2s_agfem[i][order], Cn_agfem = cns_agfem[i][order], L2_cutfem = l2s_cutfem[i][order], Cn_cutfem = cns_cutfem[i][order],
                L2_sbm = l2s_sbm[i][order], Cn_sbm = cns_sbm[i][order], L2_wsbm = l2s_wsbm[i][order], Cn_wsbm = cns_wsbm[i][order],)
        CSV.write("data/exp_pro/convergence/$(case)_$(order).csv", mn)
    end # for
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


function _plot_convergence(cases,orders)
    for order in orders
        for case in cases
            p = plot(legend=:bottomleft)
            mn = CSV.read("data/exp_pro/convergence/$(case)_$(order).csv", DataFrame)
            nₓ = mn.N
            plot!(nₓ,mn.L2_agfem,xaxis=:log,yaxis=:log,marker=:diamond,label="AgFEM",xlabel="nₓ [-]",ylabel="ε [-]",title="L2 norm error for "*case*" at order $order")
            plot!(nₓ,mn.L2_cutfem,xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
            plot!(nₓ,mn.L2_sbm,xaxis=:log,yaxis=:log,marker=:xcross,label="SBM")
            plot!(nₓ,mn.L2_wsbm,xaxis=:log,yaxis=:log,marker=:star6,label="WSBM")
            plot!(nₓ,0.05*nₓ.^(-1), xaxis=:log, yaxis=:log, labels="1st order", linestyle=:solid, color=:black)
            plot!(nₓ,0.05*nₓ.^(-2), xaxis=:log, yaxis=:log, labels="2nd order", linestyle=:dash, color=:black)
            plot!(nₓ,0.05*nₓ.^(-3), xaxis=:log, yaxis=:log, labels="3rd order", linestyle=:dashdot, color=:black)
            display(p)
            savefig(p,"data/exp_pro/figures/convergence_$(case)_$(order).pdf")
        end # for
    end # for
end # function


function _plot_condition(cases,orders)
    for order in orders
        for case in cases
            p = plot(legend=:topleft)
            mn = CSV.read("data/exp_pro/convergence/$(case)_$(order).csv", DataFrame)
            nₓ = mn.N
            plot!(nₓ,mn.Cn_agfem,xaxis=:log,yaxis=:log,marker=:diamond,label="AgFEM",xlabel="nₓ [-]",ylabel="κ [-]",title="Condition number for "*case*" at order $order")
            plot!(nₓ,mn.Cn_cutfem,xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
            plot!(nₓ,mn.Cn_sbm,xaxis=:log,yaxis=:log,marker=:xcross,label="SBM")
            plot!(nₓ,mn.Cn_wsbm,xaxis=:log,yaxis=:log,marker=:star6,label="WSBM")
            display(p)
            savefig(p,"data/exp_pro/figures/condition_$(case)_$(order).pdf")
        end # for
    end # for
end # function




# execute(["cylinder"],[8,16,32,64,128,256,512];plot_flag=false,vtk_flag=true)
# execute(["cylinder"],[12,16,20,24,27,29,30,31,33];plot_flag=false,vtk_flag=true)
# _plot_condition(["cylinder"],[1,2])
# _plot_convergence(["cylinder"],[1,2])

# execute(["sphere_stl"],[8,12,20,24];plot_flag=false,vtk_flag=true)

# _plot_convergence(["sphere_stl"],[1,2])
# _plot_condition(["sphere_stl"],[1,2])
# execute(["sphere"],[8,12,20,24];plot_flag=false,vtk_flag=true)

# _plot_convergence(["sphere"],[1,2])
# _plot_condition(["sphere"],[1,2])

execute(["stanford"],[12,16,24,30];plot_flag=false,vtk_flag=false)

_plot_convergence(["stanford"],[1,2])
_plot_condition(["stanford"],[1,2])

# execute(["sphere_stl","stanford"];plot_flag=true)





end # module