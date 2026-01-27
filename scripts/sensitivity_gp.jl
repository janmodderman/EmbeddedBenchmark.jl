module sensitivity_ghost_penalty
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
include("../src/sensitivity_cutfem.jl")
include("../src/sensitivity_wsbm.jl")
# include("time_cases.jl")

function execute(cases::Vector,γg::Float64;plot_flag=false,vtk_flag=false)
case=cases[1]
l2s_cutfem = []
cns_cutfem = []
l2s_wsbm = []
cns_wsbm = []
to_cutfem = []
to_wsbm = []
nxlist = []
orderslist = [[1,2]]

nx = 100 # set fixed n [-]
# Given radius is 0.25, fixed offset is determined w.r.t. center of the box, i.e. -0.5 + 0.5 = center at 0.0
# x₀s = [0,2.5,5,7.5,10,12.5,15,17.5,20,22.5,25]./100 .+0.5
x₀s = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]./100 .+0.5
# for x₀ in x₀s
    _, _, l2s2, cns2, tos2 = ModCutfem.cutfem(case,x₀s; vtk_flag=vtk_flag,γg=γg)
    _, _, l2s4, cns4, tos4 = ModWsbm.wsbm(case,x₀s; vtk_flag=vtk_flag,γg=γg)
    push!(l2s_cutfem,l2s2)
    push!(l2s_wsbm,l2s4)

    println(l2s_cutfem)

    push!(cns_cutfem,cns2)
    push!(cns_wsbm,cns4)
    # push!(nxlist,nx)

    push!(to_cutfem,tos2)
    push!(to_wsbm,tos4)
# end # for

for order in orderslist[1]
    # for (i,case) in enumerate(cases)
        touch("data/exp_pro/convergence/sensitivity_$(γg)_$(case)_$(order).csv")
        mn = DataFrame(x = x₀s, L2_cutfem = l2s_cutfem[1][order], Cn_cutfem = cns_cutfem[1][order],
                L2_wsbm = l2s_wsbm[1][order], Cn_wsbm = cns_wsbm[1][order],)
        CSV.write("data/exp_pro/convergence/sensitivity_$(γg)_$(case)_$(order).csv", mn)
    # end # for
end # for

if plot_flag
    # for order in orderslist[1]
    #     for (i,case) in enumerate(cases)
    #         plt = plot(legend=:bottomleft)
    #         nₓ = nxlist[i]
    #         plot!(nₓ,l2s_cutfem[i][order],xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
    #         plot!(nₓ,l2s_wsbm[i][order],xaxis=:log,yaxis=:log,marker=:heptagon,label="WSBM")
    #         plot!(nₓ,0.05*nₓ.^(-1), xaxis=:log, yaxis=:log, labels="1st order", linestyle=:solid, color=:black)
    #         plot!(nₓ,0.05*nₓ.^(-2), xaxis=:log, yaxis=:log, labels="2nd order", linestyle=:dash, color=:black)
    #         plot!(nₓ,0.05*nₓ.^(-3), xaxis=:log, yaxis=:log, labels="3rd order", linestyle=:dot, color=:black)
    #         display(plt)
    #     end
    # end # for

    for order in orderslist[1]
        for (i,case) in enumerate(cases)
            plt2 = plot(legend=:bottomleft)
            plot!(nₓ,cns_cutfem[i][order],xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
            plot!(nₓ,cns_wsbm[i][order],xaxis=:log,yaxis=:log,marker=:heptagon,label="WSBM")
            display(plt2)
        end # for
    end # for

end # if

end # function


# function _plot_convergence(cases,orders)
#     for order in orders
#         for case in cases
#             p = plot(legend=:bottomleft)
#             mn = CSV.read("data/exp_pro/convergence/$(case)_$(order).csv", DataFrame)
#             nₓ = mn.N
#             plot!(nₓ,mn.L2_agfem,xaxis=:log,yaxis=:log,marker=:diamond,label="AgFEM",xlabel="nₓ [-]",ylabel="ε [-]",title="L2 norm error for "*case*" at order $order")
#             plot!(nₓ,mn.L2_cutfem,xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
#             plot!(nₓ,mn.L2_sbm,xaxis=:log,yaxis=:log,marker=:xcross,label="SBM")
#             plot!(nₓ,mn.L2_wsbm,xaxis=:log,yaxis=:log,marker=:star6,label="WSBM")
#             plot!(nₓ,0.05*nₓ.^(-1), xaxis=:log, yaxis=:log, labels="1st order", linestyle=:solid, color=:black)
#             plot!(nₓ,0.05*nₓ.^(-2), xaxis=:log, yaxis=:log, labels="2nd order", linestyle=:dash, color=:black)
#             plot!(nₓ,0.05*nₓ.^(-3), xaxis=:log, yaxis=:log, labels="3rd order", linestyle=:dashdot, color=:black)
#             display(p)
#             savefig(p,"data/exp_pro/figures/convergence_$(case)_$(order).png")
#         end # for
#     end # for
# end # function


function _plot_condition(cases,orders)
    for order in orders
        for case in cases
            p = plot(legend=:bottomleft)
            mn = CSV.read("data/exp_pro/convergence/$(case)_$(order).csv", DataFrame)
            nₓ = mn.N
            plot!(nₓ,mn.Cn_agfem,xaxis=:log,yaxis=:log,marker=:diamond,label="AgFEM",xlabel="nₓ [-]",ylabel="κ [-]",title="Condition number for "*case*" at order $order")
            plot!(nₓ,mn.Cn_cutfem,xaxis=:log,yaxis=:log,marker=:circle,label="CutFEM")
            plot!(nₓ,mn.Cn_sbm,xaxis=:log,yaxis=:log,marker=:xcross,label="SBM")
            plot!(nₓ,mn.Cn_wsbm,xaxis=:log,yaxis=:log,marker=:star6,label="WSBM")
            display(p)
            savefig(p,"data/exp_pro/figures/condition_$(case)_$(order).png")
        end # for
    end # for
end # function



for γg in [0.001,0.01,0.1,1.0,10.0,100.0]
    println("γg = $(γg)")
    execute(["cylinder"],γg;plot_flag=false,vtk_flag=false)
end # for

function plot_sensitivity_error(cases)
    println("in function plotting")
    for order in [1,2]
        for case in cases
            p = plot()
            mn = CSV.read("data/exp_pro/convergence/sensitivity_$(case)_cylinder_$(order).csv", DataFrame)
            nₓ = (mn.x .-0.5)
            plot!(nₓ,mn.L2_cutfem,yaxis=:log,marker=:circle,label="CutFEM",xlabel="h₀ [-]",ylabel="ε [-]",title="L2 error for γg=$(case), pₑ=$order, n=50")
            plot!(nₓ,mn.L2_wsbm,yaxis=:log,marker=:star6,label="WSBM")
            display(p)
            # savefig(p,"data/exp_pro/figures/condition_$(case)_$(order).png")
        end # for
    end # for
end

function plot_sensitivity(cases)
    println("in function plotting")
    for order in [1,2]
        for case in cases
            p = plot()
            mn = CSV.read("data/exp_pro/convergence/sensitivity_$(case)_cylinder_$(order).csv", DataFrame)
            nₓ = (mn.x .-0.5)
            plot!(nₓ,mn.Cn_cutfem,yaxis=:log,marker=:circle,label="CutFEM",xlabel="h₀ [-]",ylabel="κ [-]",title="Condition number for γg=$(case), pₑ=$order, n=50")
            plot!(nₓ,mn.Cn_wsbm,yaxis=:log,marker=:star6,label="WSBM")
            display(p)
            # savefig(p,"data/exp_pro/figures/condition_$(case)_$(order).png")
        end # for
    end # for
end
cases = [0.001,0.01,0.1,1.0,10.0,100.0]
plot_sensitivity(cases)
plot_sensitivity_error(cases)


end # module