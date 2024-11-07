module run_cases
using Gridap
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra
using CSV
using Plots

include("../src/case_setup.jl")
include("../src/agfem.jl")
include("../src/cutfem.jl")
include("../src/sbm.jl")
include("../src/wsbm.jl")

function execute_time(nruns::Int64,nx::Vector,cases::Vector;plot_flag=false,print_flag=false,write_flag=true,folder=folder)

    # helper functions
    # function _helper_to_dict(nruns::Int64,to_arr::Vector)
    #     arr = []
    #     for i in 1:length(to_arr)
    #         merged = to_arr[i][1]
    #         for run in 2:nruns
    #             val = to_arr[i][run]
    #             merged = merge!(merged,val)
    #         end # for
    #         push!(arr,merged)
    #     end # for
    #     return arr
    # end # function

    # function _helper_call_funcs(case::String, nx::Vector)
    #     _, _, _, _, to1 = ModAgfem.agfem(case,nx)
    #     _, _, _, _, to2 = ModCutfem.cutfem(case,nx)
    #     _, _, _, _, to3 = ModSbm.sbm(case,nx)
    #     _, _, _, _, to4 = ModWsbm.wsbm(case,nx)
    #     return to1, to2, to3, to4
    # end # function

    function _helper_call_funcs(case::String, nx::Vector)
        _, _, l2s1, cns1, tos1 = ModAgfem.agfem(case,nx; vtk_flag=false)
        _, _, l2s2, cns2, tos2 = ModCutfem.cutfem(case,nx; vtk_flag=false)
        _, _, l2s3, cns3, tos3 = ModSbm.sbm(case,nx; vtk_flag=false)
        _, _, l2s4, cns4, tos4 = ModWsbm.wsbm(case,nx; vtk_flag=false)
        return (tos1, tos2, tos3, tos4), (l2s1,l2s2,l2s3,l2s4), (cns1,cns2,cns3,cns4)
    end # function

    function _helper_plot(order::Int64,n::Int64,to::TimerOutput,calls::Vector)
        ttotal = 0.0
        atotal = 0.0
        tvals = []
        avals = []
        for call in calls
            tval = TimerOutputs.time(to[call*" $order, $n"])#/TimerOutputs.ncalls(to[call*" $order, $n"])/(10^9)  # [s]: original time unit is ns
            ttotal+=tval
            push!(tvals,tval)
            aval = TimerOutputs.allocated(to[call*" $order, $n"])#/TimerOutputs.ncalls(to[call*" $order, $n"])/(10^9)  # [s]: original time unit is ns
            atotal+=aval
            push!(avals,aval)
        end
        tnorm=tvals./ttotal
        anorm=avals./atotal
        return tnorm, tvals, anorm, avals
    end # function

    function _plot_agfem(order::Int64,n::Int64,to::TimerOutput)
        calls = ["affine","cutting","domain","model","quadratures","solving","spaces","weak_form","interior_matrix","rhs"]
        norm, vals, anorm, avals = _helper_plot(order,n,to,calls)
        return norm, vals, calls, anorm, avals
    end # function

    function _plot_cutfem(order::Int64,n::Int64,to::TimerOutput)
        calls = ["affine","cutting","domain","model","quadratures","solving","spaces","weak_form","interior_matrix","rhs","ghost_penalty_matrix"]
        norm, vals, anorm, avals = _helper_plot(order,n,to,calls)        
        return norm, vals, calls, anorm, avals
    end # function

    function _plot_sbm(order::Int64,n::Int64,to::TimerOutput)
        calls = ["affine","cutting","distances","domain","model","quadratures","solving","spaces","weak_form","interior_matrix","rhs","shifted_boundary_matrix"]
        norm, vals, anorm, avals = _helper_plot(order,n,to,calls)
        return norm, vals, calls, anorm, avals
    end # function

    function _plot_wsbm(order::Int64,n::Int64,to::TimerOutput)
        calls = ["affine","cutting","distances","domain","model","quadratures","solving","spaces","weak_form","volume_fraction","interior_matrix","rhs","ghost_penalty_matrix","shifted_edges_matrix","shifted_boundary_matrix"]
        norm, vals, anorm, avals = _helper_plot(order,n,to,calls)
        return norm, vals, calls, anorm, avals
    end # function

    function _minmaxmean(nruns::Int64,order::Int64,n::Int64,fun::Function,data)
        arr1 = []
        arr2 = []
        calls_out = []
        for j in 2:nruns+1
            _,vals,calls,_,avals = fun(order,n,data[j])
            push!(arr1,vals)
            push!(arr2,avals)
            push!(calls_out, calls)
        end # for
        mat1 = hcat(arr1...)./1e9    # [s] 1e9 to go from ns to s
        mat2 = hcat(arr2...)./1e9    # [GiB] 1e9 to go from B to MiG
        mean_time = sum(mat1,dims=2)./nruns
        mean_all = sum(mat2,dims=2)./nruns
        return [minimum(b) for b in eachrow(mat1)], [maximum(b) for b in eachrow(mat1)], vec(mean_time), [minimum(b) for b in eachrow(mat2)], [maximum(b) for b in eachrow(mat2)], vec(mean_all), calls_out[1]
    end # function   

    # run cases nruns times (first run is warmup & discarded)
    to_agfem = []
    to_cutfem = []
    to_sbm = []
    to_wsbm = []

    l2_agfem = []
    l2_cutfem = []
    l2_sbm = []
    l2_wsbm = []

    cn_agfem = []
    cn_cutfem = []
    cn_sbm = []
    cn_wsbm = []
    # nx = [8,16,32,64,128,256,512]
    # nx = [8]
    orders = [1,2]

    for run in 1:(nruns+1)
        for case in cases
            (to1, to2, to3, to4), (l2s1,l2s2,l2s3,l2s4), (cns1,cns2,cns3,cns4) = _helper_call_funcs(case,nx)
            push!(to_agfem,to1)
            push!(to_cutfem,to2)
            push!(to_sbm,to3)
            push!(to_wsbm,to4)

            push!(l2_agfem,l2s1)
            push!(l2_cutfem,l2s2)
            push!(l2_sbm,l2s3)
            push!(l2_wsbm,l2s4)

            push!(cn_agfem,cns1)
            push!(cn_cutfem,cns2)
            push!(cn_sbm,cns3)
            push!(cn_wsbm,cns4)
        end # for
    end # for

    if print_flag
        println("AgFEM")
        show(to_agfem)
        # show(to_dict[1])
        println()
        println("CutFEM")
        show(to_cutfem)
        # show(to_dict[2])
        println()
        println("SBM")
        show(to_sbm)
        # show(to_dict[3])
        println()
        println("WSBM")
        show(to_wsbm)
        # show(to_dict[4])
    end # if

    if write_flag
        for (ni,i) in enumerate(["agfem","cutfem","sbm","wsbm"])
            for order in orders
                minT_arr = []
                for (nn,n) in enumerate(nx)
                    minT = 0.0
                    touch("data/exp_pro/$folder/to_$(i)_$(order)_$n.csv")
                    if i == "agfem"
                        min_vals, max_vals, mean_vals, min_alls, max_alls, mean_alls, calls = _minmaxmean(nruns,order,n,_plot_agfem,hcat(to_agfem...)[order,:])
                        mn = DataFrame(Rel = mean_vals./sum(mean_vals), Abs = mean_vals, Tags = calls, 
                                        Min = min_vals, Max = max_vals, Rel_all = mean_alls./sum(mean_alls), 
                                        All = mean_alls, Min_all = min_alls, Max_all = max_alls)
                        minT = sum(min_vals)
                    elseif i == "cutfem"
                        min_vals, max_vals, mean_vals, min_alls, max_alls, mean_alls, calls = _minmaxmean(nruns,order,n,_plot_cutfem,hcat(to_cutfem...)[order,:])
                        mn = DataFrame(Rel = mean_vals./sum(mean_vals), Abs = mean_vals, Tags = calls,
                                        Min = min_vals, Max = max_vals, Rel_all = mean_alls./sum(mean_alls), 
                                        All = mean_alls, Min_all = min_alls, Max_all = max_alls)
                        minT = sum(min_vals)
                    elseif i == "sbm"
                        min_vals, max_vals, mean_vals, min_alls, max_alls, mean_alls, calls = _minmaxmean(nruns,order,n,_plot_sbm,hcat(to_sbm...)[order,:])
                        mn = DataFrame(Rel = mean_vals./sum(mean_vals), Abs = mean_vals, Tags = calls, 
                                        Min = min_vals, Max = max_vals, Rel_all = mean_alls./sum(mean_alls), 
                                        All = mean_alls, Min_all = min_alls, Max_all = max_alls)
                        minT = sum(min_vals)    
                    elseif i == "wsbm"
                        min_vals, max_vals, mean_vals, min_alls, max_alls, mean_alls, calls = _minmaxmean(nruns,order,n,_plot_wsbm,hcat(to_wsbm...)[order,:])
                        mn = DataFrame(Rel = mean_vals./sum(mean_vals), Abs = mean_vals, Tags = calls, 
                                        Min = min_vals, Max = max_vals, Rel_all = mean_alls./sum(mean_alls), 
                                        All = mean_alls, Min_all = min_alls, Max_all = max_alls)
                        minT = sum(min_vals)
                    end # if
                    push!(minT_arr,minT)
                    CSV.write("data/exp_pro/$folder/to_$(i)_$(order)_$n.csv", mn)
                end # for

                if i == "agfem"
                    mn2 = DataFrame(Nx = nx, MinT = minT_arr, L2 = l2_agfem[nruns][order], CN = cn_agfem[nruns][order])
                elseif i == "cutfem"
                    mn2 = DataFrame(Nx = nx, MinT = minT_arr, L2 = l2_cutfem[nruns][order], CN = cn_cutfem[nruns][order])
                elseif i == "sbm"
                    mn2 = DataFrame(Nx = nx, MinT = minT_arr, L2 = l2_sbm[nruns][order], CN = cn_sbm[nruns][order])
                elseif i == "wsbm"
                    mn2 = DataFrame(Nx = nx, MinT = minT_arr, L2 = l2_wsbm[nruns][order], CN = cn_wsbm[nruns][order])
                end # if
                CSV.write("data/exp_pro/$folder/conv_$(i)_$(order).csv", mn2)
            end # for
        end # for
    end # if

end # function

function _setup_helper(a::Vector)
    c = ["cutting","distances","model","quadratures","weak_form"]
    findall(x->x in c, a), findall(x->!(x in c), a)
end # function

function _setup(a::Vector,b::Vector)
    val = 0.0
    for i in a
        val+=b[i]
    end # for
    val
end # function

function _plot_bar(df::DataFrame,tags::Vector,target::Vector)
    a,b = _setup_helper(df.Tags)
    setup = _setup(a,target)
    arr = [setup]
    for (ni,i) in enumerate(tags)
        if i in df.Tags
            ind = findall(x->x==i,df.Tags)
            push!(arr,target[ind[1]])
        else 
            push!(arr, 0.0)
        end # if
    end # for
    arr
end # function

function _plot_bar_helper(arr::Vector,tags::Vector)
    A=hcat(arr...)
    A = [A[i,:] for i in 1:size(A,1)]
    a_prev = [0.0,0.0,0.0,0.0]
    arrb = []
    for (ni,i) in enumerate(tags)
        b=A[ni].+a_prev
        a_prev=a_prev.+A[ni]  
        push!(arrb,b)
    end # for
    arrb = reverse(arrb)
    arrb
end # function

function make_bar_plot(nx::Vector,orders::Vector;folder="")
    tags = ["setup","domain","solving","spaces","volume_fraction","interior_matrix","ghost_penalty_matrix","shifted_edges_matrix","shifted_boundary_matrix"]
    # Tol color palette (color blindness)
    colors=[RGB(51/255,34/255,136/255),RGB(17/255,119/255,51/255),RGB(68/255,170/255,153/255),RGB(136/255,204/255,238/255),RGB(0,0,0),
    RGB(221/255,204/255,119/255),RGB(204/255,102/255,119/255),RGB(170/255,68/255,153/255),RGB(136/255,34/255,85/255)]
    
    for n in nx
        for order in orders
            # arr = []
            min = []
            # max = []
            min_all = []
            for (ni,i) in enumerate(["agfem","cutfem","sbm","wsbm"])
                mn = CSV.read("data/exp_pro/$folder/to_$(i)_$(order)_$n.csv", DataFrame)
                # push!(arr,_plot_bar(mn,tags[2:end],mn.Abs))                
                push!(min,_plot_bar(mn,tags[2:end],mn.Min))
                # push!(max,_plot_bar(mn,tags[2:end],mn.Max))
                push!(min_all,_plot_bar(mn,tags[2:end],mn.All))
            end # for
            # arrb = _plot_bar_helper(arr,tags)
            minb = _plot_bar_helper(min,tags)
            # maxb = _plot_bar_helper(max,tags)
            minb_all = _plot_bar_helper(min_all,tags)

            # p=bar(legend=:outertopright,title="Mean order=$order n=$n") 
            # for (ni,i) in enumerate(reverse(tags))
            #     p=bar!(["agfem","cutfem","sbm","wsbm"],arrb[ni],label=i,fillcolor=colors[ni])
            # end # for
            # ylabel!("Time [s]")
            # display(p)
            # savefig(p,"data/exp_pro/figures2/val_$(order)_$n.png")

            p=bar(legend=:outertopright,title="Times order=$order n=$n") 
            for (ni,i) in enumerate(reverse(tags))
                p=bar!(["agfem","cutfem","sbm","wsbm"],minb[ni],label=i,fillcolor=colors[ni])
            end # for
            ylabel!("Time [s]")
            display(p)            
            savefig(p,"data/exp_pro/figures/min_$(order)_$n.png")

            # p=bar(legend=:outertopright,title="Max order=$order n=$n") 
            # for (ni,i) in enumerate(reverse(tags))
            #     p=bar!(["agfem","cutfem","sbm","wsbm"],maxb[ni],label=i,fillcolor=colors[ni])
            # end # for
            # ylabel!("Time [s]")
            # display(p)
            # savefig(p,"data/exp_pro/figures2/max_$(order)_$n.png")

            p=bar(legend=:outertopright,title="Allocations order=$order n=$n") 
            for (ni,i) in enumerate(reverse(tags))
                p=bar!(["agfem","cutfem","sbm","wsbm"],minb_all[ni],label=i,fillcolor=colors[ni])
            end # for
            ylabel!("Allocations [GiB]")
            display(p)            
            savefig(p,"data/exp_pro/figures/min_all_$(order)_$n.png")

        end # for
    end # for
end # function

function _plot_conv_time(cases,orders;folder="")
    for order in orders
        for case in cases
            p = plot(legend=:bottomleft,xlabel="CPU time [s]",ylabel="ϵ [-]",title="L2 norm error for $case at order $order")
            labels = ["CutFEM","AgFEM","SBM","WSBM"]
            markers = [:circle,:diamond,:xcross,:star6]
            for (ni,i) in enumerate(["cutfem","agfem","sbm","wsbm"])
                mn = CSV.read("data/exp_pro/$folder/conv_$(i)_$(order).csv", DataFrame)
                plot!(mn.MinT,mn.L2,xaxis=:log,yaxis=:log,marker=markers[ni],label=labels[ni])
            end # for
            display(p)
            savefig(p,"data/exp_pro/figures/conv_time_$(case)_$(order).png")
        end # for
    end # for
end # function

# make_bar_plot([512],[2])
folder = "conv_time"
execute_time(10,[8,16,32,64,128,256,512,1024],["cylinder"];print_flag=false,folder=folder)
_plot_conv_time(["cylinder"],[1,2];folder=folder)
# make_bar_plot([512],[2];folder=folder)
# make_bar_plot([512],[1];folder=folder)

end # module