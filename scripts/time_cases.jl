module run_cases
using Gridap
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra
using CSV

include("../src/case_setup.jl")
include("agfem.jl")
include("cutfem.jl")
include("sbm.jl")
include("wsbm.jl")

function execute(nruns::Int64,cases::Vector;plot_flag=false,print_flag=false)

    # helper functions
    function _helper_to_dict(nruns::Int64,to_arr::Vector)
        arr = []
        for i in 1:length(to_arr)
            merged = to_arr[i][1]
            for run in 2:nruns
                merged = merge!(merged, to_arr[i][run])
            end # for
            push!(arr,merged)
        end # for
        return arr
    end # function

    function _helper_call_funcs(case::String, nx::Vector)
        _, _, _, _, to1 = ModAgfem.agfem(case,nx)
        _, _, _, _, to2 = ModCutfem.cutfem(case,nx)
        _, _, _, _, to3 = ModSbm.sbm(case,nx)
        _, _, _, _, to4 = ModWsbm.wsbm(case,nx)
        return to1, to2, to3, to4
    end # function

    function _helper_plot(order::Int64,n::Int64,to::TimerOutput,calls::Vector)
        total = 0.0
        vals = []
        for call in calls
            tval = TimerOutputs.time(to[call*" $order, $n"])#/TimerOutputs.ncalls(to[call*" $order, $n"])/(10^9)  # [s]: original time unit is ns
            total+=tval
            push!(vals,tval)
        end
        norm=vals./total
        return norm, vals
    end # function

    function _plot_agfem(order::Int64,n::Int64,to::TimerOutput)
        calls = ["affine","cutting","domain","model","quadratures","spaces","weak_form"]
        norm, vals = _helper_plot(order,n,to,calls)
        return norm, vals, calls
    end # function

    function _plot_cutfem(order::Int64,n::Int64,to::TimerOutput)
        norm, vals, calls = _plot_agfem(order,n,to)
        return norm, vals, calls
    end # function

    function _plot_sbm(order::Int64,n::Int64,to::TimerOutput)
        calls = ["affine","cutting","distances","domain","model","quadratures","spaces","weak_form"]
        norm, vals = _helper_plot(order,n,to,calls)
        return norm, vals, calls
    end # function

    function _plot_wsbm(order::Int64,n::Int64,to::TimerOutput)
        calls = ["affine","cutting","distances","domain","model","quadratures","spaces","weak_form","volume_fraction"]
        norm, vals = _helper_plot(order,n,to,calls)
        return norm, vals, calls
    end # function

    # run cases nruns times (first run is warmup & discarded)
    to_agfem = []
    to_cutfem = []
    to_sbm = []
    to_wsbm = []
    nx = [8,16,32]#,64,128,256,512]

    for run in 1:(nruns+1)
        for case in cases
            if run > 1
                to1, to2, to3, to4 = _helper_call_funcs(case,nx)
                push!(to_agfem,to1)
                push!(to_cutfem,to2)
                push!(to_sbm,to3)
                push!(to_wsbm,to4)
            else
                _, _, _, _ = _helper_call_funcs(case,[8,16])
            end # if
        end # for
    end # for

    # process data
    to_dict = _helper_to_dict(nruns,[to_agfem,to_cutfem,to_sbm,to_wsbm])

    if print_flag
        println("AgFEM")
        show(to_dict[1])
        println()
        println("CutFEM")
        show(to_dict[2])
        println()
        println("SBM")
        show(to_dict[3])
        println()
        println("WSBM")
        show(to_dict[4])
    end # if

    # for i in ["agfem","cutfem","sbm","wsbm"]
    #     touch("data/timer_output_$i.csv")
    # end # for

    for (ni,i) in enumerate(["agfem","cutfem","sbm","wsbm"])
        for n in nx
            for order in [1,2]
                touch("data/exp_pro/timing/to_$(i)_$(order)_$n.csv")
                if i == "agfem"
                    norms, vals, calls = _plot_agfem(order,n,to_dict[1])
                    mn = DataFrame(Rel = norms, Abs = vals, Tags = calls)
                elseif i == "cutfem"
                    norms, vals, calls = _plot_cutfem(order,n,to_dict[2])
                    mn = DataFrame(Rel = norms, Abs = vals, Tags = calls)
                elseif i == "sbm"
                    norms, vals, calls = _plot_sbm(order,n,to_dict[3])
                    mn = DataFrame(Rel = norms, Abs = vals, Tags = calls)
                elseif i == "wsbm"
                    norms, vals, calls = _plot_wsbm(order,n,to_dict[4])
                    mn = DataFrame(Rel = norms, Abs = vals, Tags = calls)
                end # if
                CSV.write("data/exp_pro/timing/to_$(i)_$(order)_$n.csv", mn)
            end # for
        end # for
    end # for
    # mn = []
    # push!(mn,mn1)
    # push!(mn,mn2)
    # push!(mn,mn3)
    # push!(mn,mn4)

    # for (ni,i) in enumerate(["agfem","cutfem","sbm","wsbm"])
    #     CSV.write("data/timer_output_$i.csv", mn[ni])
    # end # for


end # function
execute(3,["cylinder"])
end # module