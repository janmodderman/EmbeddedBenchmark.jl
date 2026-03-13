using Plots
using StatsPlots
using JSON3
using Colors

function save_benchmark(path::String, method::EmbeddingMethod,
                        min_times, min_allocs, l2s, cns,
                        nₓ_vec, orders)
    data = Dict(
        "method"     => string(typeof(method)),
        "nₓ_vec"     => nₓ_vec,
        "orders"     => orders,
        "categories" => string.(method_categories(method)),
        "min_times"  => Dict(
            string(order) => Dict(
                string(nₓ) => Dict(string(cat) => min_times[order][nₓ][cat]
                                   for cat in method_categories(method))
                for nₓ in nₓ_vec)
            for order in orders),
        "min_allocs" => Dict(
            string(order) => Dict(
                string(nₓ) => Dict(string(cat) => min_allocs[order][nₓ][cat]
                                   for cat in method_categories(method))
                for nₓ in nₓ_vec)
            for order in orders),
        "l2s"        => Dict(
            string(order) => Dict(string(nₓ) => l2s[order][nₓ]
                                  for nₓ in nₓ_vec)
            for order in orders),
        "cns"        => Dict(
            string(order) => Dict(string(nₓ) => cns[order][nₓ]
                                  for nₓ in nₓ_vec)
            for order in orders)
    )
    open(path, "w") do io
        JSON3.write(io, data)
    end
    println("Saved benchmark results to $path")
end

function load_benchmark(path::String)
    data = JSON3.read(read(path, String))

    orders   = Int.(data["orders"])
    nₓ_vec   = Int.(data["nₓ_vec"])
    cats     = Symbol.(data["categories"])
    method   = data["method"]

    min_times  = Dict(
        order => Dict(
            nₓ => Dict(
                Symbol(cat) => Float64(data["min_times"][string(order)][string(nₓ)][cat])
                for cat in cats)
            for nₓ in nₓ_vec)
        for order in orders)

    min_allocs = Dict(
        order => Dict(
            nₓ => Dict(
                Symbol(cat) => Int(data["min_allocs"][string(order)][string(nₓ)][cat])
                for cat in cats)
            for nₓ in nₓ_vec)
        for order in orders)

    l2s = Dict(
        order => Dict(
            nₓ => Float64(data["l2s"][string(order)][string(nₓ)])
            for nₓ in nₓ_vec)
        for order in orders)

    cns = Dict(
        order => Dict(
            nₓ => Float64(data["cns"][string(order)][string(nₓ)])
            for nₓ in nₓ_vec)
        for order in orders)

    return method, min_times, min_allocs, l2s, cns, nₓ_vec, orders, cats
end

# Tol color palette (color blindness friendly)
const TOL_COLORS = [
    RGB(51/255,  34/255, 136/255),   # setup
    RGB(17/255, 119/255,  51/255),   # domain
    RGB(68/255, 170/255, 153/255),   # solving
    RGB(136/255,204/255, 238/255),   # spaces
    RGB(0,       0,        0      ), # volume_fraction
    RGB(221/255,204/255, 119/255),   # interior_matrix
    RGB(204/255,102/255, 119/255),   # ghost_penalty_matrix
    RGB(170/255, 68/255, 153/255),   # shifted_edges_matrix
    RGB(136/255, 34/255,  85/255),   # shifted_boundary_matrix
]

const TOL_TAGS = [
    :model, :cutting, :domain, :quadratures, :spaces,
    :weak_form, :interior_matrix, :rhs, :affine,
    :solving, :ghost_matrix, :boundary_matrix, :shifted_edges
]

const SETUP_CATS = [:cutting, :distances, :model, :quadratures, :weak_form, :rhs]

const TAG_COLORS = Dict{Symbol, RGB{Float64}}(
    :setup                   => RGB(136/255,  34/255,  85/255),  # dark purple 
    :domain                  => RGB(170/255,  68/255, 153/255),  # purple
    :spaces                  => RGB(221/255, 204/255, 119/255),  # yellow
    :solving                 => RGB(204/255, 102/255, 119/255),  # pink
    :interior_matrix         => RGB(136/255, 204/255, 238/255),  # light blue
    :ghost_matrix            => RGB( 68/255, 170/255, 153/255),  # teal
    :boundary_matrix         => RGB( 51/255,  34/255, 136/255),  # dark blue
    :shifted_edges           => RGB( 17/255, 119/255,  51/255),  # green
    :volume_fraction         => RGB(  0/255,   0/255,   0/255),  # black
    # ungrouped — kept for merge_setup=false
    :cutting                 => RGB(136/255, 204/255, 238/255),
    :distances               => RGB(136/255, 204/255, 238/255),
    :model                   => RGB(136/255, 204/255, 238/255),
    :quadratures             => RGB(136/255, 204/255, 238/255),
    :weak_form               => RGB(136/255, 204/255, 238/255),
    :rhs                     => RGB(136/255, 204/255, 238/255),
    :affine                  => RGB(136/255, 204/255, 238/255),  
)

# Fallback for unknown tags
const FALLBACK_COLOR = RGB(0.5, 0.5, 0.5)

const CAT_ORDER = [
    :boundary_matrix,
    :shifted_edges,
    :ghost_matrix,
    :interior_matrix,
    :volume_fraction,
    :spaces,
    :solving,
    :domain,
    :setup,
]

function _sort_cats(cats::Vector{Symbol})
    ordered   = filter(cat -> cat ∈ cats, CAT_ORDER)          # known cats in fixed order
    unordered = filter(cat -> cat ∉ CAT_ORDER, cats)          # unknown cats appended at end
    return vcat(ordered, unordered)
end

function _cat_colors(cats::Vector{Symbol})
    reshape([get(TAG_COLORS, cat, FALLBACK_COLOR) for cat in cats], 1, :)
end

function merge_categories(min_times::Dict, min_allocs::Dict,
                           nₓ_vec::Vector{Int}, orders::Vector{Int},
                           merge_cats::Vector{Symbol}, new_cat::Symbol)

    # Build new category list — replace all merged cats with single new_cat
    # preserving order of first occurrence
    all_cats = collect(keys(first(values(first(values(min_times))))))

    new_cats = Symbol[]
    for cat in all_cats
        cat ∈ merge_cats && new_cat ∉ new_cats ? push!(new_cats, new_cat) : nothing
        cat ∉ merge_cats                        ? push!(new_cats, cat)    : nothing
    end

    # Sum merged categories for times and allocs
    new_times  = Dict(
        order => Dict(
            nₓ => Dict(
                cat => cat == new_cat ?
                    sum(get(min_times[order][nₓ], c, 0.0) for c in merge_cats) :
                    min_times[order][nₓ][cat]
                for cat in new_cats)
            for nₓ in nₓ_vec)
        for order in orders)

    new_allocs = Dict(
        order => Dict(
            nₓ => Dict(
                cat => cat == new_cat ?
                    sum(get(min_allocs[order][nₓ], c, 0) for c in merge_cats) :
                    min_allocs[order][nₓ][cat]
                for cat in new_cats)
            for nₓ in nₓ_vec)
        for order in orders)

    return new_times, new_allocs, new_cats
end

function plot_bar_from_file(path::String; normalized=false, quantity=:time,
                             merge_setup=true, skip_cats=[:affine], kwargs...)
    method_str, min_times, min_allocs, l2s, cns, nₓ_vec, orders, cats = load_benchmark(path)

    if merge_setup
        min_times, min_allocs, cats = merge_categories(
            min_times, min_allocs, nₓ_vec, orders, SETUP_CATS, :setup
        )
    end

    # Remove skipped categories
    filter!(cat -> cat ∉ skip_cats, cats)

    cats = _sort_cats(cats)

    data_src = quantity == :time ? min_times : min_allocs
    ylabel   = if quantity == :time
        normalized ? "Fraction of total time" : "Time (s)"
    else
        normalized ? "Fraction of total allocations" : "Allocations (bytes)"
    end

    plots = []
    for order in sort(orders)
        data = [data_src[order][nₓ][cat] for nₓ in sort(nₓ_vec), cat in cats]

        plotdata   = normalized ? data ./ sum(data, dims=2) : data
        nₓ_labels  = ["nₓ=$nₓ" for nₓ in sort(nₓ_vec)]
        cat_labels = reshape(string.(cats), 1, :)
        cat_colors = _cat_colors(cats)

        p = groupedbar(nₓ_labels, plotdata;
            label        = cat_labels,
            color        = cat_colors,
            ylabel       = ylabel,
            title        = "$method_str  order=$order",
            bar_position = :stack,
            bar_width    = 0.6,
            legend       = :outertopright,
            xrotation    = 0,
            kwargs...
        )
        push!(plots, p)
    end

    return plots
end

function plot_bar!(method::EmbeddingMethod, order::Int, nₓ_vec::Vector{Int},
                   min_times::Dict; normalized=false, kwargs...)
    cat_labels, nₓ_labels, data, data_norm = plot_bar(method, order, nₓ_vec, min_times)

    plotdata = normalized ? data_norm : data
    ylabel   = normalized ? "Fraction of total time" : "Time (s)"
    title    = "$(typeof(method))  order=$order"

    groupedbar(cat_labels, plotdata;
        label      = reshape(nₓ_labels, 1, :),
        ylabel     = ylabel,
        title      = title,
        xrotation  = 45,
        legend     = :topright,
        bar_width  = 0.7,
        kwargs...
    )
end

"""
    plot_bar(method, order, nₓ_vec, min_times) -> Figure

Plot a grouped bar chart of minimum times per category across mesh sizes.
"""
function plot_bar(method::EmbeddingMethod, order::Int, nₓ_vec::Vector{Int},
                  min_times::Dict)
    cats  = method_categories(method)
    nlabels = length(cats)
    ngroups  = length(nₓ_vec)

    # Build matrix: rows = categories, cols = nₓ values
    data = [min_times[order][nₓ][cat] for cat in cats, nₓ in nₓ_vec]

    # Normalise each nₓ column so bars sum to 1
    data_norm = data ./ sum(data, dims=1)

    cat_labels = string.(cats)
    nₓ_labels  = ["nₓ=$nₓ" for nₓ in nₓ_vec]

    return cat_labels, nₓ_labels, data, data_norm
end

function _helper_plot(method::EmbeddingMethod, order::Int, nₓ::Int,
                      min_times::Dict)
    cats   = method_categories(method)
    tvals  = [min_times[order][nₓ][cat] for cat in cats]
    ttotal = sum(tvals)
    tnorm  = tvals ./ ttotal
    return tnorm, tvals, cats
end

function _plot_agfem(order::Int, nₓ::Int, min_times::Dict)
    return _helper_plot(AGFEM(), order, nₓ, min_times)
end

function _plot_cutfem(order::Int, nₓ::Int, min_times::Dict)
    return _helper_plot(CUTFEM(), order, nₓ, min_times)
end

function _plot_sbm(order::Int, nₓ::Int, min_times::Dict)
    return _helper_plot(SBM(), order, nₓ, min_times)
end

function _plot_wsbm(order::Int, nₓ::Int, min_times::Dict)
    return _helper_plot(WSBM(), order, nₓ, min_times)
end

function plot_L2_from_files(paths::Vector{String}; marker=:circle, orders=nothing, kwargs...)

    # Load all data first
    datasets = [load_benchmark(path) for path in paths]

    # Determine orders to plot — union of all orders across files, or user-specified
    all_orders = orders !== nothing ? orders :
                 sort(unique(vcat([d[7] for d in datasets]...)))

    # Reference slopes scaled to first dataset first order
    _, _, _, l2s_ref, _, nₓ_ref, ord_ref, _ = datasets[1]
    nₓ_f = Float64.(sort(nₓ_ref))
    c    = l2s_ref[ord_ref[1]][sort(nₓ_ref)[end]]

    plots = []
    for order in all_orders
        p = plot(; xlabel="nₓ", ylabel="L2 norm", xaxis=:log, yaxis=:log,
                   title="L2 convergence: order=$order",
                   legend=:outertopright, kwargs...)

        for (i, (path, dataset)) in enumerate(zip(paths, datasets))
            method_str, _, _, l2s, _, nₓ_vec, file_orders, _ = dataset
            order ∉ file_orders && continue

            sort!(nₓ_vec)
            l2_vals = [l2s[order][nₓ] for nₓ in nₓ_vec]
            plot!(p, nₓ_vec, l2_vals;
                  marker = marker,
                  label  = method_str,
                  color  = TOL_COLORS[mod1(i, length(TOL_COLORS))])
        end

        # Reference slopes
        plot!(p, nₓ_f, c .* nₓ_f.^(-1); linestyle=:solid, color=:black, label="O(h¹)")
        plot!(p, nₓ_f, c .* nₓ_f.^(-2); linestyle=:dash,  color=:black, label="O(h²)")
        plot!(p, nₓ_f, c .* nₓ_f.^(-3); linestyle=:dot,   color=:black, label="O(h³)")

        push!(plots, p)
    end

    return plots
end

function plot_cond_from_files(paths::Vector{String}; marker=:circle, orders=nothing, kwargs...)

    datasets   = [load_benchmark(path) for path in paths]
    all_orders = orders !== nothing ? orders :
                 sort(unique(vcat([d[7] for d in datasets]...)))

    # Reference slopes
    _, _, _, _, cns_ref, nₓ_ref, ord_ref, _ = datasets[1]
    nₓ_f = Float64.(sort(nₓ_ref))
    c    = cns_ref[ord_ref[1]][sort(nₓ_ref)[1]]

    plots = []
    for order in all_orders
        p = plot(; xlabel="nₓ", ylabel="Condition number (L1)", xaxis=:log, yaxis=:log,
                   title="Condition number: order=$order",
                   legend=:outertopright, kwargs...)

        for (i, (path, dataset)) in enumerate(zip(paths, datasets))
            method_str, _, _, _, cns, nₓ_vec, file_orders, _ = dataset
            order ∉ file_orders && continue

            sort!(nₓ_vec)
            cn_vals = [cns[order][nₓ] for nₓ in nₓ_vec]
            plot!(p, nₓ_vec, cn_vals;
                  marker = marker,
                  label  = method_str,
                  color  = TOL_COLORS[mod1(i, length(TOL_COLORS))])
        end

        # Reference slopes for condition number growth
        plot!(p, nₓ_f, c .* nₓ_f.^(2); linestyle=:solid, color=:black, label="O(h⁻²)")
        plot!(p, nₓ_f, c .* nₓ_f.^(4); linestyle=:dash,  color=:black, label="O(h⁻⁴)")

        push!(plots, p)
    end

    return plots
end