# using Gridap
# using GridapEmbedded
# using GridapEmbedded.LevelSetCutters
# using STLCutters
using LinearAlgebra

# export benchmark, print_benchmark_results

# ===================================================
# Categories — shared + method-specific
# ===================================================
const CATEGORIES_BASE = [:model, :cutting, :domain, :quadratures, :spaces,
                         :weak_form, :interior_matrix, :rhs, :affine, :solving]

method_categories(::AGFEM)  = CATEGORIES_BASE
method_categories(::CUTFEM) = vcat(CATEGORIES_BASE, [:ghost_matrix])
method_categories(::SBM)    = vcat(CATEGORIES_BASE, [:boundary_matrix])
method_categories(::WSBM)   = vcat(CATEGORIES_BASE, [:ghost_matrix, :boundary_matrix])

# ===================================================
# Method-specific: build spaces
# ===================================================
function _build_spaces(method::AGFEM, domain, fe_config, sol,
                       cutgeo, embedded_geo, domain_config)
    build_spaces(method, domain, fe_config, sol, cutgeo, embedded_geo, domain_config)
end

function _build_spaces(method::EmbeddingMethod, domain, fe_config, sol,
                       cutgeo, embedded_geo, domain_config)
    build_spaces(method, domain, fe_config, sol)
end

# ===================================================
# Method-specific: build weak form
# ===================================================
function _build_weak_form(method::AGFEM, measures, domain, f₁, f₂;
                          h, γg, order, degree)
    build_weak_form(method, measures, domain, f₁, f₂)
end

function _build_weak_form(method::CUTFEM, measures, domain, f₁, f₂;
                          h, γg, order, degree)
    build_weak_form(method, measures, domain, h, γg, order, f₁, f₂)
end

function _build_weak_form(method::SBM, measures, domain, f₁, f₂;
                          h, γg, order, degree)
    n = domain.nΓ₁
    d = CellField(zero(VectorValue{2,Float64}), domain.Γ₁)
    build_weak_form(method, measures, domain, n, d, f₁, f₂, f₁)
end

function _build_weak_form(method::WSBM, measures, domain, f₁, f₂;
                          h, γg, order, degree)
    n = domain.nΓ₁
    d = CellField(zero(VectorValue{2,Float64}), domain.Γ₁)
    α = CellField(1.0, domain.Ω⁻)
    build_weak_form(method, measures, domain, n, d, α, h, γg, order, f₁, f₂, f₁)
end
# ===================================================
# Method-specific: assemble extra matrices
# ===================================================
function _assemble_extra!(t, a, wf, spaces, ::AGFEM)
    # nothing extra
end

function _assemble_extra!(t, a, wf, spaces, ::CUTFEM)
    t[:ghost_matrix] = @elapsed assemble_matrix(wf.a.ghost, spaces.U, spaces.V)
    a[:ghost_matrix] = @allocated assemble_matrix(wf.a.ghost, spaces.U, spaces.V)
end

function _assemble_extra!(t, a, wf, spaces, ::SBM)
    t[:boundary_matrix] = @elapsed assemble_matrix(wf.a.boundary, spaces.U, spaces.V)
    a[:boundary_matrix] = @allocated assemble_matrix(wf.a.boundary, spaces.U, spaces.V)
end

function _assemble_extra!(t, a, wf, spaces, ::WSBM)
    t[:ghost_matrix]    = @elapsed  assemble_matrix(wf.a.ghost,    spaces.U, spaces.V)
    a[:ghost_matrix]    = @allocated assemble_matrix(wf.a.ghost,    spaces.U, spaces.V)
    t[:boundary_matrix] = @elapsed  assemble_matrix(wf.a.boundary, spaces.U, spaces.V)
    a[:boundary_matrix] = @allocated assemble_matrix(wf.a.boundary, spaces.U, spaces.V)
end

# ===================================================
# Single full pipeline run
# ===================================================
function _single_run(method::EmbeddingMethod, params::SimulationParams{N},
                     sol::ManufacturedSolution{N}, embedded_geo,
                     domain_config::DomainConfig, fe_config::FESpaceConfig,
                     f₁::Function, f₂::Function) where {N}

    degree = 2 * params.solver.order
    h      = params.geometry.L₁ / params.solver.n
    γg     = params.solver.γg
    t      = Dict{Symbol, Float64}()
    a      = Dict{Symbol, Int}()

    # 1. Model
    t[:model] = @elapsed  model, _ = setup_model(params)
    a[:model] = @allocated setup_model(params)

    # 2. Cutting
    t[:cutting] = @elapsed begin
        cutgeo, cutgeo_facets = geometry_cut(model, embedded_geo)
    end
    a[:cutting] = @allocated begin
        _, _ = geometry_cut(model, embedded_geo)
    end

    # 3. Domain
    t[:domain] = @elapsed begin
        domain = build_domain(method, cutgeo, cutgeo_facets, domain_config)
    end
    a[:domain] = @allocated begin
        build_domain(method, cutgeo, cutgeo_facets, domain_config)
    end

    # Reference SBM domain for L2 norm — not timed
    ref_domain = build_reference_domain(cutgeo, domain_config)
    dΩsbm      = Measure(ref_domain.Ω⁻, degree)

    # 4. Quadratures
    t[:quadratures] = @elapsed begin
        measures = build_measures(domain, degree)
    end
    a[:quadratures] = @allocated begin
        build_measures(domain, degree)
    end

    # 5. FE spaces
    t[:spaces] = @elapsed begin
        spaces = _build_spaces(method, domain, fe_config, sol,
                               cutgeo, embedded_geo, domain_config)
    end
    a[:spaces] = @allocated begin
        _build_spaces(method, domain, fe_config, sol,
                      cutgeo, embedded_geo, domain_config)
    end

    # 6. Weak form
    t[:weak_form] = @elapsed begin
        wf = _build_weak_form(method, measures, domain, f₁, f₂;
                              h=h, γg=γg, order=params.solver.order, degree=degree)
    end
    a[:weak_form] = @allocated begin
        _build_weak_form(method, measures, domain, f₁, f₂;
                         h=h, γg=γg, order=params.solver.order, degree=degree)
    end

    # 7. Interior matrix
    t[:interior_matrix] = @elapsed begin
        Ai = assemble_matrix(wf.a.interior, spaces.U, spaces.V)
    end
    a[:interior_matrix] = @allocated begin
        assemble_matrix(wf.a.interior, spaces.U, spaces.V)
    end

    # 8. Method-specific extra matrices
    _assemble_extra!(t, a, wf, spaces, method)

    # 9. RHS
    t[:rhs] = @elapsed begin
        b = assemble_vector(wf.l, spaces.V)
    end
    a[:rhs] = @allocated begin
        assemble_vector(wf.l, spaces.V)
    end

    # 10. Affine operator
    t[:affine] = @elapsed begin
        op = AffineFEOperator(wf.a.interior, wf.l, spaces.U, spaces.V)
    end
    a[:affine] = @allocated begin
        AffineFEOperator(wf.a.interior, wf.l, spaces.U, spaces.V)
    end

    # 11. Solve
    t[:solving] = @elapsed begin
        ϕₕ = solve(LUSolver(), op)
    end
    a[:solving] = @allocated begin
        solve(LUSolver(), op)
    end

    # L2 norm on reference SBM domain — not timed
    u, _, _  = manufactured_functions(sol)
    l2norm   = sqrt(sum(∫((ϕₕ - (x -> u(x, fe_config.t)))*(ϕₕ - (x -> u(x, fe_config.t))))dΩsbm))

    # Condition number in L1 norm
    cn = cond(get_matrix(op), 1)

    return t, a, l2norm, cn
end

# ===================================================
# Benchmark runner
# ===================================================
"""
    benchmark(method, nₓ_vec, params; n_runs, n_warmup, geometry) -> min_times, l2s

Run the full pipeline for a given EmbeddingMethod (n_warmup + n_runs) times
per (order, nₓ). Warmup runs allow JIT to settle and are discarded.
Returns minimum elapsed time per category across timed runs.

The `n` field in params is overridden per nₓ iteration — all other
parameters are taken directly from params.
"""
function benchmark(method::EmbeddingMethod, nₓ_vec::Vector{Int},
                   params::SimulationParams{N};
                   n_runs   = 10,
                   n_warmup = 3,
                   geometry = :cylinder,
                   orders   = [params.solver.order]) where {N}

    is_stl = geometry ∈ (:sphere_stl, :bunny)

    # Unpack from params
    Lₓ  = params.geometry.L₁
    L₃  = params.geometry.L₃
    R   = params.geometry.R
    g   = params.manufactured.g
    k   = params.manufactured.k
    η₀  = params.manufactured.η₀
    γg  = params.solver.γg

    sol = N == 2 ? AirySolution2D(g, k, η₀, L₃) :
                   AirySolution3D(g, k, η₀, L₃)

    embedded_geo = if is_stl
        stl_file = geometry == :sphere_stl ? "data/meshes/sphere.stl" :
                                             "data/meshes/bunnylow.stl"
        STLGeometry(stl_file)
    elseif N == 2
        CylinderGeometry(R, L₃/2, Lₓ, L₃)
    else
        SphereGeometry(R, L₃/2, Lₓ, L₃)
    end

    domain_config = DomainConfig(OUTSIDE, true)
    cats          = method_categories(method)

    min_times  = Dict(order => Dict(nₓ => Dict(cat => Inf   for cat in cats) for nₓ in nₓ_vec) for order in orders)
    min_allocs = Dict(order => Dict(nₓ => Dict(cat => typemax(Int) for cat in cats) for nₓ in nₓ_vec) for order in orders)
    l2s        = Dict(order => Dict(nₓ => 0.0  for nₓ in nₓ_vec) for order in orders)
    cns        = Dict(order => Dict(nₓ => 0.0  for nₓ in nₓ_vec) for order in orders)

    for order in orders
        fe_config  = FESpaceConfig(order, ["DT"], 0.0)
        u, ∇u, Δu = manufactured_functions(sol)
        f₁         = x -> Δu(x, fe_config.t)
        f₂         = x -> ∇u(x, fe_config.t)

        for nₓ in nₓ_vec
            n_total    = n_warmup + n_runs
            run_params = SimulationParams(
                params.geometry,
                params.manufactured,
                SolverParams(nₓ, order, γg, params.solver.folder)
            )
            println("$(typeof(method))  order=$order  nₓ=$nₓ  ($n_warmup warmup + $n_runs timed)")

            for run in 1:n_total
                GC.gc()
                GC.enable(false)
                times, allocs, l2norm, cn = _single_run(method, run_params, sol,
                                                         embedded_geo, domain_config,
                                                         fe_config, f₁, f₂)
                GC.enable(true)

                run <= n_warmup && continue

                for cat in cats
                    if haskey(times,  cat); min_times[order][nₓ][cat]  = min(min_times[order][nₓ][cat],  times[cat])  end
                    if haskey(allocs, cat); min_allocs[order][nₓ][cat] = min(min_allocs[order][nₓ][cat], allocs[cat]) end
                end
                l2s[order][nₓ] = l2norm
                cns[order][nₓ] = cn
            end

            println("  done — min solve: $(round(min_times[order][nₓ][:solving], sigdigits=4)) s")
        end
    end

    return min_times, min_allocs, l2s, cns
end

# ===================================================
# Pretty-print results
# ===================================================
function print_benchmark_results(method::EmbeddingMethod, min_times, min_allocs,
                                  l2s, cns, nₓ_vec, orders)
    cats = method_categories(method)
    w    = 20 + 24*length(nₓ_vec)
    sep  = "="^w

    for order in orders
        println("\n$sep")
        println("$(typeof(method))  |  Order $order  |  min time (s) / min allocs (bytes)")
        println(sep)

        header = rpad("category", 20) *
                 join(lpad("nₓ=$nₓ (t)", 12) * lpad("nₓ=$nₓ (a)", 12) for nₓ in nₓ_vec)
        println(header)
        println("-"^w)

        for cat in cats
            row = rpad(string(cat), 20) *
                  join(lpad(round(min_times[order][nₓ][cat],  sigdigits=4), 12) *
                       lpad(min_allocs[order][nₓ][cat], 12)
                       for nₓ in nₓ_vec)
            println(row)
        end

        println("-"^w)
        l2row = rpad("L2 norm", 20) *
                join(lpad(round(l2s[order][nₓ], sigdigits=4), 24) for nₓ in nₓ_vec)
        println(l2row)

        cnrow = rpad("condition nr", 20) *
                join(lpad(round(cns[order][nₓ], sigdigits=4), 24) for nₓ in nₓ_vec)
        println(cnrow)
    end
end
