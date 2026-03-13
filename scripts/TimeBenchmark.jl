using EmbeddedBenchmark

# ===================================================
# Configuration
# ===================================================
method   = AGFEM()          # ← only change this: AGFEM(), CUTFEM(), SBM(), WSBM()
orders   = [1, 2]           
ns       = [16, 32, 64, 128]
n_runs   = 10
n_warmup = 3
geometry = :cylinder

# ===================================================
# Auto-derived from method — do not change below
# ===================================================
method_str = lowercase(string(typeof(method)))
savefile   = "data/$(method_str)_$(geometry).json"

params = SimulationParams(
    GeometryParams2D(1.0, 0.5, 0.25),
    ManufacturedParams(g=9.81, k=2π, η₀=0.05, d=0.5),
    SolverParams(ns[1], 1, 0.1, "data/$(method_str)/")
)

# ===================================================
# Run and save — appends if file already exists
# ===================================================
min_times, min_allocs, l2s, cns = benchmark(method, ns, params;
    n_runs   = n_runs,
    n_warmup = n_warmup,
    geometry = geometry,
    orders   = orders
)

if isfile(savefile)
    println("Appending to existing file: $savefile")
    append_benchmark(savefile, method, min_times, min_allocs, l2s, cns, ns, orders)
else
    println("Saving new file: $savefile")
    save_benchmark(savefile, method, min_times, min_allocs, l2s, cns, ns, orders)
end

print_benchmark_results(method, min_times, min_allocs, l2s, cns, ns, orders)

save_benchmark(savefile, method,
               min_times, min_allocs, l2s, cns, ns, [1])

# Later, in a separate session:
p = plot_bar_from_file(savefile; normalized=false, quantity=:time)
display(p)

p2 = plot_bar_from_file(savefile; normalized=false, quantity=:alloc)
display(p2)