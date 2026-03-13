using EmbeddedBenchmark

# ===================================================
# Configuration
# ===================================================
method   = CUTFEM()                  # ← only change this: AGFEM(), CUTFEM(), SBM(), WSBM()
orders   = [1,2]                   # ← only change this if you want to benchmark different polynomial orders (SBM and WSBM only allows up to 2nd order currently)
ns       = [16, 32, 64, 128]        # ← only change this if you want to benchmark different mesh sizes (number of cells in each direction is equal to n)
n_runs   = 10                       # ← only change this if you want to adjust the number of benchmark runs (more runs = more reliable results but longer runtime)
n_warmup = 3                        # ← only change this if you want to adjust the number of warmup runs (to mitigate compilation time effects)
geometry = :cylinder                # ← only change this if you want to benchmark a different geometry (currently only :cylinder is implemented, but you could add more in EmbeddedGeometry.jl and adjust the parameters accordingly in the SimulationParams constructor below)
run_save = false

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
# Run and save benchmark results
# ===================================================
if run_save
    min_times, min_allocs, l2s, cns = benchmark(method, ns, params;
        n_runs   = n_runs,
        n_warmup = n_warmup,
        geometry = geometry,
        orders   = orders
    )
    save_benchmark(savefile, method, min_times, min_allocs, l2s, cns, ns, orders)

    print_benchmark_results(method, min_times, min_allocs, l2s, cns, ns, orders)
end