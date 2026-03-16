using EmbeddedBenchmark
using Gridap

# ===================================================
# Configuration
# ===================================================
method   = WSBM()                  # ← only change this: AGFEM(), CUTFEM(), SBM(), WSBM()
orders   = [1,2]                   # ← only change this if you want to benchmark different polynomial orders (SBM and WSBM only allows up to 2nd order currently)
ns       = [8, 16, 24]        # ← only change this if you want to benchmark different mesh sizes (number of cells in each direction is equal to n)
geometry = :sphere                # ← only change this if you want to benchmark a different geometry (currently only :cylinder is implemented, but you could add more in EmbeddedGeometry.jl and adjust the parameters accordingly in the SimulationParams constructor below)

# ===================================================
# Auto-derived from method — do not change below
# ===================================================
method_str = lowercase(string(typeof(method)))
savefile   = "data/convergence_$(method_str)_$(geometry).json"

params = SimulationParams(
    GeometryParams3D(1.0, 1.0, 1.0, VectorValue(0.0,0.0,0.0), 0.25),
    ManufacturedParams(g=9.81, k=2π, η₀=0.05, d=1.0),
    SolverParams(ns[1], 1, 0.1, "data/$(method_str)/")
)

# Auto-save to file
l2s, cns = convergence_run(method, ns, params;
                            orders   = orders,
                            geometry = geometry,
                            savefile = savefile)