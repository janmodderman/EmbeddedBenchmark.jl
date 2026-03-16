using EmbeddedBenchmark

geometry = :sphere                # ← only change this if you want to benchmark a different geometry (currently only :cylinder is implemented, but you could add more in EmbeddedGeometry.jl and adjust the parameters accordingly in the SimulationParams constructor below)

paths = ["data/convergence_agfem_$geometry.json", "data/convergence_cutfem_$geometry.json",
            "data/convergence_sbm_$geometry.json",   "data/convergence_wsbm_$geometry.json"]

l2_plots   = plot_L2_from_files(paths)

display(l2_plots[1])   # order 1
display(l2_plots[2])   # order 2
