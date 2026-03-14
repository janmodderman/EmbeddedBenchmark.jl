using EmbeddedBenchmark

geometry = :cylinder                # ← only change this if you want to benchmark a different geometry (currently only :cylinder is implemented, but you could add more in EmbeddedGeometry.jl and adjust the parameters accordingly in the SimulationParams constructor below)

paths = ["data/agfem_$(geometry).json", "data/cutfem_$(geometry).json"]

# paths = ["data/agfem_cylinder.json", "data/cutfem_cylinder.json",
#          "data/sbm_cylinder.json",   "data/wsbm_cylinder.json"]

cond_plots = plot_cond_from_files(paths)

display(cond_plots[1])   # order 1
display(cond_plots[2])   # order 2
