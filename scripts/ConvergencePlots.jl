using EmbeddedBenchmark

geometry = :cylinder                # ← only change this if you want to benchmark a different geometry (currently only :cylinder is implemented, but you could add more in EmbeddedGeometry.jl and adjust the parameters accordingly in the SimulationParams constructor below)

# # Load later
# method, l2s, cns, nₓ_vec, orders = load_convergence("data/convergence_agfem.json")

# # Plot directly from file
# l2_plots  = plot_L2_from_files(["data/convergence_agfem.json",
#                                     "data/convergence_cutfem.json"])

# paths = ["data/agfem_$geometry.json", "data/cutfem_$geometry.json",
#             "data/sbm_$geometry.json",   "data/wsbm_$geometry.json"]

paths = ["data/convergence_agfem_$geometry.json", "data/convergence_cutfem_$geometry.json",
            "data/convergence_sbm_$geometry.json",   "data/convergence_wsbm_$geometry.json"]

l2_plots   = plot_L2_from_files(paths)

display(l2_plots[1])   # order 1
display(l2_plots[2])   # order 2
