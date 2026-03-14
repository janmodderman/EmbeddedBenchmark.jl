using EmbeddedBenchmark

method   = SBM()                  # ← only change this: AGFEM(), CUTFEM(), SBM(), WSBM()
geometry = :cylinder                # ← only change this if you want to benchmark a different geometry (currently only :cylinder is implemented, but you could add more in EmbeddedGeometry.jl and adjust the parameters accordingly in the SimulationParams constructor below)

# ===================================================
# Auto-derived from method — do not change below
# ===================================================
method_str = lowercase(string(typeof(method)))
savefile   = "data/$(method_str)_$(geometry).json"

bar_plots = plot_bar_from_file(savefile; normalized=false, quantity=:time)
for (i, p) in enumerate(bar_plots)
    display(p)
end

bar_plots2 = plot_bar_from_file(savefile; normalized=false, quantity=:alloc)
for (i, p) in enumerate(bar_plots2)
    display(p)
end