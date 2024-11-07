module test_timing
using Gridap
using GridapEmbedded
using DrWatson
using DataFrames:DataFrame
using DataFrames:Matrix
using TimerOutputs
using LinearAlgebra
using CSV

include("../src/case_setup.jl")
include("../src/agfem.jl")
include("../src/cutfem.jl")
include("../src/sbm.jl")
include("../src/wsbm.jl")

# _, _, _, _, to1 = ModAgfem.agfem("cylinder",[256])
# show(to1)

# _, _, _, _, to2 = ModCutfem.cutfem("cylinder",[256])
# show(to2)

# _, _, _, _, to3 = ModSbm.sbm("cylinder",[256])
# show(to3)
# _, _, _, _, _ = ModWsbm.wsbm("cylinder",[8])

_, _, _, _, _ = ModWsbm.wsbm("sphere_stl",[32])
# _, _, _, _, to4 = ModWsbm.wsbm("cylinder",[512])


# _, _, _, _, _ = ModCutfem.cutfem("cylinder",[8])

# _, _, _, _, _ = ModCutfem.cutfem("cylinder",[512])
# _, _, _, _, to4 = ModCutfem.cutfem("cylinder",[512])
# show(to4)

end # module