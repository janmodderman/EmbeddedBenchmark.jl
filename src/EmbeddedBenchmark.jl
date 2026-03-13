module EmbeddedBenchmark
using Gridap
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
# using ForwardDiff
# using StaticArrays
using STLCutters
using LinearAlgebra
using JSON3
using Colors

# Parameters.jl
export EmbeddingMethod, AGFEM, CUTFEM, SBM, WSBM
export GeometryParams, GeometryParams2D, GeometryParams3D
export ManufacturedParams, SolverParams, SimulationParams
export setup_model

# ManufacturedSolutions.jl
export ManufacturedSolution, AirySolution
export AirySolution2D, AirySolution3D
export manufactured_functions

# SolutionConstructor.jl
export build_solution

# DomainConstructor.jl
export DomainSide, INSIDE, OUTSIDE
export DomainConfig
export Domain, build_domain, build_reference_domain
export Measures, build_measures

# EmbeddedGeometry.jl
export EmbeddedGeometry, CylinderGeometry, SphereGeometry
export build_geometry, geometry_cut

# FESpaces.jl
export FESpaceConfig, FESpaces, build_spaces

# WeakForms.jl
export WeakForm, build_weak_form

# BenchmarkRunner.jl
export benchmark, print_benchmark_results
export method_categories

export TAG_COLORS, FALLBACK_COLOR
export plot_bar, plot_bar!
export save_benchmark, load_benchmark, plot_bar_from_file
export plot_L2_from_files, plot_cond_from_files


include("Parameters.jl")
include("ManufacturedSolutions.jl")
include("SolutionConstructor.jl")
include("DomainConstructor.jl")
include("EmbeddedGeometry.jl")
include("FESpaces.jl")
include("WeakForms.jl")
include("BenchmarkRunner.jl")
include("PostProcessing.jl")

end # module EmbeddedBenchmark