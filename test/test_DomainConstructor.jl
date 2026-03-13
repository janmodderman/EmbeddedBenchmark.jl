module TestDomain

using Test
using Gridap
using GridapEmbedded
using EmbeddedBenchmark

using GridapEmbedded.Interfaces
using Gridap.Geometry


# ===================================================
# DomainConfig tests
# ===================================================
@testset "Domain.jl" begin

    @testset "DomainConfig construction" begin

        @testset "default constructor" begin
            config = DomainConfig()
            @test config.side        == OUTSIDE
            @test config.intersected == true
            @test config.Γ₂_tags    == ["top"]
        end

        @testset "two-argument constructor" begin
            config = DomainConfig(INSIDE, false)
            @test config.side        == INSIDE
            @test config.intersected == false
            @test config.Γ₂_tags    == ["top"]
        end

        @testset "full constructor" begin
            config = DomainConfig(INSIDE, true, ["bottom"])
            @test config.side        == INSIDE
            @test config.intersected == true
            @test config.Γ₂_tags    == ["bottom"]
        end

        @testset "DomainSide values" begin
            @test INSIDE  isa DomainSide
            @test OUTSIDE isa DomainSide
            @test INSIDE  != OUTSIDE
        end

    end

    # ===================================================
    @testset "_get_flags" begin

        @testset "OUTSIDE intersected" begin
            f = EmbeddedBenchmark._get_flags(DomainConfig(OUTSIDE, true))
            @test f.physical_flag == PHYSICAL_OUT
            @test f.active_flag   == ACTIVE_OUT
            @test f.inactive_flag == ACTIVE_IN
            @test f.ghost_flag    == ACTIVE_OUT
            @test f.sbm_inner     == OUT
            @test f.sbm_cut       == CUT
            @test f.flip_normal   == true
        end

        @testset "OUTSIDE not intersected" begin
            f = EmbeddedBenchmark._get_flags(DomainConfig(OUTSIDE, false))
            @test f.physical_flag == PHYSICAL_OUT
            @test f.active_flag   == ACTIVE_OUT
            @test f.inactive_flag == IN            # differs from intersected case
            @test f.ghost_flag    == ACTIVE_OUT
            @test f.sbm_inner     == OUT
            @test f.sbm_cut       == CUT
            @test f.flip_normal   == true
        end

        @testset "INSIDE intersected" begin
            f = EmbeddedBenchmark._get_flags(DomainConfig(INSIDE, true))
            @test f.physical_flag == PHYSICAL_IN
            @test f.active_flag   == ACTIVE_IN
            @test f.inactive_flag == ACTIVE_OUT    # differs from OUTSIDE
            @test f.ghost_flag    == ACTIVE_IN
            @test f.sbm_inner     == IN
            @test f.sbm_cut       == CUT
            @test f.flip_normal   == false
        end

        @testset "INSIDE not intersected" begin
            f = EmbeddedBenchmark._get_flags(DomainConfig(INSIDE, false))
            @test f.physical_flag == PHYSICAL_IN
            @test f.active_flag   == ACTIVE_IN
            @test f.inactive_flag == OUT           # differs from intersected case
            @test f.ghost_flag    == ACTIVE_IN
            @test f.sbm_inner     == IN
            @test f.sbm_cut       == CUT
            @test f.flip_normal   == false
        end

    end

    # ===================================================
    # Domain struct tests — use a simple unit square geometry
    # ===================================================
    @testset "build_domain" begin

        # Build a simple background mesh and embedded geometry for testing
        R      = 0.25
        L      = 1.0
        n      = 8
        order  = 1

        p1     = Point(0.0, 0.0)
        p2     = Point(L,   L)
        bgmodel = CartesianDiscreteModel(p1, p2, (n, n))
        labeling = get_face_labeling(bgmodel)
        add_tag_from_tags!(labeling, "top",    [6])
        add_tag_from_tags!(labeling, "bottom", [5])
        add_tag_from_tags!(labeling, "left",   [7])
        add_tag_from_tags!(labeling, "right",  [8])

        geo     = disk(R, x0=Point(0.5, 0.5))
        cutgeo  = cut(bgmodel, geo)
        cutgeo_facets = cut_facets(bgmodel, geo)

        config_out = DomainConfig(OUTSIDE, true)
        config_in  = DomainConfig(INSIDE,  false)

        # ---------------------------------------------------
        @testset "Domain struct fields" begin
            domain = build_domain(AGFEM(), cutgeo, cutgeo_facets, config_out)
            @test domain isa Domain
            @test domain.Ω⁻    isa Triangulation
            @test domain.Ω⁻act isa Triangulation
            @test domain.Γ₁    isa Triangulation
            @test domain.nΓ₁   isa CellField
            @test domain.Γ₂    isa BoundaryTriangulation
            @test domain.nΓ₂   isa CellField
            @test domain.E⁰    === nothing
            @test domain.nE⁰   === nothing
            @test domain.Ωwsbm  === nothing
        end

        # ---------------------------------------------------
        @testset "AGFEM" begin
            domain = build_domain(AGFEM(), cutgeo, cutgeo_facets, config_out)
            @test domain.Ω⁻act isa Triangulation   # active domain present
            @test domain.E⁰    === nothing   # no ghost skeleton
            @test domain.Ωwsbm  === nothing   # no SBM domains
        end

        # ---------------------------------------------------
        @testset "CUTFEM" begin
            domain = build_domain(CUTFEM(), cutgeo, cutgeo_facets, config_out)
            @test domain.Ω⁻act isa Triangulation   # active domain present
            @test domain.E⁰    isa SkeletonTriangulation   # ghost skeleton present
            @test domain.nE⁰   isa SkeletonPair{<:CellField}   # ghost skeleton normal present
            @test domain.Ωwsbm  === nothing   # no SBM domains
        end

        # ---------------------------------------------------
        @testset "SBM" begin
            domain = build_domain(SBM(), cutgeo, nothing, config_out)
            @test domain.Ω⁻act isa Triangulation   # no separate active domain
            @test domain.E⁰    === nothing   # no ghost skeleton
            @test domain.Ωwsbm  === nothing   # no SBM domains
            @test domain.Γ₁    isa BoundaryTriangulation   # surrogate boundary present
        end

        # ---------------------------------------------------
        @testset "WSBM" begin
            domain = build_domain(WSBM(), cutgeo, nothing, config_out)
            @test domain.Ω⁻act === nothing   # no separate active domain
            @test domain.E⁰    isa SkeletonTriangulation   # ghost skeleton present
            @test domain.nE⁰   isa SkeletonPair{<:CellField}   # ghost skeleton normal present
            @test domain.Ωwsbm  isa Tuple{<:Triangulation, <:Triangulation}   # SBM domains present
            @test length(domain.Ωwsbm) == 2   # inner and cut domains
            @test domain.Ωwsbm[1] isa Triangulation
            @test domain.Ωwsbm[2] isa Triangulation
        end

        # ---------------------------------------------------
        @testset "build_reference_domain" begin
            ref = build_reference_domain(cutgeo, config_out)
            sbm = build_domain(SBM(), cutgeo, nothing, config_out)
            # Reference domain should match SBM domain structure
            @test ref isa Domain
            @test ref.Ω⁻act isa Triangulation
            @test ref.E⁰    === nothing
            @test ref.Ωwsbm  === nothing
        end

        # ---------------------------------------------------
        @testset "default config matches OUTSIDE intersected" begin
            domain_default = build_domain(AGFEM(), cutgeo, cutgeo_facets)
            domain_explicit = build_domain(AGFEM(), cutgeo, cutgeo_facets, DomainConfig(OUTSIDE, true))
            # Both should produce the same type structure
            @test typeof(domain_default) == typeof(domain_explicit)
        end

        # ---------------------------------------------------
        @testset "custom Γ₂ tags" begin
            config_bottom = DomainConfig(OUTSIDE, true, ["bottom"])
            domain = build_domain(AGFEM(), cutgeo, cutgeo_facets, config_bottom)
            @test domain.Γ₂  isa BoundaryTriangulation
            @test domain.nΓ₂ isa CellField
        end

    end

end

@testset "build_measures" begin

    n       = 8
    Lₓ      = 1.0
    L₃      = 0.5
    degree  = 2
    p1      = Point(-Lₓ/2, -L₃)
    p2      = Point( Lₓ/2,  0.0)
    bgmodel = CartesianDiscreteModel(p1, p2, (n, n))
    labeling = get_face_labeling(bgmodel)
    add_tag_from_tags!(labeling, "top", [6])

    geo     = disk(0.25, x0=Point(0.0, -0.25 + 0.25))
    cutgeo  = cut(bgmodel, geo)
    cutgeo_facets = cut_facets(bgmodel, geo)
    config  = DomainConfig(OUTSIDE, true)

    @testset "AGFEM — dE⁰ is Nothing" begin
        domain   = build_domain(AGFEM(), cutgeo, cutgeo_facets, config)
        measures = build_measures(domain, degree)
        @test measures isa Measures
        @test measures.dΩ⁻ isa Measure
        @test measures.dΓ₁ isa Measure
        @test measures.dΓ₂ isa Measure
        @test measures.dE⁰ === nothing
    end

    @testset "CUTFEM — dE⁰ is present" begin
        domain   = build_domain(CUTFEM(), cutgeo, cutgeo_facets, config)
        measures = build_measures(domain, degree)
        @test measures isa Measures
        @test measures.dΩ⁻ isa Measure
        @test measures.dΓ₁ isa Measure
        @test measures.dΓ₂ isa Measure
        @test measures.dE⁰ isa Measure
    end

    @testset "SBM — dE⁰ is Nothing" begin
        domain   = build_domain(SBM(), cutgeo, nothing, config)
        measures = build_measures(domain, degree)
        @test measures isa Measures
        @test measures.dΩ⁻ isa Measure
        @test measures.dΓ₁ isa Measure
        @test measures.dΓ₂ isa Measure
        @test measures.dE⁰ === nothing
    end

    @testset "WSBM — dE⁰ is present" begin
        domain   = build_domain(WSBM(), cutgeo, nothing, config)
        measures = build_measures(domain, degree)
        @test measures isa Measures
        @test measures.dΩ⁻ isa Measure
        @test measures.dΓ₁ isa Measure
        @test measures.dΓ₂ isa Measure
        @test measures.dE⁰ isa Measure
    end

    @testset "degree affects measure" begin
        domain    = build_domain(AGFEM(), cutgeo, cutgeo_facets, config)
        measures1 = build_measures(domain, 1)
        measures2 = build_measures(domain, 4)
        # Both valid — just verify construction succeeds at different degrees
        @test measures1.dΩ⁻ isa Measure
        @test measures2.dΩ⁻ isa Measure
    end

end

end # module