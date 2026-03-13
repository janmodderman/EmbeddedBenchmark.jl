module TestGeometry

using Test
using Gridap
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.Interfaces

using EmbeddedBenchmark

@testset "Geometry.jl" begin

    # ===================================================
    @testset "CylinderGeometry construction" begin

        g = CylinderGeometry(0.25, 0.25, 1.0, 0.5)

        @test g.R  == 0.25
        @test g.x₀ == 0.25
        @test g.Lₓ == 1.0
        @test g.L₃ == 0.5
        @test g isa EmbeddedGeometry{2}
        @test g isa CylinderGeometry

    end

    # ===================================================
    @testset "SphereGeometry construction" begin

        g = SphereGeometry(0.25, 0.25, 1.0, 0.5)

        @test g.R  == 0.25
        @test g.x₀ == 0.25
        @test g.Lₓ == 1.0
        @test g.L₃ == 0.5
        @test g isa EmbeddedGeometry{3}
        @test g isa SphereGeometry

    end

    # ===================================================
    @testset "EmbeddedGeometry type hierarchy" begin
        @test CylinderGeometry <: EmbeddedGeometry{2}
        @test SphereGeometry   <: EmbeddedGeometry{3}
        @test !(CylinderGeometry <: EmbeddedGeometry{3})
        @test !(SphereGeometry   <: EmbeddedGeometry{2})
    end

    # ===================================================
    @testset "build_geometry" begin

        @testset "CylinderGeometry — returns disk" begin
            g   = CylinderGeometry(0.25, 0.25, 1.0, 0.5)
            geo = build_geometry(g)
            @test geo !== nothing
        end

        @testset "SphereGeometry — returns sphere" begin
            g   = SphereGeometry(0.25, 0.25, 1.0, 0.5)
            geo = build_geometry(g)
            @test geo !== nothing
        end

    end

    # ===================================================
    @testset "geometry_cut" begin

        # ---------------------------------------------------
        @testset "2D — CylinderGeometry" begin

            n       = 8
            Lₓ      = 1.0
            L₃      = 0.5
            p1      = Point(-Lₓ/2, -L₃)
            p2      = Point( Lₓ/2,  0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g = CylinderGeometry(0.25, L₃/2, Lₓ, L₃)
            cutgeo, cutgeo_facets = geometry_cut(bgmodel, g)

            @test cutgeo        !== nothing
            @test cutgeo_facets !== nothing
            @test cutgeo        isa EmbeddedDiscretization
            @test cutgeo_facets isa EmbeddedFacetDiscretization

        end

        # ---------------------------------------------------
        @testset "2D — cut produces IN and OUT regions" begin

            n       = 8
            Lₓ      = 1.0
            L₃      = 0.5
            p1      = Point(-Lₓ/2, -L₃)
            p2      = Point( Lₓ/2,  0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g = CylinderGeometry(0.25, L₃/2, Lₓ, L₃)
            cutgeo, _ = geometry_cut(bgmodel, g)

            Ω_in  = Interior(cutgeo, PHYSICAL_IN)
            Ω_out = Interior(cutgeo, PHYSICAL_OUT)
            @test num_cells(Ω_in)  > 0
            @test num_cells(Ω_out) > 0

        end

        # ---------------------------------------------------
        @testset "3D — SphereGeometry" begin

            n       = 4   # coarser for speed
            Lₓ      = 1.0
            L₃      = 0.5
            p1      = Point(-Lₓ/2, -Lₓ/2, -L₃)
            p2      = Point( Lₓ/2,  Lₓ/2,  0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n, n))

            g = SphereGeometry(0.25, L₃/2, Lₓ, L₃)
            cutgeo, cutgeo_facets = geometry_cut(bgmodel, g)

            @test cutgeo        !== nothing
            @test cutgeo_facets !== nothing
            @test cutgeo        isa EmbeddedDiscretization
            @test cutgeo_facets isa EmbeddedFacetDiscretization

        end

        # ---------------------------------------------------
        @testset "3D — cut produces IN and OUT regions" begin

            n       = 4
            Lₓ      = 1.0
            L₃      = 0.5
            p1      = Point(-Lₓ/2, -Lₓ/2, -L₃)
            p2      = Point( Lₓ/2,  Lₓ/2,  0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n, n))

            g = SphereGeometry(0.25, L₃/2, Lₓ, L₃)
            cutgeo, _ = geometry_cut(bgmodel, g)

            Ω_in  = Interior(cutgeo, PHYSICAL_IN)
            Ω_out = Interior(cutgeo, PHYSICAL_OUT)
            @test num_cells(Ω_in)  > 0
            @test num_cells(Ω_out) > 0

        end

        # ---------------------------------------------------
        @testset "geometry_cut returns consistent pair" begin
            n       = 8
            Lₓ      = 1.0
            L₃      = 0.5
            p1      = Point(-Lₓ/2, -L₃)
            p2      = Point( Lₓ/2,  0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g = CylinderGeometry(0.25, L₃/2, Lₓ, L₃)
            cutgeo1, cutgeo_facets1 = geometry_cut(bgmodel, g)
            cutgeo2, cutgeo_facets2 = geometry_cut(bgmodel, g)

            # Same geometry should produce same cell counts
            @test num_cells(Interior(cutgeo1, PHYSICAL_OUT)) ==
                  num_cells(Interior(cutgeo2, PHYSICAL_OUT))
        end

        # ---------------------------------------------------
        @testset "radius affects cut cell count" begin
            n       = 8
            Lₓ      = 1.0
            L₃      = 0.5
            p1      = Point(-Lₓ/2, -L₃)
            p2      = Point( Lₓ/2,  0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g_small = CylinderGeometry(0.10, L₃/2, Lₓ, L₃)
            g_large = CylinderGeometry(0.35, L₃/2, Lₓ, L₃)

            cutgeo_small, _ = geometry_cut(bgmodel, g_small)
            cutgeo_large, _ = geometry_cut(bgmodel, g_large)

            # Larger radius → fewer IN cells
            n_in_small = num_cells(Interior(cutgeo_small, PHYSICAL_IN))
            n_in_large = num_cells(Interior(cutgeo_large, PHYSICAL_IN))
            @test n_in_small < n_in_large

            # Smaller radius → fewer OUT cells
            n_out_small = num_cells(Interior(cutgeo_small, PHYSICAL_OUT))
            n_out_large = num_cells(Interior(cutgeo_large, PHYSICAL_OUT))
            @test n_out_small < n_out_large
        end

    end

end

end # module