module TestGeometry

using Test
using Gridap
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.Interfaces

using EmbeddedBenchmark

# ===================================================
# Test helper
# ===================================================
function _test_distances(geo, bgmodel, Γ, fun, degree, pmid, R)

    dist_data = compute_distances(SBM(), bgmodel, geo, Γ, fun, degree, 0.0)
    QΓ          = CellQuadrature(Γ, degree)
    phys_points = get_cell_points(QΓ).cell_phys_point
    N           = length(pmid)  # spatial dimension

    @testset "type consistency" begin
        @test dist_data isa DistanceData
        @test dist_data.d    isa CellState
        @test dist_data.n    isa CellState
        @test dist_data.Xd   isa CellState
        @test dist_data.fsbm isa CellState
        @test dist_data.Xd.values isa Vector
    end

    @testset "Xd = x + d" begin
        for icell in 1:length(phys_points)
            for ipoint in 1:length(phys_points[icell])
                x  = phys_points[icell][ipoint]
                d  = dist_data.d.values[icell][ipoint]
                Xd = dist_data.Xd.values[icell][ipoint]
                for i in 1:N
                    @test Xd[i] ≈ x[i] + d[i]   atol=1e-14
                end
            end
        end
    end

    @testset "normal is unit vector and consistent with d" begin
        for icell in 1:length(phys_points)
            for ipoint in 1:length(phys_points[icell])
                n    = dist_data.n.values[icell][ipoint]
                d    = dist_data.d.values[icell][ipoint]
                absn = sqrt(n ⋅ n)
                absd = sqrt(d ⋅ d)
                @test absn ≈ 1.0   atol=1e-14
                for i in 1:N
                    @test n[i] ≈ d[i] / absd   atol=1e-14
                end
            end
        end
    end

    @testset "Xd lies on geometry boundary" begin
        for icell in 1:length(phys_points)
            for ipoint in 1:length(phys_points[icell])
                Xd   = dist_data.Xd.values[icell][ipoint]
                δ    = Xd - pmid
                @test sqrt(δ ⋅ δ) ≈ R   atol=1e-10
            end
        end
    end

    @testset "fsbm ≈ fun(Xd) ≈ fun(x+d)" begin
        for icell in 1:length(phys_points)
            for ipoint in 1:length(phys_points[icell])
                x          = phys_points[icell][ipoint]
                d          = dist_data.d.values[icell][ipoint]
                Xd         = dist_data.Xd.values[icell][ipoint]
                fsbm_val   = dist_data.fsbm.values[icell][ipoint]
                fun_at_xpd = fun(x + d)
                fun_at_Xd  = fun(Xd)
                for i in 1:N
                    @test fsbm_val[i] ≈ fun_at_xpd[i]   atol=1e-14
                    @test fsbm_val[i] ≈ fun_at_Xd[i]    atol=1e-14
                end
            end
        end
    end
end

@testset "Geometry.jl" begin

    # ===================================================
    @testset "CylinderGeometry construction" begin

        g = CylinderGeometry(0.25, VectorValue(0.0, 0.0))

        @test g.R  == 0.25
        @test g.x₀ == VectorValue(0.0, 0.0)
        @test g isa EmbeddedGeometry{2}
        @test g isa CylinderGeometry

    end

    # ===================================================
    @testset "SphereGeometry construction" begin

        g = SphereGeometry(0.25, VectorValue(0.0, 0.0, 0.0))

        @test g.R  == 0.25
        @test g.x₀ == VectorValue(0.0, 0.0, 0.0)   
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
            g   = CylinderGeometry(0.25, VectorValue(0.0,0.0))
            geo = build_geometry(g)
            @test geo !== nothing
        end

        @testset "SphereGeometry — returns sphere" begin
            g   = SphereGeometry(0.25, VectorValue(0.0, 0.0, 0.0))
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
            x₀      = VectorValue(0.0, 0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g = CylinderGeometry(0.25, x₀)
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
            x₀      = VectorValue(0.0, 0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g = CylinderGeometry(0.25, x₀)
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
            x₀      = VectorValue(0.0, 0.0, 0.0)          
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n, n))

            g = SphereGeometry(0.25, x₀)
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
            x₀      = VectorValue(0.0, 0.0, 0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n, n))

            g = SphereGeometry(0.25, x₀)
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
            x₀      = VectorValue(0.0, 0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g = CylinderGeometry(0.25, x₀)
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
            x₀      = VectorValue(0.0, 0.0)
            bgmodel = CartesianDiscreteModel(p1, p2, (n, n))

            g_small = CylinderGeometry(0.10, x₀)
            g_large = CylinderGeometry(0.35, x₀)

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

    @testset "Compute Distances" begin
        @testset "compute_distances SBM — 2D Cylinder" begin
            n       = 6
            Lₓ, L₃, R = 1.0, 0.5, 0.25
            pmid    = VectorValue(0.0, 0.0)
            bgmodel = CartesianDiscreteModel(Point(-Lₓ/2, -L₃), Point(Lₓ/2, 0.0), (n, n))
            geo     = CylinderGeometry(R, pmid)
            cutgeo, _ = geometry_cut(bgmodel, geo)
            Γ       = Interface(Interior(cutgeo, ACTIVE_IN), Interior(cutgeo, OUT)).⁻
            fun     = x -> VectorValue(x[1], x[2])
            _test_distances(geo, bgmodel, Γ, fun, 2, pmid, R)
        end

        @testset "compute_distances SBM — 3D Sphere" begin
            n       = 6
            Lₓ, L₃, R = 1.0, 0.5, 0.25
            pmid    = VectorValue(0.0, 0.0, 0.0)
            bgmodel = CartesianDiscreteModel(Point(-Lₓ/2, -Lₓ/2, -L₃), Point(Lₓ/2, Lₓ/2, 0.0), (n, n, n))
            geo     = SphereGeometry(R, pmid)
            cutgeo, _ = geometry_cut(bgmodel, geo)
            Γ       = Interface(Interior(cutgeo, ACTIVE_IN), Interior(cutgeo, OUT)).⁻
            fun     = x -> VectorValue(x[1], x[2], x[3])
            _test_distances(geo, bgmodel, Γ, fun, 2, pmid, R)
        end
    end
end 
end # module