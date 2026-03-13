module TestAnalyticalSolutions

using Test
using Gridap
using StaticArrays
using ForwardDiff
using EmbeddedBenchmark

# ===================================================
# Test parameters
# ===================================================
const g_test  = 9.81
const k_test  = 2π
const η₀_test = 0.05
const d_test  = 0.5
const t_test  = 0.5

const s2D = AirySolution2D(g_test, k_test, η₀_test, d_test)
const s3D = AirySolution3D(g_test, k_test, η₀_test, d_test)

const ω_test = s2D.ω   # sqrt(g*k*tanh(k*d))

# ===================================================
# Test points
# ===================================================
const points_2D = [
    Point(0.0,   0.0),
    Point(0.5,  -0.5),
    Point(1.0,  -1.0),
    Point(π/4,  -0.25),
]

const points_3D = [
    Point(0.0,  0.0,   0.0),
    Point(0.5, -0.5,   0.3),
    Point(1.0, -1.0,   0.5),
]

# ===================================================
# Helper: analytical gradient and Laplacian for Airy
# ===================================================
function airy_gradient_2D(s::AirySolution{2}, x, t)
    VectorValue(
        s.η₀*s.g/s.ω * s.k * (cosh(s.k*(x[2]+s.d)) / cosh(s.k*s.d)) *  cos(s.k*x[1] - s.ω*t),
        s.η₀*s.g/s.ω * s.k * (sinh(s.k*(x[2]+s.d)) / cosh(s.k*s.d)) *  sin(s.k*x[1] - s.ω*t)
    )
end

function airy_gradient_3D(s::AirySolution{3}, x, t)
    VectorValue(
        s.η₀*s.g/s.ω * s.k * (cosh(s.k*(x[3]+s.d)) / cosh(s.k*s.d)) * cos(s.k*x[1] + s.k*x[2] - s.ω*t),
        s.η₀*s.g/s.ω * s.k * (cosh(s.k*(x[3]+s.d)) / cosh(s.k*s.d)) * cos(s.k*x[1] + s.k*x[2] - s.ω*t),
        s.η₀*s.g/s.ω * s.k * (sinh(s.k*(x[3]+s.d)) / cosh(s.k*s.d)) * sin(s.k*x[1] + s.k*x[2] - s.ω*t)
    )
end

# ===================================================
@testset "AnalyticalSolutions.jl" begin

    # ---------------------------------------------------
    @testset "AirySolution construction" begin

        @testset "2D" begin
            @test s2D.g  == g_test
            @test s2D.k  == k_test
            @test s2D.η₀ == η₀_test
            @test s2D.d  == d_test
            @test s2D.ω  ≈  sqrt(g_test * k_test * tanh(k_test * d_test))
        end

        @testset "3D" begin
            @test s3D.g  == g_test
            @test s3D.k  == k_test
            @test s3D.η₀ == η₀_test
            @test s3D.d  == d_test
            @test s3D.ω  ≈  sqrt(g_test * k_test * tanh(k_test * d_test))
        end

        @testset "dispersion relation enforced" begin
            @test s2D.ω ≈ sqrt(s2D.g * s2D.k * tanh(s2D.k * s2D.d))
            @test s3D.ω ≈ sqrt(s3D.g * s3D.k * tanh(s3D.k * s3D.d))
        end

        @testset "invalid dimension throws" begin
            @test_throws ArgumentError AirySolution{1}(g_test, k_test, η₀_test, d_test)
            @test_throws ArgumentError AirySolution{4}(g_test, k_test, η₀_test, d_test)
        end

    end

    # ---------------------------------------------------
    @testset "AirySolution callable" begin

        @testset "2D — VectorValue input" begin
            for x in points_2D
                val = s2D(x, t_test)
                expected = η₀_test*g_test/ω_test * (cosh(k_test*(x[2]+d_test))/cosh(k_test*d_test)) * sin(k_test*x[1] - ω_test*t_test)
                @test val ≈ expected atol=1e-12
            end
        end

        @testset "2D — SVector input (ForwardDiff compatibility)" begin
            for x in points_2D
                xv = SVector(x[1], x[2])
                @test s2D(xv, t_test) ≈ s2D(x, t_test) atol=1e-12
            end
        end

        @testset "3D — VectorValue input" begin
            for x in points_3D
                val = s3D(x, t_test)
                expected = η₀_test*g_test/ω_test * (cosh(k_test*(x[3]+d_test))/cosh(k_test*d_test)) * sin(k_test*x[1] + k_test*x[2] - ω_test*t_test)
                @test val ≈ expected atol=1e-12
            end
        end

        @testset "3D — SVector input (ForwardDiff compatibility)" begin
            for x in points_3D
                xv = SVector(x[1], x[2], x[3])
                @test s3D(xv, t_test) ≈ s3D(x, t_test) atol=1e-12
            end
        end

    end

    # ---------------------------------------------------
    @testset "manufactured_functions" begin

        @testset "2D" begin
            u, ∇u, Δu = manufactured_functions(s2D, t_test)

            @testset "solution value" begin
                for x in points_2D
                    @test u(x) ≈ s2D(x, t_test) atol=1e-12
                end
            end

            @testset "gradient matches analytical" begin
                for x in points_2D
                    ∇u_analytical = airy_gradient_2D(s2D, x, t_test)
                    ∇u_computed   = ∇u(x)
                    @test ∇u_computed[1] ≈ ∇u_analytical[1] atol=1e-6
                    @test ∇u_computed[2] ≈ ∇u_analytical[2] atol=1e-6
                end
            end

            @testset "Laplacian is zero (Airy satisfies Laplace)" begin
                for x in points_2D
                    @test Δu(x) ≈ 0.0 atol=1e-6
                end
            end

            @testset "gradient returns VectorValue{2}" begin
                @test ∇u(first(points_2D)) isa VectorValue{2}
            end

            @testset "solution returns scalar" begin
                @test u(first(points_2D)) isa Real
            end
        end

        @testset "3D" begin
            u, ∇u, Δu = manufactured_functions(s3D, t_test)

            @testset "solution value" begin
                for x in points_3D
                    @test u(x) ≈ s3D(x, t_test) atol=1e-12
                end
            end

            @testset "gradient matches analytical" begin
                for x in points_3D
                    ∇u_analytical = airy_gradient_3D(s3D, x, t_test)
                    ∇u_computed   = ∇u(x)
                    @test ∇u_computed[1] ≈ ∇u_analytical[1] atol=1e-6
                    @test ∇u_computed[2] ≈ ∇u_analytical[2] atol=1e-6
                    @test ∇u_computed[3] ≈ ∇u_analytical[3] atol=1e-6
                end
            end

            @testset "Laplacian (Airy does not match Laplace: ∇²u = -k²u - k²u + k²u = -k²u ≠ 0)" begin
                for x in points_3D
                    Δu_computed   = Δu(x)
                    Δu_analytical = s3D.k^2 * u(x)
                    @test Δu_computed ≈ Δu_analytical atol=1e-6
                end
            end

            @testset "gradient returns VectorValue{3}" begin
                @test ∇u(first(points_3D)) isa VectorValue{3}
            end

            @testset "solution returns scalar" begin
                @test u(first(points_3D)) isa Real
            end
        end

    end

end
end # module 