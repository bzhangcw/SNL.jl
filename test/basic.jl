include("../src/snl.jl")

using Plots
using Printf
using Random
using ArgParse
using LinearAlgebra
using SparseArrays
using ReverseDiff
using Test
using .SNL


_args = parse_cmd()
if _args[:c] == 1
    snldata = SNL.create_snl_data(_args)
else
    snldata = matread(_args[:fp])
end
n, m, nf, pp = snldata["n"], snldata["m"], snldata["nf"], snldata["PP"]
@printf("finished loading\n")
Nx = SNL.create_neighborhood(
    snldata["n"], snldata["m"], snldata["PP"], snldata["r"], snldata["nf"], snldata["deg"]
)
edges = Dict(nx.edge => nx.distn for nx in Nx)
@printf("neighborhood created with size: %.3e\n", length(Nx))

Xv = zeros(2, n - m)
x = reshape(Xv, length(Xv))

@testset "Basic Algebra: Gradient" begin
    for _ in 1:5
        x1 = randn(Float64, length(Xv))
        g1 = g(x1)
        g2 = ReverseDiff.gradient(loss, x1)
        @test (abs.(g1 - g2) |> maximum) < 1e-8
    end
end


@testset "Basic Algebra: Hessian" begin
    for _ in 1:5
        x1 = randn(Float64, length(Xv))
        h1 = H(x1)
        h2 = ReverseDiff.hessian(loss, x1)
        @test (abs.(h1 - h2) |> maximum) < 1e-8
    end
end