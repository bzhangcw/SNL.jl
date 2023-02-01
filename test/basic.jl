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

function loss(x::AbstractVector{T}) where {T}
    xv = reshape(x, 2, n - m)
    return least_square(n, m, xv, pp, Nx)
end

function loss(xv::Matrix{T}) where {T}
    return least_square(n, m, xv, pp, Nx)
end

function g(x::AbstractVector{T}) where {T}
    xv = reshape(x, 2, n - m)
    g = zeros(size(xv))
    for nx in Nx
        i, j = nx.edge
        xij = j < n - m + 1 ? xv[:, i] - xv[:, j] : xv[:, i] - pp[:, j]
        dh = xij .^ 2 |> sum
        ff = dh - nx.distn^2
        g[:, i] += 4 * ff * xij
        if nx.type == 'x'
            g[:, j] += -4 * ff * xij
        end
    end
    return reshape(g, size(x))
end

function H(x::AbstractVector{T}) where {T}
    xv = reshape(x, 2, n - m)
    nflat = length(x)
    Hblks = spzeros(nflat, nflat)
    for nx in Nx
        i, j = nx.edge
        xij = j < n - m + 1 ? xv[:, i] - xv[:, j] : xv[:, i] - pp[:, j]
        dh = xij .^ 2 |> sum
        M = 2 * xij * xij' + dh * I
        M1 = 4 * M - 4 * nx.distn^2 * I
        Hblks[idx(i, 2), idx(i, 2)] += M1
        if nx.type == 'x'
            Hblks[idx(j, 2), idx(j, 2)] += M1
            Hblks[idx(i, 2), idx(j, 2)] = -M1
        end
    end
    return Symmetric(Hblks, :U)
end


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