using MAT

using Random
using Printf
using LinearAlgebra
using SparseArrays
using ArgParse
using JuMP


function least_square(n, m, points, pp, Nx::NeighborVector)

    function _least_square(nx::Neighbor)
        i, j = nx.edge
        xij = j < n - m + 1 ? points[:, i] - points[:, j] : points[:, i] - pp[:, j]
        dh = xij .^ 2 |> sum
        return (dh - nx.distn^2)^2
    end
    val = _least_square.(Nx) |> sum
    return val
end

function idx(i::Int, dim::Int)
    return (i-1)*dim+1:i*dim
end


"""
return if an edge is sensor-to-sensor
    true if (i, j), i in N_x
- if anchors is :nothing, then use the neighborhood attribute
- o.w., you manually select the anchors, we redefine it.
"""
function bool_edge_is_x2x(
    i::Int, j::Int,
    nx::Neighbor,
    anchors::Union{Nothing,AbstractVector{Int}}=nothing
)
    anchors === nothing ? nx.type == 'x' : ((i ∉ anchors) & (j ∉ anchors))
end

function loss(
    x::AbstractVector{T},
    points::Matrix{G},
    Nxview::Dict{Tuple{Int,Int},Neighbor},
    randmethod::Union{Nothing,Function},
    edges=nothing,
    anchors::Union{Nothing,AbstractVector{Int}}=nothing
) where {T,G}
    xv = reshape(x, 2, :)
    (edges === nothing) && (edges = randmethod(Nxview))
    counts = edges |> length
    if counts == 0
        return 1e3
    end
    function _least_square(edge)
        nx = Nxview[edge]
        i, j = nx.edge
        xij = bool_edge_is_x2x(i, j, nx, anchors) ? xv[:, i] - xv[:, j] : xv[:, i] - points[:, j]
        dh = xij .^ 2 |> sum
        return (dh - nx.distn^2)^2
    end

    return _least_square.(edges) |> sum

end

function g(
    x::AbstractVector{T},
    points::Matrix{G},
    Nxview::Dict{Tuple{Int,Int},Neighbor},
    randmethod::Union{Nothing,Function},
    edges=nothing,
    anchors::Union{Nothing,AbstractVector{Int}}=nothing
) where {T,G}
    xv = reshape(x, 2, :)
    g = zeros(size(xv))
    (edges === nothing) && (edges = randmethod(Nxview))
    counts = edges |> length
    if counts == 0
        return zeros(size(x))
    end
    for edge in edges
        i, j = edge
        nx = Nxview[edge]
        xij = bool_edge_is_x2x(i, j, nx, anchors) ? xv[:, i] - xv[:, j] : xv[:, i] - points[:, j]
        dh = xij .^ 2 |> sum
        ff = dh - nx.distn^2
        g[:, i] += 4 * ff * xij
        if bool_edge_is_x2x(i, j, nx, anchors)
            g[:, j] += -4 * ff * xij
        end
    end
    return reshape(g, size(x))
end

function H(
    x::AbstractVector{T},
    points::Matrix{G},
    Nxview::Dict{Tuple{Int,Int},Neighbor},
    randmethod::Union{Nothing,Function},
    edges=nothing,
    anchors::Union{Nothing,AbstractVector{Int}}=nothing
) where {T,G}
    xv = reshape(x, 2, :)
    nflat = length(x)
    Hblks = spzeros(nflat, nflat)
    (edges === nothing) && (edges = randmethod(Nxview))
    counts = edges |> length
    if counts == 0
        return -1
    end
    for edge in edges
        i, j = edge
        nx = Nxview[edge]
        xij = bool_edge_is_x2x(i, j, nx, anchors) ? xv[:, i] - xv[:, j] : xv[:, i] - points[:, j]
        dh = xij .^ 2 |> sum
        M = 2 * xij * xij' + dh * I
        M1 = 4 * M - 4 * nx.distn^2 * I
        Hblks[idx(i, 2), idx(i, 2)] += M1
        if bool_edge_is_x2x(i, j, nx, anchors)
            Hblks[idx(j, 2), idx(j, 2)] += M1
            Hblks[idx(i, 2), idx(j, 2)] = -M1
        end
    end
    return Symmetric(Hblks, :U)
end
