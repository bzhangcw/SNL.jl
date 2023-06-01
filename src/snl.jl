__precompile__()

"""
file: snl.jl
project: SNL
created Date: Apr 2022
author: Chuwen Zhang
-----
last Modified: Wed Apr 20 2022
modified By: Chuwen Zhang
-----
(c) 2022 Chuwen Zhang, Yinyu Ye
-----
Module for modeling Sensor Network Localization problem using SDP relaxation and DRSOM to minimize the second-stage nonlinear least-square problem
We suggest the user to go through [1] for complete description of this problem.

References:
1. Wang, Z., Zheng, S., Ye, Y., Boyd, S.: Further relaxations of the semidefinite programming approach to sensor network localization. SIAM Journal on Optimization. 19, 655–673 (2008)
2. Biswas, P., Lian, T.-C., Wang, T.-C., Ye, Y.: Semidefinite programming based algorithms for sensor network localization. ACM Transactions on Sensor Networks (TOSN). 2, 188–220 (2006)
3. Biswas, P., Liang, T.-C., Toh, K.-C., Ye, Y., Wang, T.-C.: Semidefinite programming approaches for sensor network localization with noisy distance measurements. IEEE transactions on automation science and engineering. 3, 360–371 (2006)
4. Biswas, P., Ye, Y.: Semidefinite programming for ad hoc wireless sensor network localization. In: Third International Symposium on Information Processing in Sensor Networks, IPSN 2004. pp. 46–54 (2004)
"""
module SNL

using MAT

using Random
using Printf
using LinearAlgebra
using SparseArrays
using ArgParse
using JuMP

try
    using MosekTools
    function SDR(n, m, nf::Real, pp::Matrix{Float64}, Nx::NeighborVector, edges::Dict)::Tuple
        # we build standard SDP relaxation
        solver = Mosek.Optimizer
        model = Model(solver)
        dd = n - m
        e = [ones(dd, 1); sum(pp[:, dd+1:n], dims=2)] / sqrt(n)
        C = 1.4 * nf * (e * e' - I(dd + 2))
        @variable(model, Z[1:dd+2, 1:dd+2], PSD)
        @variable(model, wp[keys(edges)] >= 0)
        @variable(model, wn[keys(edges)] >= 0)
        @constraint(model, Z[1:2, 1:2] .== [1 0; 0 1])
        for nx in Nx
            @constraint(model,
                LinearAlgebra.tr(Z * nx.vec * nx.vec') + wp[nx.edge] - wn[nx.edge] == nx.distn)
        end
        @objective(model, Min, sum(wp) + sum(wn) + tr(C * Z))
        @printf("SDR built finished \n")
        optimize!(model)

        Zv = value.(Z)
        Yv = Zv[3:end, 3:end]
        Xv = Zv[1:2, 3:end]

        @printf("SDR residual trace(Y - X'X): %.2e\n", Yv - Xv' * Xv |> LinearAlgebra.tr)
        return Zv, Yv, Xv
    end
catch e
    @warn("Unable to import JuMP and MosekTools, in which case you cannot use SDP relaxation")
    @warn(" If you intend to use SDR, then you should add JuMP and MosekTools as requirements")
end


Base.@kwdef mutable struct Neighbor
    edge::Tuple{Int,Int}
    type::Char
    vec::Vector{Float64}
    dist::Float64
    distn::Float64
end

NeighborVector = Vector{Neighbor}

function ei(i, n)
    ev = zeros(n)
    ev[i] = 1
    return ev
end

function create_snl_data(param::Dict)
    # create random data
    snldata = Dict()
    Random.seed!(param[:seed]) # for reproducibility
    snldata["m"] = param[:m]
    snldata["n"] = param[:n]
    snldata["PP"] = rand(Float64, (2, param[:n])) .- 0.5
    snldata["r"] = param[:r]
    snldata["nf"] = param[:nf]
    snldata["deg"] = param[:degree]
    matwrite(@sprintf("/tmp/test%d-%d.mat", param[:m], param[:n]), snldata)
    @info """
        generating sensor-network localization instance with 
        - m (anchors)                          := $(snldata["m"])
        - n (sensors [inc. anchors])           := $(snldata["n"])
        - r (radius)                           := $(snldata["r"])
        - nf (noise factor)                    := $(snldata["nf"])
        - deg (maximum degree for each node)   := $(snldata["deg"])
    """
    return snldata
end

function create_neighborhood(n, m, pp, radius, nf, degree)
    Random.seed!(2) # reset seed
    Nx = NeighborVector()
    if Threads.nthreads() > 1
        Threads.@threads for i::Int = 1:n-m
            flag = 0
            for j::Int = i+1:n
                rr = norm(pp[:, i] - pp[:, j])
                distn = rr * sqrt(max(0, (1 + randn() * nf)))
                if rr < radius && flag < degree
                    flag = flag + 1
                    if j <= n - m
                        nxv = [0; 0; ei(i, n - m) - ei(j, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='x', vec=nxv, dist=rr, distn=distn))
                    else
                        nav = [-pp[:, j]; ei(i, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='a', vec=nav, dist=rr, distn=distn))
                    end
                end
            end
        end
        return Nx
    else
        for i::Int = 1:n-m
            flag = 0
            for j::Int = i+1:n
                rr = norm(pp[:, i] - pp[:, j])
                distn = rr * sqrt(max(0, (1 + randn() * nf)))
                if rr < radius && flag < degree
                    flag = flag + 1
                    if j <= n - m
                        nxv = [0; 0; ei(i, n - m) - ei(j, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='x', vec=nxv, dist=rr, distn=distn))
                    else
                        nav = [-pp[:, j]; ei(i, n - m)]
                        push!(Nx, Neighbor(edge=(i, j), type='a', vec=nav, dist=rr, distn=distn))
                    end
                end
            end
        end
        return Nx
    end

end

include("diff.jl")

s = ArgParseSettings(
    description="""
        A script for the Sensor Network Localization 
            using optionally SDP relaxation 
          and DRSOM to minimize the second-stage nonlinear least-square
          see snl.jl for more descriptions.
        if you create a instance that is very large, it is suggested that you use:
            `julia -t` to invoke this script
    """,
    exit_after_help=true,
    preformatted_description=true,
    preformatted_epilog=true
)

@add_arg_table s begin
    "--c"
    help = "if true, then create a new instance"
    arg_type = Int
    default = 1
    "--seed"
    arg_type = Int
    default = 1
    help = "random seed if you create a new instance"
    "--start_seed"
    arg_type = Int
    default = 5
    help = """
    random seed if you use a random start point
    """
    "--runs"
    arg_type = Int
    default = 1
    help = """
    # of trial runs (different random start)
    """
    "--fp"
    arg_type = String
    help = "read the instance"
    "--n"
    arg_type = Int
    default = 80
    help = "total number of sensors (including the anchors)"
    "--m"
    arg_type = Int
    default = 5
    help = "total number of anchors, (suggested #: n/30)"
    "--degree"
    arg_type = Int
    default = 5
    help = "degree, the number of edges for a sensor. (suggested #: n/20)"
    "--r"
    arg_type = Float64
    default = 2.0
    help = "radio range"
    "--nf"
    arg_type = Float64
    default = 0.0
    help = "noice factor"
    "--option_plot_js"
    arg_type = Int
    help = "bool use h5+js to visualize"
    default = 1
    "--option_use_sdr"
    arg_type = Int
    help = "bool use SDP relaxation"
    default = 0
    "--option_set_comparison"
    arg_type = String
    action = :append_arg
    nargs = '*'
    help = """compare to other algorithms, options: 
    - gd: Gradient Descent (with LS)
    - cg: Conjugate Gradient (Hager-Zhang)"""
    "--timelimit"
    arg_type = Int
    default = 300
    # new for stochastic
    "--rho"
    arg_type = Float64
    default = 0.2
end

function parse_cmd(manual_args=nothing)
    _args = manual_args !== nothing ? parse_args(manual_args, SNL.s, as_symbols=true) : parse_args(SNL.s, as_symbols=true)
    display(_args)
    return _args
end

export Neighbor, NeighborVector
export parse_cmd, least_square
export idx
export loss, g, H, hvp, hvp!
export SDR
export create_neighborhood, create_snl_data

end # module
