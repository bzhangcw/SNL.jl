# include("../src/snl.jl")
# include("../src/tools.jl")

using Plots
using Printf
using Random
using ArgParse
using LinearAlgebra
using ReverseDiff
using Test
using LineSearches
using Optim
using DRSOM
using StatsBase
using LaTeXStrings
using LazyStack
using SNLOpt

# if true
# _args = parse_cmd(["--n", "8", "--m", "2"])
# else
_args = parse_cmd()
# end
if _args[:c] == 1
    snldata = SNL.create_snl_data(_args)
else
    snldata = matread(_args[:fp])
end
n, m, nf, pp = snldata["n"], snldata["m"], snldata["nf"], snldata["PP"]
@printf("finished loading\n")

# allowing all edges available to us
# set degree to n => maximum possible
Nx = SNL.create_neighborhood(
    snldata["n"], snldata["m"], snldata["PP"], snldata["r"], snldata["nf"], snldata["n"]
)
Nxview = Dict(nx.edge => nx for nx in Nx)
@printf("neighborhood (all) created with size: %.3e\n", length(Nx))

# full batch loss, gradient, and Hessian
fa(x) = loss(x, pp, Nxview, nothing, keys(Nxview))
ga(x) = g(x, pp, Nxview, nothing, keys(Nxview))
Ha(x) = H(x, pp, Nxview, nothing, keys(Nxview))

# loss, gradient, and Hessian if degree is limited
Nxd = SNL.create_neighborhood(
    snldata["n"], snldata["m"], snldata["PP"], snldata["r"], snldata["nf"], snldata["deg"]
)
Nxdview = Dict(nx.edge => nx for nx in Nxd)
@printf("neighborhood (limited degree) created with size: %.3e\n", length(Nxd))
# full batch loss, gradient, and Hessian
fd(x) = loss(x, pp, Nxdview, nothing, keys(Nxdview))
gd(x) = g(x, pp, Nxdview, nothing, keys(Nxdview))
Hd(x) = H(x, pp, Nxdview, nothing, keys(Nxdview))

# sampling method
# uniform sampling from all edges
ρ = _args[:rho]
function sample_func(Nxview, ratio::Float64=0.2)
    return [k for (k, _) in Nxview if rand(Float64) >= 1 - ratio]
end

color = Dict()
r = Dict()
X = Dict()

seed = _args[:start_seed]
Random.seed!(_args[:start_seed])


Xv = rand(Float64, (2, n - m))
x0 = reshape(Xv, length(Xv))
X[@sprintf("Start",)] = Xv
color[@sprintf("Start")] = :black
optim_r_gd = Optim.optimize(
    fa, ga, x0,
    GradientDescent(; alphaguess=LineSearches.InitialHagerZhang(),
        linesearch=LineSearches.StrongWolfe()
    ),
    SNLOpt.options;
    inplace=false
)
r[@sprintf("GD")] = optim_to_result(optim_r_gd, :GD)
X[@sprintf("GD")] = Xvg = reshape(optim_r_gd.minimizer, 2, :)
color[@sprintf("GD")] = :blue


r[@sprintf("SGD")] = SGD(
    copy(x0), pp, Nxview, sample_func;
    freq=100, maxiter=5000, α₀=1e-2, β=0.0
)

r[@sprintf("Adam")] = Adam(
    copy(x0), pp, Nxview, sample_func;
    freq=100, maxiter=5000, α₀=1e-2
)

r[@sprintf("SDRSOM")] = rsdrsom = SDRSOM(
    copy(x0), pp, Nxview, sample_func;
    maxiter=5000
)

X[@sprintf("SGD")] = Xsgd = reshape(r[@sprintf("SGD")][1], 2, :)
color[@sprintf("SGD")] = :green
X[@sprintf("Adam")] = Xsgd = reshape(r[@sprintf("Adam")][1], 2, :)
color[@sprintf("Adam")] = :pink
X[@sprintf("SDRSOM")] = Xvdrsom = reshape(rsdrsom.state.x, 2, :)
color[@sprintf("SDRSOM")] = :red



pgfplotsx()

push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
plot_function_value(n, m, r; smoothing=true, smoothing_range=15)
fig, comments = plot_realization(n, m, pp, Nxd, X, color; heval=true, Ha=Ha, fa=fa, ga=ga)
for k in comments
    println(k)
end

