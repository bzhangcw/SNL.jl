
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

bool_opt = true
bool_plot = false
bool_parse = false

if bool_parse
    _args = parse_cmd()
else
    _args = parse_cmd(["--n", "80", "--m", "5", "--rho", "1.0"])
end
ρ = 0.2
d = 2
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
@printf("neighborhood (all) created with size: %10d\n", length(Nx))

# full batch loss, gradient, and Hessian
fa(x) = loss(x, pp, Nxview, nothing, keys(Nxview))
ga(x) = g(x, pp, Nxview, nothing, keys(Nxview))
Ha(x) = H(x, pp, Nxview, nothing, keys(Nxview))

# loss, gradient, and Hessian if degree is limited
Nxd = SNL.create_neighborhood(
    snldata["n"], snldata["m"], snldata["PP"], snldata["r"], snldata["nf"], snldata["deg"]
)
Nxdview = Dict(nx.edge => nx for nx in Nxd)
@printf("neighborhood (limited degree) created with size: %10d\n", length(Nxd))
# limited batch loss, gradient, and Hessian
fd(x) = loss(x, pp, Nxdview, nothing, keys(Nxdview))
gd(x) = g(x, pp, Nxdview, nothing, keys(Nxdview))
Hd(x) = H(x, pp, Nxdview, nothing, keys(Nxdview))

sample_func = sample_nodes_and_edges
r = Dict()

seed = _args[:start_seed]
Random.seed!(_args[:start_seed])


Xv = rand(Float64, (2, n - m))
x0 = reshape(Xv, length(Xv))


if bool_opt

    # X[@sprintf("Start",)] = Xv
    # color[@sprintf("Start")] = :black
    r[@sprintf("GD")] = optim_to_result(Optim.optimize(
        fa, ga, x0,
        GradientDescent(; alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.StrongWolfe()
        ),
        SNLOpt.options;
        inplace=false
    ))

    r[@sprintf("SGD")] = SGD(
        copy(x0), n, m, d, pp, Nxview, sample_edges; ρ=ρ,
        freq=100, maxiter=5000, α₀=2e-2, β=0.0, eval=fa
    )


    # r[@sprintf("BCD")] = SGD(
    #     copy(x0), n, m, d, pp, Nxview, sample_nodes_and_edges; ρ=ρ,
    #     freq=100, maxiter=5000, α₀=2e-2, β=0.0, eval=fa
    # )

    r[@sprintf("BCDf")] = SGD(
        copy(x0), n, m, d, pp, Nxview, sample_by_fixing_nodes; ρ=ρ,
        freq=100, maxiter=5000, α₀=2e-2, β=0.0, eval=fa
    )

    r[@sprintf("BCDf-r")] = SGD(
        copy(x0), n, m, d, pp, Nxview, sample_by_fixing_nodes; ρ=ρ,
        freq=100, maxiter=5000, α₀=2e-2, β=0.0, eval=fa, inneriter=2
    )

    # r[@sprintf("Adam")] = Adam(
    #     copy(x0), n, m, d, pp, Nxview, sample_func;
    #     ρ=ρ,
    #     freq=100, maxiter=5000, α₀=1e-2
    # )

    # r[@sprintf("SDRSOM")] = rsdrsom = SDRSOM(
    #     copy(x0), n, m, d, pp, Nxview, sample_func;
    #     ρ=ρ,
    #     maxiter=5000
    # )

end
if bool_plot

    pgfplotsx()

    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
    plot_function_value(n, m, r; smoothing=true, smoothing_range=15)
    fig, comments = plot_realization(n, m, pp, Nxd, r; heval=true, Ha=Ha, fa=fa, ga=ga)
    for k in comments
        println(k)
    end

end