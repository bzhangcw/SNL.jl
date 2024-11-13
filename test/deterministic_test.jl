
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
using ADNLPModels
using SNLOpt

bool_generate = true
bool_parse = true
bool_opt = true
bool_plot = true

if bool_generate
    if bool_parse
        _args = parse_cmd()
    else
        _args = parse_cmd(["--n", "200", "--m", "20", "--rho", "1.0", "--r", "1.4", "--deg", "20"])
    end

    # end
    if _args[:c] == 1
        snldata = SNL.create_snl_data(_args)
    else
        snldata = matread(_args[:fp])
    end
    n, m, nf, pp = snldata["n"], snldata["m"], snldata["nf"], snldata["PP"]

    Xv = rand(Float64, (2, n - m))
    x0 = reshape(Xv, length(Xv))
    @info("finished loading")


    # loss, gradient, and Hessian if degree is limited
    Nxd = SNL.create_neighborhood(
        snldata["n"], snldata["m"], snldata["PP"], snldata["r"], snldata["nf"], snldata["deg"]
    )
    Nxdview = Dict(nx.edge => nx for nx in Nxd)
    @info("neighborhood (limited degree) created with size: $(length(Nxd))")
    # limited batch loss, gradient, and Hessian
    fd(x) = loss(x, pp, Nxdview, nothing, keys(Nxdview))
    gd(x) = g(x, pp, Nxdview, nothing, keys(Nxdview))
    Hd(x) = H(x, pp, Nxdview, nothing, keys(Nxdview))
    hvp(x, v, w) = SNLOpt.hvp!(x, v, w, pp, Nxdview, nothing, keys(Nxdview))

    function ADNLPModels.gradient!(adbackend::SNL.NewADGradient, w, f, x)
        copyto!(w, gd(x))
    end
    @time begin
        @info("create a SNL for NLPModels.jl API")
        nlp = ADNLPModel(fd, copy(x0), gradient_backend=SNL.NewADGradient)
    end

    seed = _args[:start_seed]
    Random.seed!(_args[:start_seed])


end

if bool_opt
    r = Dict()
    # X[@sprintf("Start",)] = Xv
    # color[@sprintf("Start")] = :black
    r[@sprintf("GD")] = optim_to_result(Optim.optimize(
            fd, gd, x0,
            GradientDescent(; alphaguess=LineSearches.InitialHagerZhang(),
                linesearch=LineSearches.StrongWolfe()
            ),
            SNLOpt.options;
            inplace=false
        ), "GD"
    )

    r[@sprintf("Newton-TR")] = optim_to_result(Optim.optimize(
            fd, gd, Hd, x0,
            NewtonTrustRegion(;
            ),
            SNLOpt.somoptions;
            inplace=false
        ), "Newton-TR"
    )
    # rarc = SNLOpt.ArC(nlp, max_time=100.0,
    #     max_iter=1000,
    #     max_eval=typemax(Int64),
    #     verbose=true)

    # r[@sprintf("DRSOM")] = DRSOM2(name=Symbol("DRSOM"))(;
    #     x0=copy(x0), f=fd, g=gd,
    #     maxiter=10000, freq=10,
    #     maxtime=10000
    # )
    # r[@sprintf("HSODM")] = HSODM(name=Symbol("HSODM"))(;
    #     x0=copy(x0), f=fd, g=gd, hvp=hvp,
    #     maxiter=10000, freq=1,
    #     maxtime=10000, tol=1e-8,
    #     direction=:warm, linesearch=:hagerzhang,
    #     adaptive=:none
    # )
    # r[@sprintf("Newton-CG-LS")] = PFH(name=Symbol("Newton"))(;
    #     x0=copy(x0), f=fd, g=gd, hvp=hvp,
    #     maxiter=10000, freq=1,
    #     step=:newton, μ₀=0.0,
    #     bool_trace=true,
    #     maxtime=10000,
    #     direction=:warm
    # )
    ##############################################################
    # Do not plot DRSOM
    ##############################################################

    r[@sprintf("DRSOM")] = DRSOM2(name=Symbol("DRSOM"))(;
        x0=copy(x0), f=fd, g=gd,
        maxiter=10000, freq=10,
        maxtime=10000
    )

end
if bool_plot

    pgfplotsx()

    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
    plot_function_value(n, m, r; smoothing=false, smoothing_range=15, metric=:ϵ)
    fig, comments = plot_realization(n, m, pp, Nxd, r; heval=false)
    for k in comments
        println(k)
    end

end