include("../src/snl.jl")

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
using .SNL


_args = parse_cmd()
if _args[:c] == 1
    snldata = SNL.create_snl_data(_args)
else
    snldata = matread(_args[:fp])
end
n, m, nf, pp = snldata["n"], snldata["m"], snldata["nf"], snldata["PP"]
@printf("finished loading\n")

# allowing all edges available to us
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
function sample1(Nxview, ratio::Float64=0.2)
    return [k for (k, _) in Nxview if rand(Float64) >= 1 - ratio]
end


function run(x::AbstractVector{T}, ls=true; randanchor=false, maxiter::Int=1000, α₀=1e-3, ρ=0.2, freq=100) where {T}
    k = 1
    while true
        samples = sample1(Nxview, ρ)
        anchors = randanchor ? sample(1:n, m) : nothing
        fs(x) = loss(x, pp, Nxview, nothing, samples, anchors)
        gs(x) = g(x, pp, Nxview, nothing, samples, anchors)
        fx = fs(x)
        gx = gs(x)
        # println(gx - ReverseDiff.gradient(fs, x))
        # @assert (abs.(gx - ReverseDiff.gradient(fs, x)) |> maximum) < 1e-4
        # univariate line search functions
        s = similar(gx)
        s = -gx
        if ls
            α, fx, kₜ = HagerZhangLineSearch(fs, gs, gx, fx, x, s)
        else
            α = α₀ / sqrt(k)
            kₜ = 1
        end
        if mod(k, freq) == 1
            @printf "f: %.1e, |g|: %.1e, α = %.1e, kₜ: %.2d, k: %.3d\n" fx norm(gx) α kₜ k
        end
        x = x + α .* s
        k += 1
        if norm(gx) < 1e-6 || k >= maxiter
            return x, k
        end
    end
end

color = Dict()
r = Dict()
X = Dict()

# deterministic
options = Optim.Options(
    g_tol=1e-7,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=100,
    time_limit=100
)
Random.seed!(_args[:start_seed])
for seed in 1:_args[:runs]

    Xv = rand(Float64, (2, n - m))
    x0 = reshape(Xv, length(Xv))
    X[@sprintf("Start@%s", seed)] = Xv
    color[@sprintf("Start@%s", seed)] = :black
    r[@sprintf("GD@%s", seed)] = optim_r_gd = Optim.optimize(
        fd, gd, x0,
        GradientDescent(; alphaguess=LineSearches.InitialHagerZhang(),
            linesearch=LineSearches.StrongWolfe()
        ),
        options;
        inplace=false
    )

    X[@sprintf("GD@%s", seed)] = Xvg = reshape(optim_r_gd.minimizer, 2, :)
    color[@sprintf("GD@%s", seed)] = :blue
    # optim_r_gd = Optim.optimize(
    #     fa, ga, -x0,
    #     GradientDescent(; alphaguess=LineSearches.InitialHagerZhang(),
    #         linesearch=LineSearches.StrongWolfe()
    #     ),
    #     options;
    #     inplace=false
    # )
    # Xvgr = reshape(optim_r_gd.minimizer, 2, :)


    r[@sprintf("HSODM@%s", seed)] = rhsodm = HSODM()(x0=copy(x0), f=fd, g=gd, H=Hd,
        maxiter=10000, tol=1e-8, freq=1,
        direction=:warm
    )
    X[@sprintf("HSODM@%s", seed)] = Xvhsodm = reshape(rhsodm.state.x, 2, :)
    color[@sprintf("HSODM@%s", seed)] = :red

    # stochastic
    r[@sprintf("SGD@%s", seed)] = (x, k) = run(copy(x0), false; freq=1000, maxiter=50000, α₀=1e-1, ρ=ρ)
    X[@sprintf("SGD@%s", seed)] = Xsgd = reshape(x, 2, :)
    color[@sprintf("SGD@%s", seed)] = :green
end

function plot(heval::Bool=true)
    plotly()
    fig = scatter(
        pp[1, 1:n-m], pp[2, 1:n-m],
        markershape=:xcross,
        markerstrokecolor=[:black],
        markersize=4,
        markerstrokewidth=1.5,
        label="Truth",
        legendfontsize=24,
        tickfontsize=16,
        size=(1080, 960),
    )

    scatter!(fig, pp[1, n-m+1:n], pp[2, n-m+1:n], markershape=:utriangle, markersize=4, markerstrokewidth=0.1, label="Anchors")
    edgesx = []
    edgesy = []
    for (idx, nx) in enumerate(Nxd[1:end-1])
        i, j = nx.edge
        nxn = Nxd[idx+1]
        i1, _ = nxn.edge
        push!(edgesx, pp[1, i])
        push!(edgesx, pp[1, j])
        push!(edgesx, pp[1, i1])
        push!(edgesy, pp[2, i])
        push!(edgesy, pp[2, j])
        push!(edgesy, NaN64)
    end
    plot!(fig, edgesx, edgesy, linestyle=:dashdot, label="edges")
    comments = []
    for (_, (k, xx)) in enumerate(X)

        if heval
            xf = reshape(xx, length(xx))
            λ₁ = eigmin(Ha(xf) |> Matrix)
            ss = @sprintf("%12s, f:%+.1e, g:%+.1e, λ₁:%+.1e", string(k), fa(xf), ga(xf) |> norm, λ₁)
            push!(comments, ss)
            scatter!(
                fig, xx[1, :], xx[2, :], markercolor="grey99",
                markerstrokecolor=[color[k]],
                fillstyle=nothing, markershape=:circle, label=ss
            )
        else
            scatter!(
                fig, xx[1, :], xx[2, :], markercolor="grey99",
                markerstrokecolor=[color[k]],
                fillstyle=nothing, markershape=:circle, label=@sprintf("%s", string(k))
            )
        end
    end
    name = @sprintf("snl_%d_%d_%d", n, m, _args[:seed])
    savefig(fig, @sprintf("/tmp/%s.html", name))

    return fig, comments
end

fig, comments = plot()
for k in comments
    println(k)
end