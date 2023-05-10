using DRSOM
using Optim
using LazyStack
using Printf
using Plots
using Statistics: mean
movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]
# results utilities
getresultfield(x, y=:fx) = getfield.(getfield(x, :trajectory), y)
getname(x) = getfield(x, :name) |> string
geteps(x) = x.g_norm

Base.@kwdef mutable struct StateOptim
    fx::Float64
    ϵ::Float64
    t::Float64
    kf::Int = 0
    kg::Int = 0
    kH::Int = 0
    kh::Int = 0
end
function optim_to_result(rr, name)
    traj = map(
        (x) -> StateOptim(fx=x.value, ϵ=x.g_norm, t=rr.time_run), rr.trace
    )
    traj[end].kf = rr.f_calls
    traj[end].kg = rr.g_calls
    traj[end].kH = rr.h_calls
    return Result(name=name, iter=rr, state=traj[end], k=rr.iterations, trajectory=traj)
end


# deterministic options
options = Optim.Options(
    g_tol=1e-7,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=100,
    time_limit=100
)

##################################
# VISUALIZATION TOOLS
##################################
function plot_realization(
    n, m, pp, Nxd, X, color;
    seed=1, heval::Bool=true, Ha=Ha, fa=fa, ga=ga
)
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
    name = @sprintf("snl_%d_%d_%d", n, m, seed)
    savefig(fig, @sprintf("/tmp/%s.html", name))

    return fig, comments
end


function plot_function_value(n, m, r; smoothing=false, smoothing_range=5, pre::Symbol=:ppt)
    getresultfield(x, y=:fx) = getfield.(getfield(x, :trajectory), y)
    smoother(x) = movingaverage(x, smoothing_range)
    if smoothing
        method_objval_ragged = rstack(
            r["SGD"][end] |> smoother,
            r["Adam"][end] |> smoother,
            getresultfield(r["SDRSOM"], :fx) |> smoother;
            fill=NaN
        )
    else
        method_objval_ragged = rstack(
            r["SGD"][end],
            r["Adam"][end],
            getresultfield(r["SDRSOM"], :fx)
            ; fill=NaN
        )
    end
    method_names = ["SGD"; "ADAM"; "SDRSOM"]

    @printf("plotting results\n")
    title = "SNL model"
    fig = plot(
        1:(method_objval_ragged|>size|>first),
        method_objval_ragged,
        label=permutedims(method_names),
        xscale=:log10,
        yscale=:log10,
        xlabel=L"$k$: iteration",
        ylabel=L"$\mathbb E f(x; \xi)$",
        title=title,
        size=pre == :ppt ? (1080, 720) : (1100, 500),
        yticks=[1e-10, 1e-6, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
        xticks=[1, 10, 100, 200, 500, 1000, 2000, 5000, 10000, 5e4, 1e5, 5e5, 1e6],
        dpi=500,
        xtickfont=font(13),
        ytickfont=font(13),
        legendfontsize=14,
        legendfontfamily="sans-serif",
        titlefontsize=16,
    )

    savefig(fig, @sprintf("/tmp/SNL-%s-%s.pdf", n, m))

end