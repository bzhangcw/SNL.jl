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

Base.@kwdef mutable struct StateOptim{R}
    fx::Float64
    ϵ::Float64
    t::Float64
    kf::Int = 0
    kg::Int = 0
    kH::Int = 0
    kh::Int = 0
    x::R = nothing
end
function optim_to_result(rr, name)
    traj = map(
        (x) -> StateOptim(fx=x.value, ϵ=x.g_norm, t=x.metadata["time"]), rr.trace
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
somoptions = Optim.Options(
    g_tol=1e-7,
    iterations=10000,
    store_trace=true,
    show_trace=true,
    show_every=5,
    time_limit=100
)

##################################
# VISUALIZATION TOOLS
##################################
VISCOLOR = Dict()
VISCOLOR[@sprintf("GD")] = :black
VISCOLOR[@sprintf("SGD")] = :green
VISCOLOR[@sprintf("BCD")] = :purple
VISCOLOR[@sprintf("BCDf")] = :purple2
VISCOLOR[@sprintf("Adam")] = :pink
VISCOLOR[@sprintf("SDRSOM")] = :red
function plot_realization(
    n, m, pp, Nxd, r;
    seed=1, heval::Bool=true, Ha=nothing, fa=nothing, ga=nothing
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
    for (_, (k, v)) in enumerate(r)

        if heval
            xx = reshape(v.state.x, :, n - m)
            xf = reshape(xx, length(xx))
            λ₁ = eigmin(Ha(xf) |> Matrix)
            ss = @sprintf("%12s, f:%+.1e, g:%+.1e, λ₁:%+.1e", string(k), fa(xf), ga(xf) |> norm, λ₁)
            push!(comments, ss)
            scatter!(
                fig, xx[1, :], xx[2, :], markercolor="grey99",
                markerstrokecolor=[VISCOLOR[k]],
                fillstyle=nothing, markershape=:circle, label=ss
            )
        else
            scatter!(
                fig, xx[1, :], xx[2, :], markercolor="grey99",
                markerstrokecolor=[VISCOLOR[k]],
                fillstyle=nothing, markershape=:circle, label=@sprintf("%s", string(k))
            )
        end
    end
    name = @sprintf("snl_%d_%d_%d", n, m, seed)
    savefig(fig, @sprintf("/tmp/%s.html", name))

    return fig, comments
end


function plot_function_value(
    n, m, r;
    smoothing=false, smoothing_range=5, pre::Symbol=:ppt, metric=:fx
)
    getresultfield(x, y=:fx) = getfield.(getfield(x, :trajectory), y)
    smoother(x) = movingaverage(x, smoothing_range)

    for xaxis in (:t, :k)

        @printf("plotting results\n")

        pgfplotsx()
        title = L"SNL, $n:=$%$(n), $m:=$%$(m)"
        fig = plot(
            xlabel=(xaxis == :t ? L"\textrm{Running Time (s)}" : L"\textrm{Iterations}"),
            ylabel=metric == :ϵ ? L"\|\nabla f\|" : L"f(x)",
            title=title,
            size=(600, 500),
            yticks=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e0, 1e1],
            # xticks=[1, 10, 100, 200, 500, 1000, 10000, 100000, 1e6],
            yscale=:log10,
            # xscale=:log2,
            dpi=500,
            labelfontsize=14,
            xtickfont=font(13),
            ytickfont=font(13),
            # leg=:bottomleft,
            legendfontsize=14,
            legendfontfamily="sans-serif",
            titlefontsize=22,
            # palette=:Paired_8
        )
        for (_, (k, rv)) in enumerate(r)
            yv = smoothing ? getresultfield(rv, metric) |> smoother : getresultfield(rv, metric)

            plot!(fig,
                xaxis == :t ? getresultfield(rv, :t) : (1:(yv|>length)),
                yv,
                label=k,
                linewidth=1.1,
                markershape=:circle,
                markersize=1.5,
                markercolor=:match,
                # linestyle=linestyles[k]
                # seriescolor=colors[k]
            )
        end
        savefig(fig, "/tmp/SNL-$n-$m-$xaxis.pdf")

    end
end