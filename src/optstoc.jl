using .SNL
using DRSOM
using LinearAlgebra
using Dates

SNLDEFAULT_TOL_ϵ₁ = 1e-14
SNLDEFAULT_TOL_ϵ₂ = 1e-6
drsom_stopping_criterion(
    state::DRSOM.DRSOMState, k; ϵ₁::F=SNLDEFAULT_TOL_ϵ₁, ϵ₂::F=SNLDEFAULT_TOL_ϵ₂, maxiter=1000
) where {F<:Real} =
    (state.fx <= ϵ₁) || (state.ϵ <= ϵ₂) || (k > maxiter)

default_rule_terminate(
    fx, gx, k; ϵ₁::F=SNLDEFAULT_TOL_ϵ₁, ϵ₂::F=SNLDEFAULT_TOL_ϵ₂, maxiter=1000
) where {F<:Real} = (fx <= ϵ₁) || (norm(gx) <= ϵ₂) || (k > maxiter)

Base.@kwdef mutable struct StocResult{T,R,F}
    x::R
    state::F
    trajectory::Vector{T}
end


# stochastic DRSOM
function SDRSOM(
    x0::AbstractVector{T},
    n::Int,
    m::Int,
    d::Int,
    points::Matrix{F},
    Nxview::Dict{E,Neighbor}, # the population
    sample_func::Function;
    maxiter=1000,
    freq=100,
    ρ=0.2,
    tol=1e-16,
    bool_keep_traj=true,
    bool_interpolation=false,
    eval=nothing
) where {T,E,F<:Real}
    ξ = sample_func(Nxview, ρ; nodes=1:n-m)
    # ξₕ = sample_func(Nxview, ρ * 5)
    fs(x) = loss(x, points, Nxview, nothing, ξ, nothing)
    gs(x) = g(x, points, Nxview, nothing, ξ, nothing)
    fsc = DRSOM.Counting(fs)
    gsc = DRSOM.Counting(gs)
    # hessians
    Hs(x) = H(x, points, Nxview, nothing, ξ, nothing)
    hsc = DRSOM.Counting(Hs)
    if bool_interpolation
        iter = DRSOM.DRSOMIteration(;
            x0=copy(x0), f=fsc, g=gsc, fog=:direct, sog=:direct
        )
    else
        iter = DRSOM.DRSOMIteration(;
            x0=copy(x0), f=fsc, g=gsc, H=hsc, fog=:direct, sog=:hess
        )
    end
    show(iter)
    arr = Vector{DRSOM.DRSOMState}()
    for (k, state::DRSOM.DRSOMState) in enumerate(iter)
        # state.γ = √(k + 1) * 1e-4
        bool_keep_traj && push!(arr, copy(state))
        ξ = sample_func(Nxview, ρ; nodes=1:n-m)
        if k % 100 == 0
            # ξₕ = sample_func(Nxview, ρ * 3)
        end
        if drsom_stopping_criterion(state, k; maxiter=maxiter)
            DRSOM.drsom_display(k, state)
            DRSOM.summarize(k, iter, state)
            return Result(name="SDRSOM", iter=iter, state=state, k=k, trajectory=arr)
            break
        end
        mod(k, freq) == 0 && DRSOM.drsom_display(k, state)
    end
end

# stochastic gradient descent
function SGD(
    x::AbstractVector{T},
    n::Int,
    m::Int,
    d::Int,
    points::Matrix{F},
    Nxview::Dict{E,Neighbor},
    sample_func::Function;
    maxiter=1000,
    inneriter=1,
    rule_terminate::Function=default_rule_terminate,
    freq=100,
    ls=false,
    ρ=0.2,
    α₀=1e-3,
    β=0.0, # the param for momentum
    eval=nothing
) where {T,E,F<:Real}
    k = 0
    τ = 1
    traj = Vector{StateOptim}()
    dₖ = zeros(size(x))
    t_start = Dates.now()
    samples = sample_func(Nxview; nodes=1:n-m)
    println("starting SGD")
    while true
        if mod(k, inneriter) == 0
            samples = sample_func(Nxview; nodes=1:n-m)
            τ += 1
            fs(x) = loss(x, points, Nxview, nothing, samples, nothing)
            gs(x) = g(x, points, Nxview, nothing, samples, nothing)
            global fsc = DRSOM.Counting(fs)
            global gsc = DRSOM.Counting(gs)
        end
        fx = fsc(x)
        gx = gsc(x)
        if ls
            α, fx, kₜ = HagerZhangLineSearch(fsc, gsc, gx, fx, x, -gx)
        else
            α = α₀ / sqrt(k + 1)
            kₜ = 1
        end
        if mod(k, freq) == 0 || (k < 20)
            @printf "f: %.1e, |g|: %.1e, α = %.1e, kₜ: %.2d, k: %.3d, τ: %.3d, |ξ|: %.3d\n" fx norm(gx) α kₜ k τ (samples |> length)
        end
        z = x
        try
            x = z - α .* gx + β .* dₖ
        catch
            println(α)
            println(β)
            println(gx)
            println(dₖ)
        end
        d = x - z
        k += 1
        push!(traj, StateOptim(
            x=x,
            fx=eval === nothing ? fx : eval(x),
            ϵ=norm(gx),
            t=(Dates.now() - t_start).value / 1e3)
        )
        if rule_terminate(fx, gx, k; maxiter=maxiter)
            @printf "f: %.1e, |g|: %.1e, α = %.1e, kₜ: %.2d, k: %.3d\n" fx norm(gx) α kₜ k
            return StocResult(x=x, state=StateOptim(x=x, fx=fx, ϵ=norm(gx), t=0.0), trajectory=traj)
        end
    end
end



@doc raw"""
 Adam: using the PyTorch style
```math
 x_{k+1} = x_k - \alpha g_k + \beta d_k
```
"""
function Adam(
    x::AbstractVector{T},
    n::Int,
    m::Int,
    d::Int,
    points::Matrix{F},
    Nxview::Dict{E,Neighbor},
    sample_func::Function;
    maxiter=1000,
    rule_terminate::Function=default_rule_terminate,
    freq=100,
    β=(0.9, 0.999),
    ρ=0.2,
    α₀=1e-3,
    eval=nothing
) where {T,E,F<:Real}
    k = 1
    traj = Vector{StateOptim}()
    M₁ = similar(x) * 0.0
    M₂ = similar(x) * 0.0
    while true
        samples = sample_func(Nxview, ρ; nodes=1:n-m)
        fs(x) = loss(x, points, Nxview, nothing, samples, nothing)
        gs(x) = g(x, points, Nxview, nothing, samples, nothing)
        fsc = DRSOM.Counting(fs)
        gsc = DRSOM.Counting(gs)
        fx = fsc(x)
        gx = gsc(x)
        # α = α₀ / sqrt(k)
        α = α₀
        kₜ = 1
        if mod(k, freq) == 0
            @printf "f: %.1e, |g|: %.1e, α = %.1e, kₜ: %.2d, k: %.3d\n" fx norm(gx) α kₜ k
        end
        M₁ = k == 1 ? gx : (1 - β[1]) * gx + β[1] * M₁
        M₂ = k == 1 ? (gx .^ 2) : (1 - β[2]) * (gx .^ 2) + β[2] * M₂
        fM₁ = M₁ / (1 - β[1]^(k))
        fM₂ = M₂ / (1 - β[2]^(k))
        x = x - α .* (fM₁ ./ fM₂)
        k += 1
        push!(traj, StateOptim(x=x, fx=fx, ϵ=norm(gx), t=0.0))
        if rule_terminate(fx, gx, k; maxiter=maxiter)
            @printf "f: %.1e, |g|: %.1e, α = %.1e, kₜ: %.2d, k: %.3d\n" fx norm(gx) α kₜ k
            return StocResult(x=x, state=StateOptim(x=x, fx=fx, ϵ=norm(gx), t=0.0), trajectory=traj)
        end
    end
end
