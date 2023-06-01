using Random


# sampling method
# uniform sampling from all edges

function sample_edges(Nxview, ratio::Float64=0.2; nodes=nothing, ξmax=500)
    cc = [k for (k, _) in Nxview if rand(Float64) >= 1 - ratio]
    cl = min(cc |> length, ξmax)
    return Random.shuffle(cc)[1:cl]
end


function sample_nodes_and_edges(Nxview, ratio::Float64=0.2; pₑ=0.8, nodes=nothing, ξmax=500)

    _nn = [k for k in nodes if rand(Float64) >= 1 - ratio]
    bool_select(i, j, v) = ((i ∈ _nn) || (j ∈ _nn)) && (rand(Float64) >= 1 - pₑ)
    cc = [(i, j) for ((i, j), v) in Nxview if bool_select(i, j, v)]
    cl = min(cc |> length, ξmax)
    return Random.shuffle(cc)[1:cl]


end


function sample_by_fixing_nodes(Nxview, ratio::Float64=0.7; pₑ=0.8, nodes=nothing, ξmax=500)

    _nn = [k for k in nodes if rand(Float64) >= 1 - ratio]
    bool_select(i, j, v) = ((i ∉ _nn) && (j ∉ _nn)) && (rand(Float64) >= 1 - pₑ)
    cc = [(i, j) for ((i, j), v) in Nxview if bool_select(i, j, v)]
    cl = min(cc |> length, ξmax)
    return Random.shuffle(cc)[1:cl]
end