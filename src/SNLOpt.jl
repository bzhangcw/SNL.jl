module SNLOpt


using ArgParse
using ReverseDiff
using Test
using Random
using LaTeXStrings

include("./snl.jl")
include("./tools.jl")
include("./opt.jl")
include("./optstoc.jl")
include("./sampling.jl")
include("./nlp.jl")


loss = SNL.loss
g = SNL.g
H = SNL.H
hvp = SNL.H
hvp! = SNL.hvp!
export parse_cmd
export Neighbor, NeighborVector
export parse_cmd, least_square
export loss, g, H, hvp, hvp!
export SNL
export optim_to_result
# export optimization method
export SGD, SDRSOM, Adam
export plot_realization, plot_function_value
export sample_edges, sample_nodes_and_edges, sample_by_fixing_nodes
end