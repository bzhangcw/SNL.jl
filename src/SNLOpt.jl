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



loss = SNL.loss
g = SNL.g
H = SNL.H
export parse_cmd
export Neighbor, NeighborVector
export parse_cmd, least_square
export loss, g, H
export SNL
export optim_to_result
# export optimization method
export SGD, SDRSOM, Adam
export plot_realization, plot_function_value
end