using NLPModels


mutable struct NLPSNLModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
end

