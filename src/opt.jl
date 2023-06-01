using AdaptiveRegularization
using Stopping
using NLPModels

ht, pt, sm, ka = (AdaptiveRegularization.HessOp,
    AdaptiveRegularization.PDataKARC,
    AdaptiveRegularization.solve_modelKARC,
    [:shifts => 10.0 .^ (collect(-20.0:1.0:20.0))]
)
# fun = :ARCqKOp
# @eval begin
#     function $fun(nlpstop::NLPStopping; kwargs...)
#         kw_list = Dict{Symbol,Any}()
#         if $ka != ()
#             for t in $ka
#                 push!(kw_list, t)
#             end
#         end
#         merge!(kw_list, Dict(kwargs))
#         MyTRARC(nlpstop; hess_type=$ht, pdata_type=$pt, solve_model=$sm, kw_list...)
#     end
# end
# @eval begin
#     function $fun(nlp::AbstractNLPModel{T,S}; kwargs...) where {T,S}
#         nlpstop = NLPStopping(nlp; optimality_check=(pb, state) -> norm(state.gx), kwargs...)
#         nlpstop = $fun(nlpstop; kwargs...)
#         return nlpstop
#     end
# end




function ArC(nlp::AbstractNLPModel{T,S}; kwargs...) where {T,S}
    nlpstop = NLPStopping(nlp; optimality_check=(pb, state) -> norm(state.gx), kwargs...)

    nlpstop = TRARC(nlpstop; hess_type=ht, pdata_type=pt, solve_model=sm, kwargs...)
    return nlpstop
end
