# This file is a part of NeighborhoodApproximationIndex.jl

using SearchModels, Random, StatsBase
import SearchModels: combine, mutate
import SimilaritySearch: optimize!, MinRecall, ParetoRecall, ParetoRadius, ErrorFunction, setconfig!, runconfig, optimization_space
export optimize!, KnrOptSpace

@with_kw struct KnrOptSpace <: AbstractSolutionSpace
    ksearch = 1:3:21
    ksearch_scale = (s=1.1, p1=0.8, p2=0.8, lower=1, upper=128)
end

Base.hash(c::KnrOpt) = hash(c.ksearch)
Base.isequal(a::KnrOpt, b::KnrOpt) = a.ksearch == b.ksearch
Base.eltype(::KnrOptSpace) = KnrOpt
Base.rand(space::KnrOptSpace) = KnrOpt(rand(space.ksearch))
 
combine(a::KnrOpt, b::KnrOpt) = KnrOpt((a.ksearch + b.ksearch) ÷ 2)
mutate(sp::KnrOptSpace, a::KnrOpt, iter) = KnrOpt(SearchModels.scale(a.ksearch; sp.ksearch_scale...))

optimization_space(index::KnrIndex) = KnrOptSpace()

function setconfig!(c::KnrOpt, index::KnrIndex, perf)
    index.opt.ksearch = c.ksearch
end

function runconfig(c::KnrOpt, index::KnrIndex, q, res::KnnResult, pools)
    search(index, q, res; pools, ksearch=c.ksearch)
end
