# This file is part of InvertedFiles.jl

module InvertedFiles
using Intersections
import SimilaritySearch:
    search, index!, getcontext, getknnresult, getminbatch, AbstractContext, KnnResult
using SimilaritySearch.AdjacencyLists
using Base.Threads: SpinLock
using Polyester

include("sortedintset.jl")
include("plists.jl")

struct InvertedFileContext{A,B,C,D} <: AbstractContext
    logger
    minbatch::Int
    parallel_block::Int
    positions::A
    cont_u32::B
    cont_iw::C
    cont_iiw::D
    knn::Vector{KnnResult}
end

function InvertedFileContext(
        ctx::InvertedFileContext;
        logger = ctx.loggger,                 
        minbatch = ctx.minbatch,
        parallel_block = ctx.parallel_block,
        positions=ctx.positions,
        cont_u32=ctx.cont_u32,
        cont_iw=ctx.cont_iw,
        cont_iiw=ctx.cont_iiw,
        knn=ctx.knn
    )
    InvertedFileContext(ctx, logger, minbatch, parallel_block, positions, cont_u32, cont_iw, cont_iiw, knn) 
end

function InvertedFileContext(; 
        logger = InformativeLog(),                 
        minbatch = 0,
        parallel_block = 256,
        positions = [Vector{UInt32}(undef, 32) for _ in 1:Threads.nthreads()],
        cont_u32 = [Vector{PostingList{Vector{UInt32}}}(undef, 32) for _ in 1:Threads.nthreads()],
        cont_iw = [Vector{PostingList{Vector{IdWeight}}}(undef, 32) for _ in 1:Threads.nthreads()],
        cont_iiw = [Vector{PostingList{Vector{IdIntWeight}}}(undef, 32) for _ in 1:Threads.nthreads()],
        knn = [KnnResult(32) for _ in 1:Threads.nthreads()]
    )
    InvertedFileContext(logger, minbatch, parallel_block, positions, cont_u32, cont_iw, cont_iiw, knn)
end

include("invfile.jl")
include("winvfile.jl")
include("binvfile.jl")
include("invfilesearch.jl")
include("winvfilesearch.jl")
include("binvfilesearch.jl")

include("knr.jl")
include("knrsearch.jl")
include("knropt.jl")
include("io.jl")

DEFAULT_CACHE_INVFILES = Ref(InvertedFileContext())

function __init__()
    n = Threads.nthreads()
    DEFAULT_CACHE_INVFILES[] = InvertedFileContext()
end

getcontext(invfile::AbstractInvertedFile) = DEFAULT_CACHE_INVFILES[]

end
