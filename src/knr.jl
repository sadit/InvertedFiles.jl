# This file is a part of NeighborhoodApproximationIndex.jl

using Dates, InvertedFiles, Intersections, KCenters, StatsBase, Parameters, LinearAlgebra
export KnrIndex, index!, search, KnrOrderingStrategy, DistanceOrdering, InternalDistanceOrdering, DistanceOnTopKOrdering

mutable struct KnrOpt
    ksearch::Int32
end

abstract type KnrOrderingStrategy end

"""
    struct DistanceOrdering <: KnrOrderingStrategy end

Used as `ordering` parameter of `KnrIndex` specifies that each object `u` found by the inverted index will be evaluated against a given query.
This is the default ordering strategy.
"""
struct DistanceOrdering <: KnrOrderingStrategy end

"""
    struct InternalDistanceOrdering <: KnrOrderingStrategy end

Used as `ordering` parameter of `KnrIndex` specifies that the internal distance of the underlying inverted index will be used to ordering the `k` nearest neighbors.
Useful when the internal distance is quite representative of the real distance metric or whenever speed is the major objective.
"""
struct InternalDistanceOrdering <: KnrOrderingStrategy end

"""
    struct DistanceOnTopKOrdering <: KnrOrderingStrategy
        top::Int
    end

Used as `ordering` parameter of `KnrIndex` specifies that only the `top` elements are evaluated against the distance metric. Useful for very costly distance functions. If you are in doubt please use `DistanceOrdering` instead.
"""
mutable struct DistanceOnTopKOrdering <: KnrOrderingStrategy
    top::Int
end

"""
    struct KnrIndex <: AbstractSearchIndex

The K nearest references inverted index

# Parameters

- `dist`: the distance function of the index
- `db`: the database of indexed objects
- `centers`: a search index for a set of references
- `invfile`: an inverted file data structure
- `kbuild`: the number of references to be computed and stored by each indexed object
- `ordering`: specifies how the index performs final `k` nn selection
- `opt`: the parameters to be optimized by `optimize_index!`
"""
struct KnrIndex{
                KnrEncoder<:Knr,
                DataType<:AbstractDatabase,
                InvIndexType<:AbstractInvertedFile,
                OrderingType<:KnrOrderingStrategy
        } <: AbstractSearchIndex
    encoder::KnrEncoder
    db::DataType
    invfile::InvIndexType
    ordering::OrderingType
    opt::KnrOpt
end

function KnrIndex(I::KnrIndex; encoder=I.encoder, db=I.db, invfile=I.invfile, ordering=I.ordering, opt=I.opt)
    KnrIndex(encoder, db, invfile, ordering, opt)
end

Base.copy(I::KnrIndex; kwargs...) = KnrIndex(I; kwargs...)

@inline Base.length(idx::KnrIndex) = length(idx.invfile)
@inline SimilaritySearch.getcontext(idx::KnrIndex) = DEFAULT_CACHE_INVFILES[]
@inline SimilaritySearch.database(idx::KnrIndex) = idx.db
@inline SimilaritySearch.database(idx::KnrIndex, i) = idx.db[i]
@inline SimilaritySearch.distance(idx::KnrIndex) = distance(idx.encoder.refs)

Base.show(io::IO, idx::KnrIndex) = print(io, "{$(typeof(idx)), n=$(length(idx)), ordering=$(idx.ordering)}")

"""
    push_item!(idx::KnrIndex, ctx::InvertedFileContext, obj

Inserts `obj` into the indexed
"""
function SimilaritySearch.push_item!(idx::KnrIndex, ctx::InvertedFileContext, obj)
    res = encode_object_res!(idx.encoder, obj)
    idx.invfile isa WeightedInvertedFile && knr_as_similarity!(res)
    push_item!(idx.invfile, ctx, res)
    idx
end

function knr_as_similarity!(knr::KnnResult)
    #=
    for i in eachindex(knr)
        p = knr[i]
        knr[i] = IdWeight(p.id, 1/(0.5 + p.weight))
    end
    =#
end

"""
    get_parallel_block(n)

An heuristic to compute the `parallel_block` with respect with the number of elements to insert
"""
get_parallel_block(n) = min(n, 8 * Threads.nthreads())

"""
    append_items!(idx::KnrIndex, ctx::InvertedFileContext, db; kwargs...)


Appends all items in the database `db` into the index

# Arguments
- `idx`: the index structure
- `db`: the objects to be appended

# Keyword arguments
- `parallel_block`: the number of elements to be inserted in parallel
- `verbose`: controls the verbosity of the procedure
"""
function SimilaritySearch.append_items!(idx::KnrIndex, ctx::InvertedFileContext, db)
    append_items!(idx.db, db)
    index!(idx, ctx)
end

"""
    index!(idx::KnrIndex, ctx::InvertedFileContext)

Indexes all non indexed items in the database

# Arguments

- `idx`: the index structure
- `ctx`: the index's context

"""
function index!(idx::KnrIndex, ctx::InvertedFileContext)
    sp = length(idx) + 1
    n = length(idx.db)
    parallel_block = ctx.parallel_block
    E = [KnnResult(idx.encoder.k) for _ in 1:parallel_block]
    while sp < n
        ep = min(n, sp + parallel_block - 1)
    
        @batch minbatch=getminbatch(ctx.minbatch, ep-sp+1) per=thread for i in sp:ep
            res = reuse!(E[i - sp + 1])
            idx.invfile isa WeightedInvertedFile && knr_as_similarity!(res)
            encode_object_res!(idx.encoder, res, database(idx, i))
        end

        append_items!(idx.invfile, ctx, VectorDatabase(E), ep-sp+1)
        sp = ep + 1
    end

    idx
end

"""
    KnrIndex(
        db::AbstractDatabase,
        refs::AbstractSearchIndex;
        kbuild=4,
        ksearch=8,
        parallel_block::Integer=get_parallel_block(length(db)),
        verbose=false
    )

A convenient function to create a `KnrIndex`, it uses several default arguments. After the construction, use [`optimize_index!`](@ref) to adjust the index to some performance.

# Arguments
- `db`: The database of objects to be indexed. 
- `refs`: The index used for centers/references

# Keyword arguments
- `kbuild`: the number of references to compute and store on construction
- `ksearch`: the number of references to compute on searching
"""
function KnrIndex(
        db::AbstractDatabase,
        refs::AbstractSearchIndex;
        kbuild=4,
        ksearch=8,
    )
    
    encoder = Knr(UInt32, refs; k=kbuild)
    # other setups should be composed without calling this function
    invfile = BinaryInvertedFile(length(refs)) 
    ksearch = convert(Int32, ksearch)
    idx = KnrIndex(encoder, db, invfile, DistanceOrdering(), KnrOpt(ksearch))
    index!(idx, getcontext(idx))
end

function KnrIndex(
        db::AbstractDatabase,
        refs::AbstractSearchIndex,
        odist::OrdDistType, 
        ordering::KnrOrderingStrategy=DistanceOnTopKOrdering(1000);
        kbuild=4,
        ksearch=8
    ) where {OrdDistType<:DistancesForBinaryInvertedFile}
    
    encoder = Knr(UInt32, refs; k=kbuild)
    # other setups should be composed without calling this function
    invfile = BinaryInvertedFile(length(refs), odist) 
    ksearch = convert(Int32, ksearch)
    idx = KnrIndex(encoder, db, invfile, ordering, KnrOpt(ksearch))
    index!(idx, getcontext(idx))
end

function KnrIndex(
        db::AbstractDatabase,
        refs::AbstractSearchIndex,
        odist::OrdDistType,
        ordering::KnrOrderingStrategy=DistanceOnTopKOrdering(1000);
        kbuild=4,
        ksearch=8,
    ) where {OrdDistType<:Union{CosineDistance,NormalizedCosineDistance}}
    
    encoder = Knr(UInt32, refs; k=kbuild)
    # other setups should be composed without calling this function
    invfile = WeightedInvertedFile(length(refs)) 
    ksearch = convert(Int32, ksearch)
    idx = KnrIndex(encoder, db, invfile, ordering, KnrOpt(ksearch))
    index!(idx, getcontext(idx))
end

