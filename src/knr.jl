# This file is a part of NeighborhoodApproximationIndex.jl

import SimilaritySearch: search, getpools, index!
using InvertedFiles, Intersections, KCenters, StatsBase, Parameters, LinearAlgebra
export KnrIndex, index!, search, KnrOrderingStrategies, DistanceOrdering, InternalDistanceOrdering, DistanceOnTopKOrdering

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
    struct KnrIndex <: AbstractSearchContext

The K nearest references inverted index

# Parameters

- `dist`: the distance function of the index
- `db`: the database of indexed objects
- `centers`: a search index for a set of references
- `invfile`: an inverted file data structure
- `kbuild`: the number of references to be computed and stored by each indexed object
- `ordering`: specifies how the index performs final `k` nn selection
- `opt`: the parameters to be optimized by `optimize!`
"""
struct KnrIndex{
            DistType<:SemiMetric,
            DataType<:AbstractDatabase,
            CentersIndex<:AbstractSearchContext,
            InvIndexType<:AbstractInvertedFile,
            OrderingType<:KnrOrderingStrategy
        } <: AbstractSearchContext
    dist::DistType
    db::DataType
    centers::CentersIndex
    invfile::InvIndexType
    kbuild::Int32
    ordering::OrderingType
    opt::KnrOpt
end

@inline Base.length(idx::KnrIndex) = length(idx.invfile)
Base.show(io::IO, idx::KnrIndex) = print(io, "{$(typeof(idx)) centers=$(typeof(idx.centers)), n=$(length(idx)), ordering=$(idx.ordering)}")

"""
    struct KnrCaches
        enc
    end
    
Caches used for `KnrIndex` (one per thread)

# Properties
- `enc`: `KnnResult` for encoding purposes
"""
struct KnrCaches
    enc::KnnResult
end

const GlobalKnrCachesPool = Vector{KnrCaches}(undef, 0)

function __init__knr()
    n = Threads.nthreads()

    while length(GlobalKnrCachesPool) < n
        push!(GlobalKnrCachesPool, KnrCaches(KnnResult(10)))
    end
end

getpools(idx::KnrIndex) = GlobalKnrCachesPool

function getencodeknnresult(k::Integer, pools::Vector{KnrCaches})
    reuse!(pools[Threads.threadid()].enc, k)
end

"""
    push!(idx::KnrIndex, obj; pools=getpools(idx), encpools=getpools(idx.centers))

Inserts `obj` into the indexed
"""
function Base.push!(idx::KnrIndex, obj; pools=getpools(idx), encpools=getpools(idx.centers))
    res = getencodeknnresult(idx.kbuild, pools)
    search(idx.centers, obj, res; pools=encpools)
    push!(idx.invfile, res)
    idx
end

"""
    get_parallel_block(n)

An heuristic to compute the `parallel_block` with respect with the number of elements to insert
"""
get_parallel_block(n) = min(n, 8 * Threads.nthreads())

"""
    append!(idx::KnrIndex, db; <kwargs>)


Appends all items in the database `db` into the index

# Arguments
- `idx`: the index structure
- `db`: the objects to be appended

# Keyword arguments
- `parallel_block`: the number of elements to be inserted in parallel
- `pools`: unused argument
- `verbose`: controls the verbosity of the procedure
"""
function Base.append!(idx::KnrIndex, db;
        parallel_block=get_parallel_block(length(db)),
        pools=getpools(idx),
        verbose=true
    )

    append!(idx.db, db)
    index!(idx; parallel_block, pools, verbose)
end

"""
    index!(idx::KnrIndex; parallel_block=get_parallel_block(length(idx.db)), pools=nothing, verbose=true)

Indexes all non indexed items in the database

# Arguments

- `idx`: the index structure

# Keyword arguments
- `parallel_block`: the number of elements to be inserted in parallel
- `pools`: unused parameter
- `verbose`: controls verbosity of the procedure

"""
function index!(idx::KnrIndex; parallel_block=get_parallel_block(length(idx.db)), pools=nothing, verbose=true)
    sp = length(idx) + 1
    n = length(idx.db)
    E = [KnnResult(idx.kbuild) for _ in 1:parallel_block]
    while sp < n
        ep = min(n, sp + parallel_block - 1)
        verbose && println(stderr, "$(typeof(idx)) appending chunk ", (sp=sp, ep=ep, n=n), " ", Dates.now())
    
        Threads.@threads for i in sp:ep
            begin
                res = reuse!(E[i - sp + 1], idx.kbuild)
                search(idx.centers, idx[i], res)
            end
        end

        append!(idx.invfile, VectorDatabase(E), ep-sp+1; parallel_block)
        sp = ep + 1
    end
end

"""
    KnrIndex(
        dist::SemiMetric,
        db::AbstractDatabase;
        invfiletype=BinaryInvertedFile,
        invfiledist=JaccardDistance(),
        initial=:dnet,
        maxiters=0,
        refs=references(dist, db; initial),
        centers=nothing,
        kbuild=3,
        ksearch=1,
        centersrecall::AbstractFloat=length(db) > 10^3 ? 0.95 : 1.0,
        ordering=DistanceOrdering(),
        pools=nothing,
        parallel_block=get_parallel_block(length(db)),
        verbose=false
    )

A convenient function to create a `KnrIndex`, it uses several default arguments. After the construction, use [`optimize!`](@ref) to adjust the index to some performance.

# Arguments
- `dist`: Distance object (a `SemiMetric` object, see `Distances.jl`)
- `db`: The database of objects to be indexed. 

# Keyword arguments
- `invfiletype`: the type of the underlying inverted file (`BinaryInvertedFile` or `WeightedInvertedFile`)
- `invfiledist`: the distance of the underlying inverted file (see [`InvertedFiles.jl`](https://github.com/sadit/InvertedFiles.jl) package)
- `centers`: The index used for centers/references, if `centers === nothing` then a sample of `db` will be used.
- `initial`: indicates how references are selected, only used if `refs` will be computed; see [`references`](@ref) for more detail.
- `maxiters`: how many iterations of the Lloyd algorithm are applied to initial references, only used if `refs` will be computed; see [`references`](@ref) for more detail.
- `refs`: the set of reference, only used if `centers === nothing`
- `centersrecall`: used when `centers === nothing`; if `centersrecall == 1` then it will create an exact index on `refs` or an approximate if `0 < centersrecall < 1`
- `kbuild`: the number of references to compute and store on construction
- `ksearch`: the number of references to compute on searching
- `ordering`: the ordering strategy
- `pools`: an object with preallocated caches specific for `KnrIndex`, if `pools=nothing` it will use default caches.
- `parallel_block` Parallel construction works on batches, this is the size of these blocks
- `verbose` true if you want to see messages
"""
function KnrIndex(
        dist::SemiMetric,
        db::AbstractDatabase;
        invfiletype=BinaryInvertedFile,
        invfiledist=JaccardDistance(),
        initial=:dnet,
        maxiters=0,
        refs=references(dist, db; initial, maxiters),
        centers=nothing,
        kbuild::Integer=3,
        ksearch::Integer=1,
        centersrecall::AbstractFloat=length(db) > 10^3 ? 0.95 : 1.0,
        ordering::KnrOrderingStrategy=DistanceOrdering(),
        pools=nothing,
        parallel_block::Integer=get_parallel_block(length(db)),
        verbose=false
    )
    kbuild = convert(Int32, kbuild)
    ksearch = convert(Int32, ksearch)

    if centers === nothing
        if centersrecall == 1.0
            centers = ExhaustiveSearch(; db=refs, dist)
        else
            0 < centersrecall < 1 || throw(ArgumentError("the expected recall for centers index should be 0 < centersrecall < 0"))
            centers = SearchGraph(; db=refs, dist, verbose)
            index!(centers; parallel_block)
            optimize!(centers, OptimizeParameters(kind=MinRecall(centersrecall)))
        end
    end
    
    invfile = invfiletype(length(centers), invfiledist)
    idx = KnrIndex(dist, db, centers, invfile, kbuild, ordering, KnrOpt(ksearch))
    pools = pools === nothing ? getpools(idx) : pools
    index!(idx; parallel_block, pools, verbose)
    idx
end
