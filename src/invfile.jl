# This file is part of InvertedFiles.jl

using SimilaritySearch, LinearAlgebra, SparseArrays

export AbstractInvertedFile

"""
    abstract type AbstractInvertedFile <: AbstractSearchIndex end

Abstract inverted file, actual data structures are [`WeightedInvertedFile`](@ref) and [`BinaryInvertedFile`](@ref)
"""
abstract type AbstractInvertedFile <: AbstractSearchIndex end

"""
    length(idx::AbstractInvertedFile)

Number of indexed elements
"""
Base.length(idx::AbstractInvertedFile) = length(idx.sizes)
# SimilaritySearch.getpools(::AbstractInvertedFile, results=SimilaritySearch.GlobalKnnResult) = results
Base.show(io::IO, idx::AbstractInvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.lists)), n=$(length(idx))}")
SimilaritySearch.database(idx::AbstractInvertedFile) = idx.db

"""
    struct InvertedFilesCaches
        Q
        P
    end
    
Caches used for `BinaryInvertedFile` (one per thread)

# Properties
- `Q`: posting lists involved in a query
- `P`: positions for merge algorithms
"""
struct InvertedFilesCaches
    Q::Vector{PostingList}
    P::Vector{UInt32}
end

function getcachepostinglists(pools::Vector{InvertedFilesCaches})
    Q = pools[Threads.threadid()].Q
    empty!(Q)
    Q
end

function getcachepositions(k::Integer, pools::Vector{InvertedFilesCaches})
    P = pools[Threads.threadid()].P
    resize!(P, k)
    fill!(P, 1)
    P
end

const GlobalInvertedFilesCachesPool = Vector{InvertedFilesCaches}(undef, 0)

function __init__invfile()
    n = Threads.nthreads()

    while length(GlobalInvertedFilesCachesPool) < n
        push!(GlobalInvertedFilesCachesPool, InvertedFilesCaches(Vector{PostingList}(undef, 10), Vector{UInt32}(undef, 10)))
    end
end

getpools(invfile::AbstractInvertedFile) = GlobalInvertedFilesCachesPool

"""
    sparseiterator(db, i)

Creates an iterator for indices and values of the `i`-th db's element (e.g., column).
Several specializations are provided.
"""
function sparseiterator(db::MatrixDatabase{<:SparseMatrixCSC}, i)
    r = nzrange(db.matrix, i)
    v = nonzeros(db.matrix)
    zip(r, view(v, r))
end

sparseiterator(db::MatrixDatabase{<:Matrix}, i) = enumerate(view(db.matrix, i))
sparseiterator(db::AbstractDatabase, i) = sparseiterator(db[i])

"""
    sparseiterator(obj)

`(id, weight)` iterator for `obj` for generic databases.
"""
sparseiterator(obj::KnnResult) = obj
sparseiterator(obj::AbstractVector{<:AbstractFloat}) = enumerate(obj)
sparseiterator(obj::AbstractVector{<:Integer}) = ((u, 1) for u in obj)
sparseiterator(obj::Set) = (convertpair(u) for u in obj)
sparseiterator(obj::SortedIntSet) = (convertpair(u) for u in obj)
sparseiterator(obj) = (convertpair(u) for u in obj)

"""
    convertpair(u)

Converts an element of an `sparseiterator` into an usable pair.
"""
convertpair(u::Integer) = (u, 1)
convertpair(u::Tuple) = u # assert length(u) = 2
convertpair(u::Vector) = u # assert length(u) = 2
convertpair(u::Pair) = u

function parallel_append!(idx, db, startID, n, minbatch, tol)
    resize!(idx.sizes, startID + n)
    minbatch = getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        objID = i + startID
        nz = 0

        @inbounds for (tokenID, weight) in sparseiterator(db, i)
            weight < tol && continue
            tokenID == 0 && continue # tokenID == 0 is allowed as centinel (useful for plain distance evaluation of cosine)
            nz += 1
            lock(idx.locks[tokenID])
            try
                _internal_push!(idx, tokenID, objID, weight, true)
            finally
                unlock(idx.locks[tokenID])
            end
        end

        @inbounds idx.sizes[objID] = nz
    end

    idx
end

function SimilaritySearch.index!(idx::AbstractInvertedFile; minbatch=0, pools=nothing, tol=1e-6)
    startID = length(idx)
    db = database(idx)
    n = length(db) - startID
    n == 0 && return idx
    parallel_append!(idx, db, startID, n, minbatch, tol)
end

"""
    Base.append!(idx, db; minbatch=1000, pools=nothing, tol=1e-6)

Appends all `db` elements into the index `idx`. It work in parallel using all available threads.

# Arguments:
- `idx`: The inverted index
- `db`: The database of sparse objects, it can be only indices if each object is a list of integers or a set of integers (useful for `BinaryInvertedFile`),
    sparse matrices, dense matrices, among other combinations.
- `n`: The number of items to insert (defaults to all)

# Keyword arguments:
- `minbatch`: how many elements are inserted per available thread.
- `pools`: unused argument but necessary by `searchbatch` (from `SimilaritySearch`)
- `tol`: controls what is a zero (i.e., weights < tol will be ignored).
"""
function Base.append!(idx::AbstractInvertedFile, db::AbstractDatabase, n=length(db); minbatch=0, pools=nothing, tol=1e-6)
    startID = length(idx)
    resize!(idx.sizes, startID + n)
    !isnothing(idx.db) && append!(idx.db, db)

    parallel_append!(idx, db, startID, n, minbatch, tol)
end

"""
    push!(idx::AbstractInvertedFile, obj; pools=nothing, tol=1e-6)

Inserts a single element into the index. This operation is not thread-safe.

# Arguments
- `idx`: The inverted index
- `obj`: The object to be indexed

# Keyword arguments
- `pools`: unused argument
- `tol`: controls what is a zero (i.e., `weight < tol` will be ignored)
"""
function Base.push!(idx::AbstractInvertedFile, obj, objID=length(idx) + 1; pools=nothing, tol=1e-6)
    n = length(idx) + 1
    nz = 0

    @inbounds for (tokenID, weight) in sparseiterator(obj)
        weight < tol && continue
        nz += 1
        _internal_push!(idx, tokenID, objID, weight, false)
    end

    push!(idx.sizes, nz)
    !isnothing(idx.db) && push!(idx.db, obj)
    idx
end
