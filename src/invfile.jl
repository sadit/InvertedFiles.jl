# This file is part of InvertedFiles.jl

using SimilaritySearch, SimilaritySearch.AdjacencyLists, LinearAlgebra, SparseArrays
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
Base.show(io::IO, idx::AbstractInvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.adj)), n=$(length(idx))}")
SimilaritySearch.database(idx::AbstractInvertedFile) = idx.db


function getcachepostinglists(idx::AbstractInvertedFile)
    getcachepostinglists(idx.adj)
end

function getcachepostinglists(adj::AdjacencyList{UInt32})
    Q = CACHE_CONTAINERS_U32[Threads.threadid()]
    empty!(Q)
    Q
end

function getcachepostinglists(adj::AdjacencyList{IdWeight})
    Q = CACHE_CONTAINERS_IW[Threads.threadid()]
    empty!(Q)
    Q
end

function getcachepostinglists(adj::AdjacencyList{IdIntWeight})
    Q = CACHE_CONTAINERS_IIW[Threads.threadid()]
    empty!(Q)
    Q
end

function getcachepostinglists(adj::StaticAdjacencyList)
    Q = [PostingList(neighbors(adj, 1), zero(UInt32), 0f0)]
    empty!(Q)
    sizehint!(Q, 32)
    Q
end

function getcachepositions(k::Integer)
    P = CACHE_LIST_POSITIONS[Threads.threadid()]
    resize!(P, k)
    fill!(P, 1)
    P
end


const CACHE_LIST_POSITIONS = [Vector{UInt32}(undef, 32)]
const CACHE_CONTAINERS_U32 = [Vector{PostingList{Vector{UInt32}}}(undef, 32)]
const CACHE_CONTAINERS_IW = [Vector{PostingList{Vector{IdWeight}}}(undef, 32)]
const CACHE_CONTAINERS_IIW = [Vector{PostingList{Vector{IdIntWeight}}}(undef, 32)]

function __init__invfile()
    n = Threads.nthreads()

    while length(CACHE_LIST_POSITIONS) < n
        push!(CACHE_LIST_POSITIONS, deepcopy(CACHE_LIST_POSITIONS[1]))
        push!(CACHE_CONTAINERS_U32, deepcopy(CACHE_CONTAINERS_U32[1]))
        push!(CACHE_CONTAINERS_IW, deepcopy(CACHE_CONTAINERS_IW[1]))
        push!(CACHE_CONTAINERS_IIW, deepcopy(CACHE_CONTAINERS_IIW[1]))
    end
end

getpools(invfile::AbstractInvertedFile) = nothing

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
sparseiterator(obj::AbstractVector{<:AbstractFloat}) = enumerate(obj)
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
convertpair(u::IdWeight) = (u.id, u.weight)
convertpair(u::IdIntWeight) = (u.id, u.weight)

function SimilaritySearch.index!(idx::AbstractInvertedFile; minbatch=0, pools=nothing, tol=1e-6)
    startID = length(idx)
    db = database(idx)
    n = length(db) - startID
    n == 0 && return idx
    parallel_append!(idx, db, startID, n, minbatch, tol, true)
end

"""
    append_items!(idx, db; minbatch=1000, pools=nothing, tol=1e-6, sort=true)

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
- `sort`: if true keep posting lists always sorted, use `sort=false` to skip sorting but note that most methods expect sorted entries, so you must sort them in a posterior step.
"""
function SimilaritySearch.append_items!(idx::AbstractInvertedFile, db::AbstractDatabase, n=length(db); minbatch=0, pools=nothing, tol=1e-6, sort=true)
    startID = length(idx)
    !isnothing(idx.db) && append_items!(idx.db, db)

    parallel_append!(idx, db, startID, n, minbatch, tol, sort)
end

"""
    push_item!(idx::AbstractInvertedFile, obj; pools=nothing, tol=1e-6)

Inserts a single element into the index. This operation is not thread-safe.

# Arguments
- `idx`: The inverted index
- `obj`: The object to be indexed

# Keyword arguments
- `pools`: unused argument
- `tol`: controls what is a zero (i.e., `weight < tol` will be ignored)
"""
function SimilaritySearch.push_item!(idx::AbstractInvertedFile, obj, objID=length(idx) + 1; pools=nothing, tol=1e-6)
    internal_push_object!(idx, objID, obj, tol, false, true)
    !isnothing(idx.db) && push_item!(idx.db, obj)
    idx
end

function internal_push_object!(idx::AbstractInvertedFile, objID::Integer, obj, tol::Float64, sort, is_push)
    nz = 0
    @inbounds for (tokenID, weight) in sparseiterator(obj)
        weight < tol && continue
        tokenID == 0 && continue  # object 0 is a centinel
        nz += 1
        internal_push!(idx, tokenID, objID, weight, sort)
    end

    if is_push
        push!(idx.sizes, nz)
    else
        idx.sizes[objID] = nz
    end
end


function parallel_append!(idx, db::AbstractDatabase, startID::Int, n::Int, minbatch::Int, tol::Float64, sort::Bool)
    internal_parallel_prepare_append!(idx, startID + n)
    minbatch = getminbatch(minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        objID = i + startID
        internal_push_object!(idx, objID, db[i], tol, sort, false)
    end

    idx
end

function internal_parallel_prepare_append!(idx::AbstractInvertedFile, new_size::Integer)
    resize!(idx.sizes, new_size)
end
