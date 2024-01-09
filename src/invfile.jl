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
Base.show(io::IO, idx::AbstractInvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.adj)), n=$(length(idx))}")
SimilaritySearch.database(idx::AbstractInvertedFile) = idx.db


function getcontainer(idx::AbstractInvertedFile, ctx::InvertedFileContext)
    Q = getcontainer(idx.adj, ctx)
    empty!(Q)
    Q
end

getcontainer(adj::AdjacencyList{UInt32}, ctx) = ctx.cont_u32[Threads.threadid()]
getcontainer(adj::AdjacencyList{IdWeight}, ctx) = ctx.cont_iw[Threads.threadid()]
getcontainer(adj::AdjacencyList{IdIntWeight}, ctx) = ctx.cont_iiw[Threads.threadid()]

function getcontainer(adj::StaticAdjacencyList, ctx)
    Q = [PostingList(neighbors(adj, 1), zero(UInt32), 0f0)]
    empty!(Q)
    sizehint!(Q, 32)
    Q
end

function getpositions(k::Integer, ctx::InvertedFileContext)
    P = ctx.positions[Threads.threadid()]
    resize!(P, k)
    fill!(P, 1)
    P
end


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

function SimilaritySearch.index!(idx::AbstractInvertedFile, ctx::InvertedFileContext; tol=1e-6)
    startID = length(idx)
    db = database(idx)
    n = length(db) - startID
    n == 0 && return idx
    parallel_append!(idx, ctx, db, startID, n, tol, true)
end

"""
    append_items!(idx, ctx, db; tol=1e-6, sort=true)

Appends all `db` elements into the index `idx`. It work in parallel using all available threads.

# Arguments:
- `idx`: The inverted index
- `db`: The database of sparse objects, it can be only indices if each object is a list of integers or a set of integers (useful for `BinaryInvertedFile`),
    sparse matrices, dense matrices, among other combinations.
- `n`: The number of items to insert (defaults to all)

# Keyword arguments:
- `tol`: controls what is a zero (i.e., weights < tol will be ignored).
- `sort`: if true keep posting lists always sorted, use `sort=false` to skip sorting but note that most methods expect sorted entries, so you must sort them in a posterior step.
"""
function SimilaritySearch.append_items!(idx::AbstractInvertedFile, ctx::InvertedFileContext, db::AbstractDatabase, n=length(db); tol=1e-6, sort=true)
    startID = length(idx)
    !isnothing(idx.db) && append_items!(idx.db, db)

    parallel_append!(idx, ctx, db, startID, n, tol, sort)
    ctx.logger !== nothing && LOG(ctx.logger, append_items!, idx, startID, length(idx), length(idx))
    idx
end

"""
    push_item!(idx::AbstractInvertedFile, ctx::InvertedFileContext, obj; tol=1e-6)

Inserts a single element into the index. This operation is not thread-safe.

# Arguments
- `idx`: The inverted index
- `ctx`: the index's context
- `obj`: The object to be indexed

# Keyword arguments
- `tol`: controls what is a zero (i.e., `weight < tol` will be ignored)
"""
function SimilaritySearch.push_item!(idx::AbstractInvertedFile, ctx::InvertedFileContext, obj, objID=length(idx) + 1; tol=1e-6)
    internal_push_object!(idx, ctx, objID, obj, tol, false, true)
    !isnothing(idx.db) && push_item!(idx.db, obj)
    ctx.logger !== nothing && LOG(ctx.logger, push_item!, idx, objID)
    idx
end

function internal_push_object!(idx::AbstractInvertedFile, ctx::InvertedFileContext, objID::Integer, obj, tol::Float64, sort, is_push)
    nz = 0
    @inbounds for (tokenID, weight) in sparseiterator(obj)
        weight < tol && continue
        tokenID == 0 && continue  # object 0 is a centinel
        nz += 1
        internal_push!(idx, ctx, tokenID, objID, weight, sort)
    end

    if is_push
        push!(idx.sizes, nz)
    else
        idx.sizes[objID] = nz
    end
end


function parallel_append!(idx, ctx::InvertedFileContext, db::AbstractDatabase, startID::Int, n::Int, tol::Float64, sort::Bool)
    internal_parallel_prepare_append!(idx, startID + n)
    minbatch = getminbatch(ctx.minbatch, n)

    @batch minbatch=minbatch per=thread for i in 1:n
        objID = i + startID
        internal_push_object!(idx, ctx, objID, db[i], tol, sort, false)
    end

    idx
end

function internal_parallel_prepare_append!(idx::AbstractInvertedFile, new_size::Integer)
    resize!(idx.sizes, new_size)
end
