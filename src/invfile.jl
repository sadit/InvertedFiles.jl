# This file is part of InvertedFiles.jl

using SimilaritySearch, LinearAlgebra, SparseArrays
using Base.Threads: SpinLock, @threads

abstract type AbstractInvertedFile <: AbstractSearchContext end


Base.length(idx::AbstractInvertedFile) = length(idx.sizes)
SimilaritySearch.getpools(::AbstractInvertedFile, results=SimilaritySearch.GlobalKnnResult) = results
Base.show(io::IO, idx::AbstractInvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.lists)), n=$(length(idx))}")

function sparseiterator(db::MatrixDatabase{<:SparseMatrixCSC}, i)
    r = nzrange(db.matrix, i)
    v = nonzeros(db.matrix)
    zip(r, view(v, r))
end

sparseiterator(db::MatrixDatabase{<:Matrix}, i) = enumerate(view(db.matrix, i))
sparseiterator(db::AbstractDatabase, i) = sparseiterator(db[i])
sparseiterator(obj::DVEC) = obj
sparseiterator(obj::AbstractVector{<:AbstractFloat}) = enumerate(obj)
sparseiterator(obj::AbstractVector{<:Integer}) = ((u, 1) for u in obj)
sparseiterator(obj::Set) = (convertpair(u) for u in obj)
sparseiterator(obj::SortedIntSet) = (convertpair(u) for u in obj)
sparseiterator(obj) = (convertpair(u) for u in obj) 
convertpair(u::Integer) = (u, 1)
convertpair(u::Tuple) = u # assert length(u) = 2
convertpair(u::Vector) = u # assert length(u) = 2
convertpair(u::Pair) = u

"""
    Base.append!(idx::AbstractInvertedFile, db; parallel=false)

Appends all `db` elements into the index
"""
function Base.append!(idx::AbstractInvertedFile, db::AbstractDatabase; parallel_block=1000, pools=nothing, tol=1e-6)
    parallel_block = min(parallel_block, length(db))
    startID = length(idx)
    n = length(db)
    sp = 1
    resize!(idx.sizes, length(idx) + n)
    
    while sp < n
        ep = min(n, sp + parallel_block)
        @threads for i in sp:ep
            objID = i + startID
            nz = 0

            @inbounds for (tokenID, weight) in sparseiterator(db, i)
                weight < tol && continue
                nz += 1
                try
                    lock(idx.locks[tokenID])
                    _internal_push!(idx, tokenID, objID, weight, true)
                finally
                    unlock(idx.locks[tokenID])
                end
            end

            @inbounds idx.sizes[objID] = nz
        end

        sp = ep + 1
    end

    idx
end

function Base.push!(idx::AbstractInvertedFile, obj; pools=nothing, tol=1e-6)
    n = length(idx) + 1
    nz = 0

    @inbounds for (tokenID, weight) in sparseiterator(obj)
        weight < tol && continue
        nz += 1
        _internal_push!(idx, tokenID, n, weight, false)
    end

    push!(idx.sizes, nz)
    idx
end
