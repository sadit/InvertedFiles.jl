# This file is part of InvertedFiles.jl

using SimilaritySearch, LinearAlgebra
export WeightedInvertedFile

using Base.Threads: SpinLock, @threads

mutable struct WeightedInvertedFile{IntListType<:AbstractVector{<:Integer},RealListType<:AbstractVector{<:Real}} <: AbstractSearchContext
    rowvals::Vector{IntListType}  ## identifiers
    nonzeros::Vector{RealListType}  ## weights
    nnzcount::Vector{Int32}  ## nonzeros elements per vector
    locks::Vector{SpinLock}
end

Base.length(idx::WeightedInvertedFile) = length(idx.nnzcount)
SimilaritySearch.getpools(::WeightedInvertedFile, results=SimilaritySearch.GlobalKnnResult) = results

function WeightedInvertedFile(vocsize::Integer, ::Type{IntType}=Int32, ::Type{RealType}=Float32) where {IntType<:Integer,RealType<:Real}
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    WeightedInvertedFile([IntType[] for i in 1:vocsize], [RealType[] for i in 1:vocsize], Int32[],  [SpinLock() for i in 1:vocsize])
end

Base.show(io::IO, idx::WeightedInvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.rowvals)), n=$(length(idx))}")

"""
    Base.append!(idx::WeightedInvertedFile, db::AbstractDatabase; parallel=false)

Appends all `db` elements into the index
"""
function Base.append!(idx::WeightedInvertedFile, db::AbstractDatabase; parallel_block=1000, pools=nothing, tol=1e-6)
    parallel_block = min(parallel_block, length(db))
    startID = length(idx)
    n = length(db)
    sp = 1
    resize!(idx.nnzcount, length(idx) + n)
    
    while sp < n
        ep = min(n, sp + parallel_block)
        @threads for i in sp:ep
            objID = i + startID
            obj = db[i]
            @inbounds idx.nnzcount[objID] = length(obj)
            @inbounds for (tokenID, weight) in obj
                weight < tol && continue
                try
                    lock(idx.locks[tokenID])
                    push!(idx.rowvals[tokenID], objID)
                    push!(idx.nonzeros[tokenID], weight)
                    sortlastpush!(idx.rowvals[tokenID], idx.nonzeros[tokenID])
                finally
                    unlock(idx.locks[tokenID])
                end
            end
        end

        sp = ep + 1
    end

    idx
end

function Base.push!(idx::WeightedInvertedFile, obj; pools=nothing, tol=1e-6)
    push!(idx.nnzcount, length(obj))
    n = length(idx)

    @inbounds for (tokenID, weight) in obj
        weight < tol && continue
        push!(idx.rowvals[tokenID], n)
        push!(idx.nonzeros[tokenID], weight)
    end

    idx
end
