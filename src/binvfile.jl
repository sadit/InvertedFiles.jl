# This file is part of InvertedFiles.jl

using SimilaritySearch, LinearAlgebra
export BinaryInvertedFile
using Base.Threads: SpinLock, @threads

mutable struct BinaryInvertedFile{IntListType<:AbstractVector{<:Integer},DistType<:Union{IntersectionDissimilarity,DiceDistance,JaccardDistance,CosineDistanceSet}} <: AbstractSearchContext
    dist::DistType
    rowvals::Vector{IntListType}
    nonzeros::Vector{Int32}
    locks::Vector{SpinLock}
end

Base.length(idx::BinaryInvertedFile) = length(idx.nonzeros)

set_distance_evaluate(::IntersectionDissimilarity, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - intersection / max(size1, size2)
set_distance_evaluate(::DiceDistance, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - (2intersection) / (size1 + size2)
set_distance_evaluate(::JaccardDistance, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - (intersection) / (size1 + size2 - intersection)
set_distance_evaluate(::JaccardDistance, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - (intersection) / (size1 + size2 - intersection)
set_distance_evaluate(::CosineDistanceSet, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - (intersection) / (sqrt(size1) * sqrt(size2))

SimilaritySearch.getpools(::BinaryInvertedFile, results=SimilaritySearch.GlobalKnnResult) = results

function BinaryInvertedFile(vocsize::Integer, dist=JaccardDistance(), ::Type{IntType}=Int32) where {IntType<:Integer}
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    BinaryInvertedFile(dist, [IntType[] for i in 1:vocsize], Int32[],  [SpinLock() for i in 1:vocsize])
end

Base.show(io::IO, idx::BinaryInvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.rowvals)), n=$(length(idx))}")

"""
    Base.append!(idx::BinaryInvertedFile, db::AbstractDatabase; parallel=false)

Appends all `db` elements into the index
"""
function Base.append!(idx::BinaryInvertedFile, db::AbstractDatabase; parallel_block=1000, pools=nothing)
    parallel_block = min(parallel_block, length(db))
    startID = length(idx)
    n = length(db)
    sp = 1
    resize!(idx.nonzeros, length(idx) + n)
    
    while sp < n
        ep = min(n, sp + parallel_block)
        @threads for i in sp:ep
            objID = i + startID
            obj = db[i]
            @inbounds idx.nonzeros[objID] = length(obj)
            @inbounds for tokenID in obj
                try
                    lock(idx.locks[tokenID])
                    push!(idx.rowvals[tokenID], objID)
                    sortlastpush!(idx.rowvals[tokenID])
                finally
                    unlock(idx.locks[tokenID])
                end
            end
        end

        sp = ep + 1       
    end

    idx
end

function Base.push!(idx::BinaryInvertedFile, obj; pools=nothing)
    push!(idx.nonzeros, length(obj))
    n = length(idx)

    @inbounds for tokenID in obj
        push!(idx.rowvals[tokenID], n)
    end

    idx
end