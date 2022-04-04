# This file is part of InvertedFiles.jl

using SimilaritySearch, LinearAlgebra
export InvertedFile

#mutable struct InvertedFile{U,PostingType<:PostingList} <: AbstractSearchContext
#    lists::Dict{U,PostingType}
#    n::Int
#end

mutable struct InvertedFile{I,F,MapType} <: AbstractSearchContext
    lists::Vector{PostingList{I,F}}
    n::Int
    map::MapType
end

function InvertedFile(vocsize::Integer, I=Int32, F=Float32)
    InvertedFile([PostingList{I,F}() for i in 1:vocsize], 0, nothing)
end

function InvertedFile(U=UInt64, I=Int32, F=Float32)
    InvertedFile(Vector{PostingList{I,F}}(undef, 0), 0, Dict{U,I}())
end

Base.show(io::IO, idx::InvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.lists)), n=$(idx.n)}")

"""
    Base.append!(idx::InvertedFile, db; parallel=false)

Appends all vectors in db to the index
"""
function Base.append!(idx::InvertedFile, db)
    for i in eachindex(db)
        push!(idx, idx.n+1 => db[i])
    end

    idx
end

"""
    push!(index::InvertedFile, p::Pair)

Inserts a weighted vector into the index.

"""
function Base.push!(idx::InvertedFile{I,F,<:Dict}, p::Pair) where {I,F}
    idx.n += 1
    id_, vec_ = p
    vec_ = vec_ isa Pair ? zip(vec_[1], vec_[2]) : vec_
    invmap_push!(idx, id_, vec_)
end

function invmap_push!(idx::InvertedFile{I,F,<:Dict}, id_, vec_) where {I,F}
    @inbounds for (token, weight) in vec_
        m = length(idx.lists)
        tokenID = get!(idx.map, token, m + 1)
        if tokenID > m
            push!(idx.lists, PostingList{I,F}())
        end
        P = idx.lists[tokenID]
        push!(P.I, id_)
        push!(P.W, weight)
    end

    idx
end

function Base.push!(idx::InvertedFile{I,F,Nothing}, p::Pair) where {I,F}
    idx.n += 1
    id_, vec_ = p
    inv_push!(idx, id_, vec_)
end

function inv_push!(idx::InvertedFile, id_, vec_)
    @inbounds for (tokenID, weight) in vec_
        P = idx.lists[tokenID]
        push!(P.I, id_)
        push!(P.W, weight)
    end

    idx
end