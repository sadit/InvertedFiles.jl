# This file is part of InvertedFiles.jl

export BinaryInvertedFile, set_distance_evaluate

"""
    struct BinaryInvertedFile <: AbstractInvertedFile

Creates a binary weighted inverted index. An inverted index is an sparse matrix representation optimized for computing `k` nn elements (columns) under some distance.

# Properties:

- `dist`: Distance function to be applied, valid values are: `IntersectionDissimilarity()`, `DiceDistance()`, `JaccardDistance()`, and `CosineDistanceSet()
- `lists`: posting lists (non-zero values of the rows in the matrix)
- `sizes`: number of non-zero values per object (number of non-zero values per column)
- `locks`: Per row locks for multithreaded construction
"""
struct BinaryInvertedFile{IntListType<:AbstractVector{<:Integer},DistType<:Union{IntersectionDissimilarity,DiceDistance,JaccardDistance,CosineDistanceSet}} <: AbstractInvertedFile
    dist::DistType
    lists::Vector{IntListType}
    sizes::Vector{Int32}
    locks::Vector{SpinLock}
end

"""
    set_distance_evaluate(dist::SemiMetric, intersection::Integer, size1::Integer, size2::Integer)

Computes the distance function `dist` on a [`BinaryInvertedFile`](@ref).
"""
set_distance_evaluate(::IntersectionDissimilarity, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - intersection / max(size1, size2)
set_distance_evaluate(::DiceDistance, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - (2intersection) / (size1 + size2)
set_distance_evaluate(::JaccardDistance, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - (intersection) / (size1 + size2 - intersection)
set_distance_evaluate(::CosineDistanceSet, intersection::Integer, size1::Integer, size2::Integer) = 1.0 - (intersection) / (sqrt(size1) * sqrt(size2))

"""
    BinaryInvertedFile(vocsize::Integer, dist=JaccardDistance(), ::Type{IntType}=Int32)

Creates an `BinaryInvertedFile` with the given vocabulary size and for the given distance function `dist`:

# Arguments:
- `vocsize`: the vocabulary size of the index
- `dist`: the distance function to be used in searches
- IntType: the integer type to store positions in the posting lists.

"""
function BinaryInvertedFile(vocsize::Integer, dist=JaccardDistance(), ::Type{IntType}=Int32) where {IntType<:Integer}
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    BinaryInvertedFile(dist, [IntType[] for i in 1:vocsize], Int32[],  [SpinLock() for i in 1:vocsize])
end

function _internal_push!(idx::BinaryInvertedFile, tokenID, objID, _, sort)
    @inbounds L = idx.lists[tokenID]
    push!(L, objID)
    sort && sortlastpush!(L)
end