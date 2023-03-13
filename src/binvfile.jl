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
struct BinaryInvertedFile{
            DistType<:Union{IntersectionDissimilarity,DiceDistance,JaccardDistance,CosineDistanceSet},
            AdjType<:AbstractAdjacencyList,
            DbType<:Union{<:AbstractDatabase,Nothing}
        } <: AbstractInvertedFile
    dist::DistType
    db::DbType
    adj::AdjType
    sizes::Vector{UInt32}
    locks::Vector{SpinLock}
end

BinaryInvertedFile(invfile::BinaryInvertedFile;
    dist=invfile.dist,
    db=invfile.db,
    adj=invfile.adj,
    sizes=invfile.sizes,
    locks=invfile.locks
) = BinaryInvertedFile(dist, db, adj, sizes, locks)

SimilaritySearch.distance(idx::BinaryInvertedFile) = idx.dist

function SimilaritySearch.saveindex(filename::AbstractString, index::BinaryInvertedFile, meta::Dict)
    I = BinaryInvertedFile(index; adj=SimilaritySearch.StaticAdjacencyList(index.adj))
    jldsave(filename; index=I, meta)
end

function restoreindex(index::BinaryInvertedFile, meta::Dict, f)
    BinaryInvertedFile(index; adj=SimilaritySearch.AdjacencyList(index.adj)), meta
end

"""
    set_distance_evaluate(dist::SemiMetric, intersection::Integer, size1::Integer, size2::Integer)

Computes the distance function `dist` on a [`BinaryInvertedFile`](@ref).
"""
set_distance_evaluate(::IntersectionDissimilarity, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - intersection / max(size1, size2)
set_distance_evaluate(::DiceDistance, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - (2intersection) / (size1 + size2)
set_distance_evaluate(::JaccardDistance, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - (intersection) / (size1 + size2 - intersection)
set_distance_evaluate(::CosineDistanceSet, intersection::Int32, size1::Int32, size2::Int32)::Float32 = 1.0 - (intersection) / (sqrt(size1) * sqrt(size2))
set_distance_evaluate(t, intersection::Integer, size1::Integer, size2::Integer)::Float32 = set_distance_evaluate(t, convert(Int32, intersection), convert(Int32, size1), convert(Int32, size2))
"""
    BinaryInvertedFile(vocsize::Integer, dist=JaccardDistance())

Creates an `BinaryInvertedFile` with the given vocabulary size and for the given distance function `dist`:

# Arguments:
- `vocsize`: the vocabulary size of the index
- `dist`: the distance function to be used in searches
"""
function BinaryInvertedFile(vocsize::Integer, dist=JaccardDistance(), db=nothing)
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    BinaryInvertedFile(dist, db, AdjacencyList(UInt32, n=vocsize), UInt32[],  [SpinLock() for i in 1:vocsize])
end

function internal_push!(idx::BinaryInvertedFile, tokenID, objID, _, sort)
    if sort
        add_edge!(idx.adj, tokenID, objID, IdOrder)
    else
        add_edge!(idx.adj, tokenID, objID, nothing)
    end
end
