# This file is part of InvertedFiles.jl

export WeightedInvertedFile

"""
    struct WeightedInvertedFile <: AbstractInvertedFile

An inverted index is a sparse matrix representation of with floating point weights, it supports only positive non-zero values.
This index is optimized to efficiently solve `k` nearest neighbors (cosine distance, using previously normalized vectors).

# Parameters

- `lists`: posting lists (non-zero id-elements in rows)
- `weights`: non-zero weights (in rows)
- `sizes`: number of non-zero values in each element (non-zero values in columns)
- `locks`: per-row locks for multithreaded construction
"""
struct WeightedInvertedFile{DbType<:Union{<:AbstractDatabase,Nothing}} <: AbstractInvertedFile
    db::DbType
    lists::Vector{Vector{UInt32}}  ## posting lists
    weights::Vector{Vector{Float32}}  ## associated weights
    sizes::Vector{Int32}  ## number of non zero elements per vector
    locks::Vector{SpinLock}
end

SimilaritySearch.distance(idx::WeightedInvertedFile) = NormalizedCosineDistance()

function SimilaritySearch.saveindex(filename::AbstractString, index::InvFileType, meta::Dict) where {InvFileType<:AbstractInvertedFile}
    lists = SimilaritySearch.flat_adjlist(UInt32, index.lists)
    weights = SimilaritySearch.flat_adjlist(UInt32, index.weights)
    index = InvFileType(index; lists=Vector{UInt32}[], weights=Vector{Float32}[])
    jldsave(filename; index, meta, lists, weights)
end

function restoreindex(index::InvFileType, meta::Dict, f) where {InvFileType<:AbstractInvertedFile}
    lists = unflat_adjlist(UInt32, f["lists"])
    weights = unflat_adjlist(UInt32, f["weights"])
    copy(index; lists, weights), meta
end

WeightedInvertedFile(invfile::WeightedInvertedFile;
    db=invfile.db,
    lists=invfile.lists,
    weights=invfile.weights,
    sizes=invfile.sizes,
    locks=invfile.locks
) = WeightedInvertedFile(db, lists, weights, sizes, locks)

"""
    WeightedInvertedFile(vocsize::Integer)

Convenient function to create an empty `WeightedInvertedFile` with the given vocabulary size.
"""
function WeightedInvertedFile(vocsize::Integer, db=nothing)
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    WeightedInvertedFile(
        db,
        [UInt32[] for i in 1:vocsize],
        [Float32[] for i in 1:vocsize],
        Vector{Int32}(undef, 0), 
        [SpinLock() for i in 1:vocsize]
    )
end

function internal_push!(idx::WeightedInvertedFile, tokenID, objID, weight, sort)
    push!(idx.lists[tokenID], objID)
    push!(idx.weights[tokenID], weight)
    sort && sortlastpush!(idx.lists[tokenID], idx.weights[tokenID])
end
