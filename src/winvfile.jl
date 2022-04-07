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
struct WeightedInvertedFile <: AbstractInvertedFile
    lists::Vector{Vector{UInt32}}  ## posting lists
    weights::Vector{Vector{Float32}}  ## associated weights
    sizes::Vector{Int32}  ## number of non zero elements per vector
    locks::Vector{SpinLock}
end

"""
    WeightedInvertedFile(vocsize::Integer)

Convenient function to create an empty `WeightedInvertedFile` with the given vocabulary size.
"""
function WeightedInvertedFile(vocsize::Integer)
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    WeightedInvertedFile(
        [Vector{Vector{UInt32}}(undef, 0) for i in 1:vocsize],
        [Vector{Float32}(undef, 0) for i in 1:vocsize],
        Vector{UInt32}(undef, 0), 
        [SpinLock() for i in 1:vocsize]
    )
end

function _internal_push!(idx::WeightedInvertedFile, tokenID, objID, weight, sort)
    push!(idx.lists[tokenID], objID)
    push!(idx.weights[tokenID], weight)
    sort && sortlastpush!(idx.lists[tokenID], idx.weights[tokenID])
end
