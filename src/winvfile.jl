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
"""
struct WeightedInvertedFile{DbType<:Union{<:AbstractDatabase,Nothing}, AdjType<:AbstractAdjacencyList} <: AbstractInvertedFile
    db::DbType    
    adj::AdjType
    sizes::Vector{UInt32}  ## number of non zero elements per vector
end

SimilaritySearch.distance(idx::WeightedInvertedFile) = NormalizedCosineDistance()

WeightedInvertedFile(invfile::WeightedInvertedFile;
    db=invfile.db,
    adj=invfile.adj,
    sizes=invfile.sizes,
) = WeightedInvertedFile(db, adj, sizes)

Base.copy(invfile::WeightedInvertedFile; kwargs...) = WeightedInvertedFile(invfile; kwargs...)
"""
    WeightedInvertedFile(vocsize::Integer)

Convenient function to create an empty `WeightedInvertedFile` with the given vocabulary size.
"""
function WeightedInvertedFile(vocsize::Integer, db=nothing)
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    WeightedInvertedFile(
        db,
        AdjacencyList(IdWeight; n=vocsize),
        Vector{UInt32}(undef, 0)
    )
end

function internal_push!(idx::WeightedInvertedFile, tokenID, objID, weight, sort)
    if sort
        add_edge!(idx.adj, tokenID, IdWeight(objID, weight), IdOrder)
    else
        add_edge!(idx.adj, tokenID, IdWeight(objID, weight), nothing)
    end

end
