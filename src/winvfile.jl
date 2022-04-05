# This file is part of InvertedFiles.jl

export WeightedInvertedFile

struct WeightedInvertedFile{IntListType<:AbstractVector{<:Integer},RealListType<:AbstractVector{<:Real}} <: AbstractInvertedFile
    lists::Vector{IntListType}  ## posting lists
    weights::Vector{RealListType}  ## associated weights
    sizes::Vector{Int32}  ## number of non zero elements per vector
    locks::Vector{SpinLock}
end

function WeightedInvertedFile(vocsize::Integer, ::Type{IntType}=Int32, ::Type{RealType}=Float32) where {IntType<:Integer,RealType<:Real}
    vocsize > 0 || throw(ArgumentError("voc must not be empty"))
    WeightedInvertedFile([IntType[] for i in 1:vocsize], [RealType[] for i in 1:vocsize], Int32[],  [SpinLock() for i in 1:vocsize])
end

function _internal_push!(idx::WeightedInvertedFile, tokenID, objID, weight, sort)
    push!(idx.lists[tokenID], objID)
    push!(idx.weights[tokenID], weight)
    sort && sortlastpush!(idx.lists[tokenID], idx.weights[tokenID])
end
