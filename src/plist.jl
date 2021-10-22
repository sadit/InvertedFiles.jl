# This file is a part of InvertedFiles.jl

using SimilaritySearch
export PostingList

struct PostingList{IType,WType} <: AbstractVector{Tuple{IType,WType}}
    I::Vector{IType}
    W::Vector{WType}
    weight::Float64  # useful for search time and saving global data
end

Base.eltype(::PostingList{I,W}) where {I,W} = Tuple{I,W}
PostingList{IType,WType}() where {IType,WType} = PostingList{IType,WType}(IType[], WType[], 0.0)
PostingList(I::Vector{IType}, W::Vector{WType}, w=0.0) where {IType,WType} = PostingList{IType,WType}(I, W, w)
PostingList(plist::PostingList{IType,WType}, w=0.0) where {IType,WType} = PostingList{IType,WType}(plist.I, plist.W, w)

@inline Base.size(plist::PostingList) = size(plist.I)
@inline Base.length(plist::PostingList) = length(plist.I)
@inline Base.eachindex(plist::PostingList) = eachindex(plist.I)
@inline Base.getindex(plist::PostingList, index) = @inbounds (plist.I[index], plist.W[index])
@inline Intersections._get_key(plist::PostingList, i) = @inbounds plist.I[i]

@inline Base.first(plist::PostingList) = @inbounds plist[1]
@inline Base.last(plist::PostingList) = @inbounds plist[end]

function Base.setindex!(plist::PostingList, pair, index)
    plist.I[index] = first(pair)
    plist.W[index] = last(pair)
end
