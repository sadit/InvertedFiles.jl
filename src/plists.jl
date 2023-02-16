# This file is part of InvertedFiles.jl

import Intersections: _get_key

"""
    struct PostingList

A paired list of identifiers and weights
"""
struct PostingList{EndPointType}
    list::Vector{EndPointType}
    tokenID::UInt32
    weight::Float32  # useful for search time and saving global data
end

"""
    PostingList(I)

Creates a posting lists with an empty array of weights
"""
PostingList(I) = PostingList(I, 0, 1f0)

@inline Base.length(plist::PostingList) = length(plist.list)
@inline _get_key(plist::PostingList{T}, i) where {T<:Number} = @inbounds plist.list[i]
@inline _get_key(plist::PostingList{WeightedEndPoint}, i) = @inbounds plist.list[i].id
