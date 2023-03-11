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

@inline Base.length(plist::PostingList) = length(plist.list)
@inline _get_key(plist::PostingList{UInt32}, i::Integer)::UInt32 = @inbounds plist.list[i]
@inline _get_key(plist::PostingList{IdWeight}, i::Integer)::UInt32 = @inbounds plist.list[i].id
@inline _get_key(plist::PostingList{IdIntWeight}, i::Integer)::UInt32 = @inbounds plist.list[i].id
