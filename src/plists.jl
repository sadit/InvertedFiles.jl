# This file is part of InvertedFiles.jl

import Intersections: _get_key

"""
    struct PostingList

A paired list of identifiers and weights
"""
struct PostingList{ListType<:AbstractVector}
    list::ListType
    tokenID::UInt32
    weight::Float32  # useful for search time and saving global data
end

@inline Base.length(plist::PostingList) = length(plist.list)

@inline function _get_key(plist::PostingList{VectorType}, i::Integer)::UInt32 where {VectorType<:AbstractVector{UInt32}} 
    @inbounds plist.list[i]
end

@inline function _get_key(plist::PostingList{VectorType}, i::Integer)::UInt32 where {VectorType<:AbstractVector{IdWeight}}
    @inbounds plist.list[i].id
end

@inline function _get_key(plist::PostingList{VectorType}, i::Integer)::UInt32 where {VectorType<:AbstractVector{IdIntWeight}}
    @inbounds plist.list[i].id
end
