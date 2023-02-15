# This file is part of InvertedFiles.jl

const EmptyWeightsPostingList = Float32[]

"""
    struct PostingList

A paired list of identifiers and weights
"""
struct PostingList
    I::Vector{UInt32}
    W::Vector{Float32}
    tokenID::UInt32
    weight::Float32  # useful for search time and saving global data
end

"""
    PostingList(I)

Creates a posting lists with an empty array of weights
"""
PostingList(I) = PostingList(I, EmptyWeightsPostingList, 0, 1f0)

@inline Base.length(plist::PostingList) = length(plist.I)
@inline Intersections._get_key(plist::PostingList, i) = @inbounds plist.I[i]
