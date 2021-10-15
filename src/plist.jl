# This file is a part of InvertedIndex.jl

using SimilaritySearch
export PostingList

struct PostingList{IType,WType}
    I::Vector{IType}
    W::Vector{WType}
end

Base.length(plist::PostingList) = length(plist.I)
Base.eachindex(plist::PostingList) = eachindex(plist.I)
Base.getindex(plist::PostingList, index) = (plist.I[index], plist.W[index])

function Base.setindex!(plist::PostingList, pair, index)
    plist.I[index] = first(pair)
    plist.W[index] = last(pair)
end

PostingList() = PostingList{Int32,Float32}(Int32[], Float32[])

function topk(plist::PostingList{I,F}) where {I,F}
    T = KnnResult{I,F}(top)
    for i in eachindex(plist)
        push!(T, plist.I[i], -plist.W[i])
    end

    for i in eachindex(T.dist)
        T.dist[i] = -T.dist[i]
    end

    PostingList(T.id, T.dist)
    sort!(T)
end