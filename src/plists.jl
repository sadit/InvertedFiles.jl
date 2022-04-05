# This file is part of InvertedFiles.jl

struct PostingList{IVecType,WVecType}
    I::IVecType
    W::WVecType
    val::Float64  # useful for search time and saving global data
end

Base.eltype(::PostingList{I,W}) where {I,W} = Tuple{I,W}

@inline Base.size(plist::PostingList) = size(plist.I)
@inline Base.length(plist::PostingList) = length(plist.I)
@inline Base.eachindex(plist::PostingList) = eachindex(plist.I)
@inline Base.getindex(plist::PostingList, index) = @inbounds (plist.I[index], plist.W[index])
@inline Intersections._get_key(plist::PostingList, i) = @inbounds plist.I[i]

@inline Base.first(plist::PostingList) = @inbounds plist[1]
@inline Base.last(plist::PostingList) = @inbounds plist[end]

#=
function Base.setindex!(plist::PostingList, pair, index)
    plist.I[index] = first(pair)
    plist.W[index] = last(pair)
end =#