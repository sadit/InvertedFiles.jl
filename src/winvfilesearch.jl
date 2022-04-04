# This file is part of InvertedFiles.jl

using Intersections
import SimilaritySearch: search
export isearch, usearch, search, prepare_posting_lists_for_querying

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


function prepare_posting_lists_for_querying(idx::WeightedInvertedFile{I,F}, q, Q=nothing, tol=1e-6) where {I,F}
	if Q === nothing
		Q = PostingList{I,F}[]
		sizehint!(Q, length(q))
	end
	
	for (tokenID, val) in q
		val < tol && continue
		if length(idx.rowvals[tokenID]) > 0
			p = PostingList(idx.rowvals[tokenID], idx.nonzeros[tokenID], convert(Float64, val))
			@inbounds push!(Q, p)
		end
	end
	
	Q
end

"""
	usearch(idx::WeightedInvertedFile, q, res::KnnResult; pools=nothing)

Searches `q` in `idx` using the cosine dissimilarity, it computes the full operation on `idx`. `res` specify the query
"""
function search(idx::WeightedInvertedFile, q, res::KnnResult; pools=nothing)
	Q = prepare_posting_lists_for_querying(idx, q)

	umerge(Q) do L, P, m
		@inbounds w = 1.0 - L[1].val * L[1].W[P[1]]
		@inbounds objID = L[1].I[P[1]]
		@inbounds @simd for i in 2:m
			w -= L[i].val * L[i].W[P[i]]
		end

		push!(res, objID, w)
	end

    (res=res, cost=0)
end
