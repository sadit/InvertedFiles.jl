# This file is part of InvertedFiles.jl

using Intersections: _sort!, _remove_empty!
import SimilaritySearch: search
export isearch, usearch, search, prepare_posting_lists_for_querying


function prepare_posting_lists_for_querying(idx::InvertedFile{I,F,Nothing}, q) where {I,F}
	@inbounds [PostingList(idx.lists[tokenID], weight) for (tokenID, weight) in q if length(idx.lists[tokenID]) > 0]
end

function prepare_posting_lists_for_querying(idx::InvertedFile{I,F,<:Dict}, q) where {I,F}
	Q = valtype(idx.lists)[]
	for (token, weight) in q
		tokenID = get(idx.map, token, 0)
		if tokenID > 0 && length(idx.lists[tokenID]) > 0
			@inbounds push!(Q, PostingList(idx.lists[tokenID], weight))
		end
	end
	
	Q
end

"""
	isearch(idx::InvertedFile, q::DVEC, res::KnnResult)

Searches `q` in `idx` using the cosine dissimilarity, it computes a partial operation on `idx`. `res` specify the query.
"""
function isearch(idx::InvertedFile, q::DVEC, res::KnnResult)
	Q = prepare_posting_lists_for_querying(idx, q)
	bk(Q) do L, P
		w = 1.0
		@inbounds @simd for i in eachindex(P)
			w -= L[i].W[P[i]] * L[i].weight
		end

		@inbounds push!(res, L[1].I[P[1]], w)
	end

	(res=res, cost=0)
end

"""
	usearch(idx::InvertedFile, q::DVEC, res::KnnResult)

Searches `q` in `idx` using the cosine dissimilarity, it computes the full operation on `idx`. `res` specify the query
"""

function usearch(idx::InvertedFile, q::DVEC, res::KnnResult)
	Q = prepare_posting_lists_for_querying(idx, q)

	umerge(Q) do L, P, m
		w = 1.0 - L[1].weight * L[1].W[P[1]]
		@inbounds @simd for i in 2:m
			w -= L[i].weight * L[i].W[P[i]]
		end

		@inbounds push!(res, L[1].I[P[1]], w)
	end

    (res=res, cost=0)
end

"""
	search(idx::InvertedFile, q::DVEC, res::KnnResult; intersection=false)

Searches `q` in `idx` using the cosine dissimilarity; `res` specify the query.  If `intersection` is true, then
an approximation is computed based on the intersection of posting lists. Note that intersection could be faster for large lists.
"""

function search(idx::InvertedFile, q::DVEC, res::KnnResult; intersection=false)
	intersection ? isearch(idx, q, res) : usearch(idx, q, res)
end