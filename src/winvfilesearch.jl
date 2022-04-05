# This file is part of InvertedFiles.jl

using Intersections
import SimilaritySearch: search
export isearch, usearch, search, prepare_posting_lists_for_querying

function prepare_posting_lists_for_querying(idx::WeightedInvertedFile{I,F}, q, Q=nothing, tol=1e-6) where {I,F}
	if Q === nothing
		Q = PostingList{I,F}[]
		sizehint!(Q, length(q))
	end
	
	@inbounds for (tokenID, val) in sparseiterator(q)
		val < tol && continue
		L = idx.lists[tokenID]
		if length(L) > 0
			p = PostingList(L, idx.weights[tokenID], convert(Float64, val))
			@inbounds push!(Q, p)
		end
	end
	
	Q
end

"""
	search(idx::WeightedInvertedFile, q, res::KnnResult; pools=nothing)

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
