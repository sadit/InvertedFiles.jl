# This file is part of InvertedFiles.jl

using Intersections
import SimilaritySearch: search
export search, prepare_posting_lists_for_querying


"""
	prepare_posting_lists_for_querying(idx::WeightedInvertedFile{I,F}, q, Q=nothing, tol=1e-6)

Fetches and prepares the involved posting lists to solve `q`
"""
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

	search(idx, Q) do objID, d
		push!(res, objID, d)
	end

    (res=res, cost=0)
end

"""
	search(callback::Function, idx::WeightedInvertedFile, Q; pools=nothing)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments:
- `callback`: callback function on each candidate
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`prepare_posting_lists_for_querying`](@ref)
"""
function search(callback::Function, idx::WeightedInvertedFile, Q; pools=nothing)
    n = length(Q)

	umerge(Q) do L, P, m
		@inbounds w = 1.0 - L[1].val * L[1].W[P[1]]
		@inbounds objID = L[1].I[P[1]]
		@inbounds @simd for i in 2:m
			w -= L[i].val * L[i].W[P[i]]
		end

		callback(objID, w)
	end
end
