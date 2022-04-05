# This file is part of InvertedFiles.jl

import SimilaritySearch: search
export search, prepare_posting_lists_for_querying

"""
	prepare_posting_lists_for_querying(idx::BinaryInvertedFile, q, Q=nothing, tol=1e-6)

Fetches and prepares the involved posting lists to solve `q`
"""
function prepare_posting_lists_for_querying(idx::BinaryInvertedFile, q, Q=nothing, tol=1e-6)
	if Q === nothing
		Q = valtype(idx.lists)[]
	end
	
	@inbounds for (tokenID, weight) in sparseiterator(q)
		weight < tol && continue
		L = idx.lists[tokenID]
		if length(L) > 0
			@inbounds push!(Q, L)
		end
	end
	
	Q
end

"""
	search(idx::BinaryInvertedFile, q, res::KnnResult)

Searches the set `q` in `idx` using the query specification of `res` (also put the result on `res`)
"""
function search(idx::BinaryInvertedFile, q, res::KnnResult; pools=nothing)
	Q = prepare_posting_lists_for_querying(idx, q)
	search(idx, Q) do objID, d
		push!(res, objID, d)
	end

    (res=res, cost=0)
end

"""
	search(callback::Function, idx::BinaryInvertedFile, Q; pools=nothing)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments:
- `callback`: callback function on each candidate
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`prepare_posting_lists_for_querying`](@ref)
"""
function search(callback::Function, idx::BinaryInvertedFile, Q; pools=nothing)
    n = length(Q)

	umerge(Q) do L, P, isize
        @inbounds objID = L[1][P[1]]
		@inbounds m = idx.sizes[objID]
        d = set_distance_evaluate(idx.dist, isize, n, m)
		callback(objID, d)
	end
end
