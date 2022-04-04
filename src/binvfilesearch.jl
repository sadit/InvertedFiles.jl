# This file is part of InvertedFiles.jl

import SimilaritySearch: search
export search, prepare_posting_lists_for_querying

function prepare_posting_lists_for_querying(idx::BinaryInvertedFile, q, Q=nothing)
	if Q === nothing
		Q = valtype(idx.rowvals)[]
	end
	
	for tokenID in q
		if length(idx.rowvals[tokenID]) > 0
			@inbounds push!(Q, idx.rowvals[tokenID])
		end
	end
	
	Q
end

"""
	search(idx::BinaryInvertedFile, q, res::KnnResult)

Searches the set `q` in `idx`.
"""
function search(idx::BinaryInvertedFile, q, res::KnnResult; pools=nothing)
	Q = prepare_posting_lists_for_querying(idx, q)
    n = length(q)

	umerge(Q) do L, P, isize
        @inbounds objID = L[1][P[1]]
		m = idx.nonzeros[objID]
        d = set_distance_evaluate(idx.dist, isize, n, m)
		@inbounds push!(res, objID, d)
	end

    (res=res, cost=0)
end
