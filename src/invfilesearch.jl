# This file is part of InvertedFiles.jl

using Intersections
import SimilaritySearch: search
export search, prepare_posting_lists_for_querying

"""
	prepare_posting_lists_for_querying(idx::AbstractInvertedFile, q, pools=getpools(idx), tol=1e-6)

Fetches and prepares the involved posting lists to solve `q`
"""
function prepare_posting_lists_for_querying(idx::AbstractInvertedFile, q, pools=getpools(idx), tol=1e-6)
	Q = getcachepostinglists(pools)
	
	@inbounds for (tokenID, val) in sparseiterator(q)
		val < tol && continue
		L = idx.lists[tokenID]
		if length(L) > 0
            push_posting_list!(Q, idx, tokenID, val)
		end
	end
	
	Q
end

"""
	search(idx::AbstractInvertedFile, q, res::KnnResult; pools=nothing)

Searches `q` in `idx` using the cosine dissimilarity, it computes the full operation on `idx`. `res` specify the query
"""
function search(idx::AbstractInvertedFile, q, res::KnnResult; pools=getpools(idx), tol=1e-6, t=1)
	Q = prepare_posting_lists_for_querying(idx, q, pools, tol)
    P = getcachepositions(length(Q), pools)

    cost = search(idx, Q, P, t) do objID, d
		push!(res, objID, d)
	end

    (res=res, cost=cost)
end
