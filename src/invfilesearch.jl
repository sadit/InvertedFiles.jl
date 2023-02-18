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
	
	@inbounds for (tokenID, weight) in sparseiterator(q)
		weight < tol && continue
		tokenID == 0 && continue
		N = neighbors(idx.adj, tokenID)
		if length(N) > 0
			L = PostingList(N, convert(UInt32, tokenID), convert(Float32, weight))
			push!(Q, L)
		end
	end
	
	Q
end

"""
	search(idx::AbstractInvertedFile, q, res::KnnResult; pools=nothing)

Searches `q` in `idx` using the cosine dissimilarity, it computes the full operation on `idx`. `res` specify the query
"""
function search(idx::AbstractInvertedFile, q, res::KnnResult; pools=getpools(idx), tol=1e-6, t=1)
	search_invfile(idx, q, res::KnnResult, pools, tol, t)
end

function search_invfile(idx::AbstractInvertedFile, q, res::KnnResult, pools, tol, t)
	Q = prepare_posting_lists_for_querying(idx, q, pools, tol)
	length(Q) == 0 && return SearchResult(res, 0)
    P = getcachepositions(length(Q), pools)
    cost = search_invfile(idx, Q, P, t) do objID, d
		push_item!(res, objID, d)
	end

	SearchResult(res, cost)
end
