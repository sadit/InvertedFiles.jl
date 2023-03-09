# This file is part of InvertedFiles.jl

using Intersections
import SimilaritySearch: search
export search, prepare_posting_lists_for_querying

"""
	prepare_posting_lists_for_querying(idx::AbstractInvertedFile, q, tol, pools)

Fetches and prepares the involved posting lists to solve `q`
"""
function prepare_posting_lists_for_querying(accept::Function, idx::AbstractInvertedFile, q, pools)
	Q = getcachepostinglists(pools)
	
	@inbounds for (tokenID, weight) in sparseiterator(q)
    accept((; idx, q, tokenID, weight)) || continue
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
	search(idx::AbstractInvertedFile, q, res::KnnResult; pools=nothing, tol=1e-6, t=1)

Searches `q` in `idx` using the cosine dissimilarity, it computes the full operation on `idx`. `res` specify the query
"""
function search(idx::AbstractInvertedFile, q, res::KnnResult; pools=getpools(idx), tol=1e-6, t=1)
  tol::Float32 = convert(Float32, tol)
	search_invfile(idx, q, res::KnnResult, pools, t) do plist
		plist.weight >= tol
	end
end

function search_invfile(accept_posting_list::Function, idx::AbstractInvertedFile, q, res::KnnResult, pools, t)
	Q = prepare_posting_lists_for_querying(accept_posting_list, idx, q, pools)
	length(Q) == 0 && return SearchResult(res, 0)
  P = getcachepositions(length(Q), pools)
  cost = search_invfile(idx, Q, P, t) do objID, d
		push_item!(res, IdWeight(objID, d))
	end

	SearchResult(res, cost)
end

