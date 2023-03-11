# This file is part of InvertedFiles.jl

using Intersections
import SimilaritySearch: search
export search, select_posting_lists

"""
	select_posting_lists(idx::AbstractInvertedFile, q, tol, pools)

Fetches and prepares the involved posting lists to solve `q`
"""
function select_posting_lists(accept::Function, idx::AbstractInvertedFile, q; pools=getpools(idx))
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
    search_invfile(idx, q, res, t, pools) do plist
        plist.weight >= tol
    end
end

function search_invfile(accept_posting_list::Function, idx::AbstractInvertedFile, q, res::KnnResult, t, pools)
    Q = select_posting_lists(accept_posting_list, idx, q; pools)
    n = length(Q)
    n == 0 && return SearchResult(res, 0)
    search_invfile(idx, Q, res, t, pools)
end
