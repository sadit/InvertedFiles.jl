# This file is part of InvertedFiles.jl

"""
  search_invfile(accept_posting_list::Function, idx::BinaryInvertedFile, Q, res::KnnResult, t, pools)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments

- `accept_posting_list`: predicate to accept or reject a posting list
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`select_posting_lists`](@ref)
- `t`: threshold (t=1 union, t > 1 solves the t-threshold problem)
"""

function search_invfile(idx::BinaryInvertedFile, Q::Vector{PostType}, res::KnnResult, t, pools) where {PostType<:PostingList}
    n = length(Q)
    P_ = getcachepositions(n, pools)
	
    cost = xmergefun(Q, P_; t) do L, P, isize
        @inbounds objID = L[1].list[P[1]]
        @inbounds d = set_distance_evaluate(idx.dist, isize, n, idx.sizes[objID])
        push_item!(res, IdWeight(objID, d))
    end

    SearchResult(res, cost)
end
