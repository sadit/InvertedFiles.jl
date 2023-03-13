# This file is part of InvertedFiles.jl

"""
  search_invfile(accept_posting_list::Function, idx::WeightedInvertedFile, q, res::KnnResult, t, pools)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments:
- `accept_posting_list`: predicate to accept or reject a posting list
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`select_posting_lists`](@ref)
"""
function search_invfile(idx::WeightedInvertedFile, Q::Vector{PostType}, res::KnnResult, t, pools) where {PostType<:PostingList}
    P_ = getcachepositions(length(Q), pools)
    cost = xmergefun(Q, P_; t) do L, P, m
        @inbounds w = 1.0 - L[1].weight * L[1].list[P[1]].weight
        @inbounds objID = L[1].list[P[1]].id
        @inbounds @simd for i in 2:m
            w -= L[i].weight * L[i].list[P[i]].weight
        end

        push_item!(res, IdWeight(objID, w))
    end

    SearchResult(res, cost)
end
