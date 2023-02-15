# This file is part of InvertedFiles.jl

function push_posting_list!(Q, idx::BinaryInvertedFile, tokenID, val)
	@inbounds push!(Q, PostingList(idx.lists[tokenID]))
end

"""
	search(callback::Function, idx::BinaryInvertedFile, Q, P, t)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments

- `callback`: callback function on each candidate
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`prepare_posting_lists_for_querying`](@ref)
- `P`: a vector of starting positions in Q (initial state as ones)
- `t`: threshold (t=1 union, t > 1 solves the t-threshold problem)
"""
function search(callback::Function, idx::BinaryInvertedFile, Q::Vector{LType}, P_::Vector{UInt32}, t) where {LType<:PostingList}
	search_(callback, idx, idx.dist, Q, P_, t)
end

function search_(callback::Function, idx::BinaryInvertedFile, dist, Q::Vector{LType}, P_::Vector{UInt32}, t) where {LType<:PostingList}
    n = length(Q)

	umerge(Q, P_; t) do L, P, isize
        @inbounds objID = L[1].I[P[1]]
        @inbounds d = set_distance_evaluate(dist, isize, n, idx.sizes[objID])
		callback(objID, d)
	end
end
