# This file is part of InvertedFiles.jl

function push_posting_list!(Q, idx::BinaryInvertedFile, tokenID, val)
	@inbounds push!(Q, PostingList(idx.lists[tokenID]))
end

"""
	search(callback::Function, idx::BinaryInvertedFile, Q)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments

- `callback`: callback function on each candidate
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`prepare_posting_lists_for_querying`](@ref)
- `P`: a vector of starting positions in Q (initial state as ones)

# Keyword arguments
- `t`: threshold (t=1 union, t > 1 solves the t-threshold problem)
"""
function search(callback::Function, idx::BinaryInvertedFile, Q::Vector{PostingList}, P_::Vector{UInt32}; t=1)
    search(callback, idx.dist, idx, Q, P_, t)
end

@inline function search(callback::Function, dist, idx::BinaryInvertedFile, Q::Vector{PostingList}, P_::Vector{UInt32}, t)
    n = length(Q)
	umerge(Q, P_; t) do L, P, isize
        @inbounds objID = L[1].I[P[1]]
        @inbounds d = set_distance_evaluate(dist, isize, n, idx.sizes[objID])
		callback(objID, d)
	end
end
