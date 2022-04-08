# This file is part of InvertedFiles.jl

function push_posting_list!(Q, idx::WeightedInvertedFile, tokenID, val)
	@inbounds p = PostingList(idx.lists[tokenID], idx.weights[tokenID], convert(Float64, val))
	push!(Q, p)
end

"""
search(callback::Function, idx::WeightedInvertedFile, Q, P; t=1)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments:
- `callback`: callback function on each candidate
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`prepare_posting_lists_for_querying`](@ref)
- `P`: a vector of starting positions in Q (initial state as ones)
"""
function search(callback::Function, idx::WeightedInvertedFile, Q, P_, t)
	umerge(Q, P_; t) do L, P, m
		@inbounds w = 1.0 - L[1].val * L[1].W[P[1]]
		@inbounds objID = L[1].I[P[1]]
		@inbounds @simd for i in 2:m
			w -= L[i].val * L[i].W[P[i]]
		end

		callback(objID, w)
	end
end
