# This file is part of InvertedFiles.jl

"""
	search_invfile(callback::Function, idx::WeightedInvertedFile, Q, P_, t)

Find candidates for solving query `Q` using `idx`. It calls `callback` on each candidate `(objID, dist)`

# Arguments:
- `callback`: callback function on each candidate
- `idx`: inverted index
- `Q`: the set of involved posting lists, see [`prepare_posting_lists_for_querying`](@ref)
- `P`: a vector of starting positions in Q (initial state as ones)
"""
function search_invfile(callback::Function, idx::WeightedInvertedFile, Q, P_, t)
	umerge(Q, P_; t) do L, P, m
		@inbounds w = 1.0 - L[1].weight * L[1].list[P[1]].weight
		@inbounds objID = L[1].list[P[1]].id
		@inbounds @simd for i in 2:m
			w -= L[i].weight * L[i].list[P[i]].weight
		end

		callback(objID, w)
	end
end
