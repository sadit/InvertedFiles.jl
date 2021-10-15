# This file is part of InvertedFiles.jl

using Intersections
import SimilaritySearch: search

function icos(W, Q, I, res::KnnResult, findpos=doublingsearch)
	P = ones(Int, length(Q))  # TODO: remove extra allocations

	for i in I
		d = 1.0
		for j in eachindex(P)
			p = findpos(Q[j].I, i, P[j])
			#@show (i, j, p, Q[j].W[p], W[j])
			d -= Q[j].W[p] * W[j]
			P[j] = p + 1
		end

		push!(res, i, d)
	end

	res
end

function search(idx::InvertedFile, q::DVEC, res::KnnResult)
	n = length(q)
	Q = Vector{PostingList}(undef, n) # TODO: remove extra allocations
	W = Vector{Float32}(undef, n)

	i = 0
	for (tokenID, weight) in q
		i += 1
		Q[i] = idx.lists[tokenID]
		W[i] = weight
		
	end

	if n == 2
		I = baezayates(Q[1].I, Q[2].I)
	else
		I = svs([plist.I for plist in Q]) # TODO: remove extra allocations
	end

	icos(W, Q, I, res)
end