# This file is part of InvertedFiles.jl

using Intersections: _sort!, _remove_empty!
import SimilaritySearch: search
export isearch, usearch, search, prepare_posting_lists_for_querying

function prepare_posting_lists_for_querying(idx, q)
	Q = valtype(idx.lists)[]
	for (tokenID, weight) in q
		plist = get(idx.lists, tokenID, nothing)
		if plist !== nothing
			plist = PostingList(plist, weight)
			push!(Q, plist)
		end
	end
	
	Q
end


"""
	isearch(idx::InvertedFile, q::DVEC, res::KnnResult)

Searches `q` in `idx` using the cosine dissimilarity, it computes a partial operation on `idx`. `res` specify the query.
"""
function isearch(idx::InvertedFile, q::DVEC, res::KnnResult)
	Q = prepare_posting_lists_for_querying(idx, q)

	bk(Q) do L, P
		w = 1.0
		@inbounds @simd for i in eachindex(P)
			w -= L[i].W[P[i]] * L[i].weight
		end

		push!(res, L[1].I[P[1]], w)
	end

	res
end

"""
	usearch(idx::InvertedFile, q::DVEC, res::KnnResult)

Searches `q` in `idx` using the cosine dissimilarity, it computes the full operation on `idx`. `res` specify the query
"""
#=function usearch(idx::InvertedFile, q::DVEC, res::KnnResult)
	L = prepare_posting_lists_for_querying(idx, q)
	sort!(L, by=first)
    P = ones(Int, length(L))

    @inbounds while true
        _remove_empty!(P, L)
        n = length(P)
        n == 0 && break
        n > 1 && _sort!(P, L)
        plist = L[1]
		_p = P[1]
        id = plist.I[_p]
		w = plist.W[_p]
        weight = convert(Float64, plist.weight * w)
        P[1] += 1

        for i in 2:n
            plist = L[i]
			_p = P[i]
			id_ = plist.I[_p]
			w = plist.W[_p]
            if id == id_
                weight += plist.weight * w
                P[i] = _p + 1
			else
				break
			end
        end

		push!(res, id, 1.0 - weight)
    end

    res

end =#

function usearch(idx::InvertedFile, q::DVEC, res::KnnResult)
	Q = prepare_posting_lists_for_querying(idx, q)
	umerge(Q) do L, P, m
		w = 1.0 - L[1].weight * L[1].W[P[1]]
		@inbounds @simd for i in 2:m
			w -= L[i].weight * L[i].W[P[i]]
		end

		@inbounds push!(res, L[1].I[P[1]], w)
	end

    res
end

"""
	search(idx::InvertedFile, q::DVEC, res::KnnResult; intersection=false)

Searches `q` in `idx` using the cosine dissimilarity; `res` specify the query.  If `intersection` is true, then
an approximation is computed based on the intersection of posting lists. Note that intersection could be faster for large lists.
"""

function search(idx::InvertedFile, q::DVEC, res::KnnResult; intersection=false)
	intersection ? isearch(idx, q, res) : usearch(idx, q, res)
end