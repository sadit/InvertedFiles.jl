# This file is part of InvertedFiles.jl

export vectors, topk

"""
vectors(idx::InvertedFile{U,I,F}; minweight=0.0, maxk=typemax(I), normalize=true) where {U,I,F} 

Reconstruct vectors from the inverted index.
You can prune the vectors accepting only entries having at least a weight of `minweight`,
or keeping top entries per posting list. You can also ignore taking attributes from posting lists larger
than `maxlen`. Default values don't change the original vectors.
Resulting vectors are normalized if `normalize=true`.

"""
function vectors(idx::InvertedFile{I,F,M}; minweight=0.0, top=typemax(I), maxlen=idx.n, normalize=false) where {I,F,M}
    L = idx.lists
    D = [Dict{Int,F}() for i in 1:idx.n]
    invmap = idx.map === nothing ? nothing : Dict(v => k for (k, v) in idx.map)

    for tokenID in eachindex(idx.lists)
        plist = idx.lists[tokenID]
        length(plist) > maxlen && continue
        plist = top < length(plist) ? topk(plist, top) : plist

        @inbounds for i in eachindex(plist)
            docID, weight = plist[i]
            weight < minweight && continue
            if invmap === nothing
                D[docID][tokenID] = weight
            else
                D[docID][invmap[tokenID]] = weight
            end
        end
    end

    normalize && normalize!(D)
    D
end

"""
    topk(plist::PostingList{I,F}, top) where {I,F}

Creates a new posting lists with the topk weighted entries
"""
function topk(plist::PostingList{I,F}, top) where {I,F}
    T = KnnResult(I[], F[], top)
    for i in eachindex(plist)
        push!(T, plist.I[i], -plist.W[i])
    end

    for i in eachindex(T.dist)
        T.dist[i] = -T.dist[i]
    end

    sort!(PostingList(T.id, T.dist))
end