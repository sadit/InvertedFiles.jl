# This file is part of InvertedFiles.jl

using SimilaritySearch, LinearAlgebra

export InvertedFile, prune, vectors


mutable struct InvertedFile{U,I,F} <: AbstractSearchContext
    lists::Dict{U,PostingList{I,F}}
    n::Int
end

InvertedFile{U,I,F}() where {U,I,F} = InvertedFile{U,I,F}(Dict{U,PostingList{I,F}}(), 0)
InvertedFile() = InvertedFile{UInt64,Int32,Float32}()

Base.show(io::IO, idx::InvertedFile) = print(io, "{$(typeof(idx)) vocsize=$(length(idx.lists)), n=$(idx.n)}")

"""
    Base.append!(idx::InvertedFile, db)

Appends all vectors in db to the index
"""
function Base.append!(idx::InvertedFile, db)
    for i in eachindex(db)
        push!(idx, i => db[i])
    end

    idx
end

"""
    push!(index::InvertedFile, p::Pair)

Inserts a weighted vector into the index.

"""
function Base.push!(idx::InvertedFile{TokenType,IntType,FloatType}, p::Pair) where {TokenType,IntType,FloatType}
    idx.n += 1
    id_, vec_ = p

    @inbounds for (tokenID, weight) in vec_
        P = if haskey(idx.lists, tokenID)
            idx.lists[tokenID]
        else
            idx.lists[tokenID] = PostingList{IntType,FloatType}()
        end

        push!(P.I, id_)
        push!(P.W, weight)
    end
end

"""
    vectors(idx::InvertedFile{U,I,F}; minweight=0.0, maxk=typemax(I), normalize=true) where {U,I,F} 

Reconstruct vectors from the inverted index.
You can prune the vectors accepting only entries having at least a weight of `minweight`,
or keeping top entries per posting list. You can also ignore taking attributes from posting lists larger
than `maxlen`. Default values don't change the original vectors.
Resulting vectors are normalized if `normalize=true`.

"""
function vectors(idx::InvertedFile{U,I,F}; minweight=0.0, top=typemax(I), maxlen=idx.n, normalize=false) where {U,I,F} 
    D = Dict{I,Dict{U,F}}()
    
    for (tokenID, plist) in idx.lists
        length(plist) > maxlen && continue
        plist = top < length(plist) ? topk(plist) : plist

        @inbounds for i in eachindex(plist)
            docID, weight = plist[i]            
            weight < minweight && continue

            v = get(D, docID, nothing)
            if v === nothing
                D[docID] = Dict{U,F}(tokenID => weight)
            else
                D[docID][tokenID] = weight
            end
        end
    end

    if normalize
        for v in values(D)
            normalize!(v)
        end
    end

    D
end
