# This file is part of InvertedFiles.jl

import SparseArrays: sparse

"""
    sparse(idx::BinaryInvertedFile, one::Type{RealType}=1f0)

Creates an sparse matrix (from SparseArrays) from `idx` using `one` as value.

```
   I  
   ↓    1 2 3 4 5 … n  ← J
 L[1] = 0 1 0 0 1 … 0
 L[2] = 1 0 0 1 0 … 1
 L[3] = 1 0 1 0 0 … 1
 ⋮
 L[m] = 0 0 1 1 0 … 0
```
"""
function sparse(idx::BinaryInvertedFile, one::Type{RealType}=1f0) where {RealType<:Real}
    n = length(idx)
    I = eltype(idx.lists[1])[]
    J = eltype(idx.lists[1])[]
    F = RealType[]
    sizehint!(I, n)
    sizehint!(J, n)
    sizehint!(F, n)

    for i in eachindex(idx.lists)
        L = idx.lists[i]
        for j in L
            push!(I, i)
            push!(J, j)
            push!(F, one)
        end
    end

    sparse(I, J, F, length(idx.lists), n)
end


"""
    sparse(idx::WeightedInvertedFile) 
 
Creates an sparse matrix (from SparseArrays) from `idx`
"""
function sparse(idx::WeightedInvertedFile)
    n = length(idx)
    I = eltype(idx.lists[1])[]
    J = eltype(idx.lists[1])[]
    F = eltype(idx.weights[1])[]
    sizehint!(I, n)
    sizehint!(J, n)
    sizehint!(F, n)

    for i in eachindex(idx.lists)
        L = idx.lists[i]
        W = idx.weights[i]
        for j in L
            push!(I, i)
            push!(J, j)
            push!(F, W[j])
        end
    end

    sparse(I, J, F, length(idx.lists), n)
end
