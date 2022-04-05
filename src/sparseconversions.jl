# This file is part of InvertedFiles.jl

export dvec
import SparseArrays: sparsevec, sparse

"""
    dvec(x::AbstractSparseVector)

Converts an sparse vector into a DVEC sparse vector
"""
function dvec(x::AbstractSparseVector)
    DVEC{eltype(x.nzind), eltype(x.nzval)}(t => v for (t, v) in zip(x.nzind, x.nzval))
end

sparse2dvec(x) = dvec

"""
    sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}

Creates a sparse vector from a DVEC sparse vector
"""
function sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}
    I = Ti[]
    F = Tv[]

    for (t, weight) in vec
        if t > 0
            push!(I, t)
            push!(F, weight)
        end
    end

    if m == 0
        sparsevec(I, F)
    else
        sparsevec(I, F, m)
    end
end

"""
    sparse(idx::BinaryInvertedFile{IntVec}, one::Type{RealType}=1f0)

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
function sparse(idx::BinaryInvertedFile{IntVec}, one::Type{RealType}=1f0) where {IntVec,RealType<:Real}
    n = length(idx)
    I = eltype(IntVec)[]
    J = eltype(IntVec)[]
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
function sparse(idx::WeightedInvertedFile{IntVec,RealVec}) where {IntVec,RealVec}
    n = length(idx)
    I = eltype(IntVec)[]
    J = eltype(IntVec)[]
    F = eltype(RealVec)[]
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
