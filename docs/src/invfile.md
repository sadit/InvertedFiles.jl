```@meta

CurrentModule = InvertedFiles
DocTestSetup = quote
    using InvertedFiles
end
```

# Inverted files
```@docs
append!
push!
```

```@docs
prepare_posting_lists_for_querying
search!
```

## WeightedInvertedFile
```@docs
WeightedInvertedFile
```

## BinaryInvertedFile
```@docs
BinaryInvertedFile
```

## Sparse matrices
Inverted indexes/files are representations of sparse matrices optimized for certain operations.
We provide some functions to convert inverted files to sparse matrices.
```@docs
sparse
sparsevec
```

Inverted indexes constructors also support sparse matrices as input (wrapped on `MatrixDatabase` structs)
