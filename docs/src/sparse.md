```@meta

CurrentModule = InvertedFiles
DocTestSetup = quote
    using InvertedFiles
end
```

## Sparse matrices
Inverted indexes/files are representations of sparse matrices optimized for certain operations.
We provide some functions to convert inverted files to sparse matrices.
```@docs
sparse
```

Inverted indexes constructors also support sparse matrices as input (wrapped on `MatrixDatabase` structs)
