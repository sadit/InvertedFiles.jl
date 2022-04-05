```@meta
CurrentModule = InvertedFiles
```

# InvertedFiles.jl

InvertedFiles.jl is a library for construction and searching of InvertedFiles. Despite its name, it only works for in memory representations.

An inverted file is a sparse matrix that it is optimized to retrieve top-k columns under some distance function; in fact, it will compute `k` nearest neighbors.
This package implements both binary and floating point weighted inverted files. The search api is identical to that found in [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl).

Additionally, it defines convertions to traditional sparse matrices and also a convenient sparse vector based on dictionaries (only basic methods are supported).