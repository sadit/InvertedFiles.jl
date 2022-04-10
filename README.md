[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sadit.github.io/InvertedFiles.jl/dev)
[![CI](https://github.com/sadit/InvertedFiles.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/sadit/InvertedFiles.jl/actions/workflows/ci.yml)

# Inverted Files

This package implements inverted files, also known as inverted indexes, that are data structures that represents a large sparse matrix, specially organized to compute some distance functions and fetch `k` nearest neighbors.
It is mainly used for full text search and other search tasks where data can be formulated as large sparse vectors.
In particular, the package implements three types of inverted files:

- `WeightedInvertedFile`: Inverted files for sparse vectors, it can solve $k$ nearest neighbors using the  normalized cosine distance, $1 - dot(u, q)$
- `BinaryInvertedFile`: Inverted file for sparse binary data, it can solve $k$ nearest neighbors using Jaccard, Dice, and Cosine distances, and also the intersection dissimilarity measure.
- `KnrIndex`: An approximated similarity search index based on inverted files. It supports general metric spaces.

These structs integrates with the `SimilaritySearch` environment, such that you can use it as a drop-in replacement of other indexes. In particular, inverted files are well-known for its scalability when the proper setup is used.

## Installation

```julia
using Pkg
Pkg.add("InvertedFiles")

```

## About the `KnrIndex`

This package defines the `KnrIndex` similarity search index that takes advantage of multithreading systems.
It is based on inverted files [`InvertedFiles`](https://github.com/sadit/InvertedFiles.jl) and [`SimilaritySearch`](https://github.com/sadit/SimilaritySearch.jl).
In particular, it supports any SemiMetric distance function, as defined in [`Distances`](https://github.com/JuliaStats/Distances.jl) package. For instance, you can use distances for vectors, sets, strings, etc.,
as defined in `SimilaritySearch` or [`StringDistances`](https://github.com/matthieugomez/StringDistances.jl) packages.

As `SearchGraph` in `SimilaritySearch`, the `KnrIndex` supports auto-configuration, using `optimize!`. Contrary to `SimilaritySearch`, the optimization it is only performed for searching purposes.

See [https://github.com/sadit/SimilaritySearchDemos](https://github.com/sadit/SimilaritySearchDemos), almost all examples should be reproduced using the `KnrIndex` defined in `NeighborhoodApproximationIndex`,
just using the index and calling `optimize!` with the corresponding arguments (see documentation for more details).

The basic ideas of this package are described in  

```
Edgar Chavez, Mario Graff, Gonzalo Navarro, Eric S. Tellez:
Near neighbor searching with K nearest references. Inf. Syst. 51: 43-61 (2015)

Eric S. Tellez, Edgar Chavez, Gonzalo Navarro: Succinct nearest neighbor search. Inf. Syst. 38(7): 1019-1030 (2013)
Eric S. Tellez, Edgar Chavez, Gonzalo Navarro: Succinct nearest neighbor search. SISAP 2011: 33-40 (2011)

```

The `KnrIndex` supports appending and automatic optimization to achieve some desired performance, also, user-based distance functions could work pretty fast since the indexes are written in the Julia language.
Nonetheless, the inverted files are plain in-memory structures without any kind of compression or compactness. The compressed datastructure can be found in the old C# library [natix](https://github.com/sadit/natix).
