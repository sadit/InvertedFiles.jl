var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"\nCurrentModule = InvertedFiles\nDocTestSetup = quote\n    using InvertedFiles\nend","category":"page"},{"location":"api/#WeightedInvertedFile","page":"API","title":"WeightedInvertedFile","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"WeightedInvertedFile","category":"page"},{"location":"api/#InvertedFiles.WeightedInvertedFile","page":"API","title":"InvertedFiles.WeightedInvertedFile","text":"struct WeightedInvertedFile <: AbstractInvertedFile\n\nAn inverted index is a sparse matrix representation of with floating point weights, it supports only positive non-zero values. This index is optimized to efficiently solve k nearest neighbors (cosine distance, using previously normalized vectors).\n\nParameters\n\nlists: posting lists (non-zero id-elements in rows)\nweights: non-zero weights (in rows)\nsizes: number of non-zero values in each element (non-zero values in columns)\nlocks: per-row locks for multithreaded construction\n\n\n\n\n\n","category":"type"},{"location":"api/#BinaryInvertedFile","page":"API","title":"BinaryInvertedFile","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"BinaryInvertedFile","category":"page"},{"location":"api/#InvertedFiles.BinaryInvertedFile","page":"API","title":"InvertedFiles.BinaryInvertedFile","text":"struct BinaryInvertedFile <: AbstractInvertedFile\n\nCreates a binary weighted inverted index. An inverted index is an sparse matrix representation optimized for computing k nn elements (columns) under some distance.\n\nProperties:\n\ndist: Distance function to be applied, valid values are: IntersectionDissimilarity(), DiceDistance(), JaccardDistance(), and `CosineDistanceSet()\nlists: posting lists (non-zero values of the rows in the matrix)\nsizes: number of non-zero values per object (number of non-zero values per column)\nlocks: Per row locks for multithreaded construction\n\n\n\n\n\n","category":"type"},{"location":"api/#Searching-algorithms","page":"API","title":"Searching algorithms","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"prepare_posting_lists_for_querying\nsearch","category":"page"},{"location":"api/#InvertedFiles.prepare_posting_lists_for_querying","page":"API","title":"InvertedFiles.prepare_posting_lists_for_querying","text":"prepare_posting_lists_for_querying(idx::AbstractInvertedFile, q, pools=getpools(idx), tol=1e-6)\n\nFetches and prepares the involved posting lists to solve q\n\n\n\n\n\n","category":"function"},{"location":"api/#SimilaritySearch.search","page":"API","title":"SimilaritySearch.search","text":"search(idx::AbstractInvertedFile, q, res::KnnResult; pools=nothing)\n\nSearches q in idx using the cosine dissimilarity, it computes the full operation on idx. res specify the query\n\n\n\n\n\nsearch(callback::Function, idx::WeightedInvertedFile, Q, P; t=1)\n\nFind candidates for solving query Q using idx. It calls callback on each candidate (objID, dist)\n\nArguments:\n\ncallback: callback function on each candidate\nidx: inverted index\nQ: the set of involved posting lists, see prepare_posting_lists_for_querying\nP: a vector of starting positions in Q (initial state as ones)\n\n\n\n\n\nsearch(callback::Function, idx::BinaryInvertedFile, Q, P, t)\n\nFind candidates for solving query Q using idx. It calls callback on each candidate (objID, dist)\n\nArguments\n\ncallback: callback function on each candidate\nidx: inverted index\nQ: the set of involved posting lists, see prepare_posting_lists_for_querying\nP: a vector of starting positions in Q (initial state as ones)\nt: threshold (t=1 union, t > 1 solves the t-threshold problem)\n\n\n\n\n\n","category":"function"},{"location":"api/#Sparse-matrices","page":"API","title":"Sparse matrices","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Inverted indexes/files are representations of sparse matrices optimized for certain operations. We provide some functions to convert inverted files to sparse matrices.","category":"page"},{"location":"api/","page":"API","title":"API","text":"sparse\nsparsevec","category":"page"},{"location":"api/#SparseArrays.sparse","page":"API","title":"SparseArrays.sparse","text":"sparse(idx::BinaryInvertedFile, one::Type{RealType}=1f0)\n\nCreates an sparse matrix (from SparseArrays) from idx using one as value.\n\n   I  \n   ↓    1 2 3 4 5 … n  ← J\n L[1] = 0 1 0 0 1 … 0\n L[2] = 1 0 0 1 0 … 1\n L[3] = 1 0 1 0 0 … 1\n ⋮\n L[m] = 0 0 1 1 0 … 0\n\n\n\n\n\nsparse(idx::WeightedInvertedFile)\n\nCreates an sparse matrix (from SparseArrays) from idx\n\n\n\n\n\n","category":"function"},{"location":"api/#SparseArrays.sparsevec","page":"API","title":"SparseArrays.sparsevec","text":"sparsevec(vec::DVEC{Ti,Tv}, m=0) where {Ti<:Integer,Tv<:Number}\n\nCreates a sparse vector from a DVEC sparse vector\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"Inverted indexes constructors also support sparse matrices as input (wrapped on MatrixDatabase structs)","category":"page"},{"location":"api/#Dictionary-based-sparse-vectors","page":"API","title":"Dictionary-based sparse vectors","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Some application domains could take advantage of hash based sparse vectors, and some of them are the target of InvertedFiles, therefore, the package also provide a partial implementation of sparse vectors using Dict.","category":"page"},{"location":"api/","page":"API","title":"API","text":"dvec\nDVEC\nSVEC\nSVEC32\nSVEC64\nnnz\nfindmax\nargmax\nmaximum\nfindmin\nargmin\nminimum\nnormalize!\ndot\nnorm\nzero\nadd!\nsum\n+\n-\n*\n/\ncentroid\nevaluate\nNormalizedAngleDistance\nNormalizedCosineDistance\nAngleDistance\nCosineDistance\nevaluate","category":"page"},{"location":"api/#InvertedFiles.dvec","page":"API","title":"InvertedFiles.dvec","text":"dvec(x::AbstractSparseVector)\n\nConverts an sparse vector into a DVEC sparse vector\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.normalize!","page":"API","title":"LinearAlgebra.normalize!","text":"normalize!(bow::DVEC)\n\nInplace normalization of bow\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.dot","page":"API","title":"LinearAlgebra.dot","text":"dot(a::DVEC, b::DVEC)::Float64\n\nComputes the dot product for two DVEC vectors\n\n\n\n\n\n","category":"function"},{"location":"api/#LinearAlgebra.norm","page":"API","title":"LinearAlgebra.norm","text":"norm(a::DVEC)::Float64\n\nComputes a normalized DVEC vector\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.zero","page":"API","title":"Base.zero","text":"zero(::Type{DVEC{Ti,Tv}}) where {Ti,Tv}\n\nCreates an empty DVEC vector\n\n\n\n\n\n","category":"function"},{"location":"api/#InvertedFiles.add!","page":"API","title":"InvertedFiles.add!","text":"add!(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::AbstractSparseArray) where {Ti,Tv<:Real}\nadd!(a::DVEC{Ti,Tv}, b::Pair{Ti,Tv}) where {Ti,Tv<:Real}\n\nUpdates a to the sum of a+b\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.sum","page":"API","title":"Base.sum","text":"Base.sum(col::AbstractVector{<:DVEC})\n\nComputes the sum of the given list of vectors\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.:+","page":"API","title":"Base.:+","text":"+(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n+(a::DVEC, b::Pair)\n\nComputes the sum of a and b\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.:-","page":"API","title":"Base.:-","text":"-(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n\nSubstracts of b of a\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.:*","page":"API","title":"Base.:*","text":"*(a::DVEC{Ti,Tv}, b::DVEC{Ti,Tv}) where {Ti,Tv<:Real}\n*(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise product of a and b\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.:/","page":"API","title":"Base.:/","text":"/(a::DVEC{K, V}, b::F) where K where {V<:Real} where {F<:Real}\n\nComputes the element-wise division of a and b\n\n\n\n\n\n","category":"function"},{"location":"api/#InvertedFiles.centroid","page":"API","title":"InvertedFiles.centroid","text":"centroid(cluster::AbstractVector{<:DVEC})\n\nComputes a centroid of the given list of DVEC vectors\n\n\n\n\n\n","category":"function"},{"location":"api/#Distances.evaluate","page":"API","title":"Distances.evaluate","text":"evaluate(::NormalizedCosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\nIt supposes that bags are normalized (see normalize! function)\n\n\n\n\n\nevaluate(::CosineDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the cosine distance between two DVEC sparse vectors\n\n\n\n\n\nevaluate(::NormalizedAngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle  between two DVEC sparse vectors\n\nIt supposes that all bags are normalized (see normalize! function)\n\n\n\n\n\nevaluate(::AngleDistance, a::DVEC, b::DVEC)::Float64\n\nComputes the angle between two DVEC sparse vectors\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = InvertedFiles","category":"page"},{"location":"#InvertedFiles.jl","page":"Home","title":"InvertedFiles.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"InvertedFiles.jl is a library for construction and searching of InvertedFiles. Despite its name, it only works for in memory representations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"An inverted file is a sparse matrix that it is optimized to retrieve top-k columns under some distance function; in fact, it will compute k nearest neighbors. This package implements both binary and floating point weighted inverted files. The search api is identical to that found in SimilaritySearch.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Additionally, it defines convertions to traditional sparse matrices and also a convenient sparse vector based on dictionaries (only basic methods are supported).","category":"page"}]
}
