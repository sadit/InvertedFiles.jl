```@meta

CurrentModule = InvertedFiles
DocTestSetup = quote
    using InvertedFiles
end
```

## KnrIndex
The `KnrIndex` index structure
```@docs
KnrIndex
````

## Searching the index
We follow the searching api of `SimilaritySearch` such that you can use `searchbatch`, and `allknn` for free.

```@docs
search
```

## Inserting elements into the index

```@docs
index!
append!
push!
```

## Ordering (reranking) strategies
KnrOrderingStrategies, DistanceOrdering, InternalDistanceOrdering, DistanceOnTopKOrdering

## Optimizing performance
```@docs
optimize!
```

## Computing references
The `KnrIndex` index uses a small set of references, that follow the dataset distribution to encode objects and
search construct and search the index. Please note that in average, we expect $n/m$ posting lists if `kbuild=1`, and therefore this will be the number of elements to verify. In practice, the distribution is far from being uniform and vary with the data. In part, this can be manipulated with a proper selection of the set of references.

```@docs
references
```

The function `references` is a convenient function to select references efficiently and easily.
In any case, it is possible to use any sampling or clustering algorithm to compute the set of references. See for example

- [`KCenters.jl`](https://github.com/sadit/KCenters.jl)
- [`Clustering.jl`](https://github.com/JuliaStats/Clustering.jl)