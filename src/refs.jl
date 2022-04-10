export references

"""
    references(dist::SemiMetric, db; <kwargs>) -> SubDatabase

Computes a set of references, a sample of `db`, using the [`KCenters`](@ref) specification, it is a wrapper to `kcenters` function in the `KCenters.jl` package.
The set of references are computed taking into account the specified `r` but also the distance function `dist`.

# Arguments

- `dist`: a distance function
- `db`: the database to be sampled

# Keyword arguments
- `k::Int`: the number of centers to compute, defaults to `sqrt(|db|)`
- `sample::Real`: indicates the sampling size before computing the set of `k` references, defaults to `log(|db|) k`; `sample=0` means for no sampling.
- `maxiters::Int`: number of iterationso  of the Lloyd algorithm that should be applied on the initial computation of centers, that is, `maxiters > 0` applies `maxiters` iterations of the algorithm.
- `tol::Float64`: change tolerance to stop the Lloyd algorithm (error changes smaller than `tol` among iterations will stop the algorithm)
- `initial`: initial centers or a strategy to compute initial centers, symbols `:rand`, `:fft`, and `:dnet`.
There are several interactions between initial values and parameter interactions (described in `KCenters` object), for instance,
the `maxiters > 0` apply the Lloyd's algorithm to the initial computation of references.

- if `initial=:rand`:
  - `maxiters = 0` will retrieve a simple random sampling
  - `maxiters > 0' achieve kmeans-centroids, `maxiters` should be set appropiately for the the dataset
- if `initial=:dnet`:
  - `maxiters = 0` computes a pure density-net
  - `maxiters > 0` will compute a kmeans centroids but with an initialization based on the dnet
- if `initial=:fft`:
  - `maxiters = 0` computes `k` centers with the farthest first traversal algorithm
  - `maxiters > 0` will use the FFT based kcenters as initial points for the Lloyd algorithm

Note 1: `maxiters > 0` needs to compute centroids and these centroids should be _defined_
for the specific data model, and also be of use in the specific metric distance and error function.

Note 2: The error function is defined as the mean of distances of all objects in `db` to its associated nearest centers in each iteration.

Note 3: The computation of references on large databases can be prohibitive, in these cases please consider to work on a sample of `db`
"""
function references(
      dist::SemiMetric, db;
      k=ceil(Int, sqrt(length(db))),
      sample=ceil(Int, log2(length(db)) * k),
      maxiters=0,
      tol=0.001, initial=:rand)
    0 < k < length(db) || throw(ArgumentError("invalid relation between k and n, must follow 0 < k < n"))
    C = if sample > 0
      s = unique(rand(eachindex(db), sample))
      kcenters(dist, SubDatabase(db, s), k; initial, maxiters, tol)
    else
      kcenters(dist, db, k; initial, maxiters, tol)
    end
    
    C.centers[C.dmax .> 0.0]  # centers covering more than itself
end
