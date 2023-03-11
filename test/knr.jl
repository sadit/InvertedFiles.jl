# This file is a part of NeighborhoodApproximationIndex.jl

using Test, SimilaritySearch, InvertedFiles, LinearAlgebra, UnicodePlots
using SimilaritySearch: neighbors
using JET

function runtest(; dim, n, m,
    numcenters=5ceil(Int, sqrt(n)), k=10, centersrecall=0.95, kbuild=1, ksearch=1,
    parallel_block=256, ordering=DistanceOrdering(), minrecall=1.0, initial=:dnet, maxiters=0)
    A = randn(Float32, dim, n)
    B = randn(Float32, dim, m)
    X = MatrixDatabase(A)
    Q = MatrixDatabase(B)
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    @info "creating gold standard"
    gsearchtime = @elapsed Igold, Dgold = searchbatch(seq, Q, k)
    @info "creating the KnrIndex"
    indextime = @elapsed index = KnrIndex(dist, X; kbuild, ksearch, parallel_block, centersrecall, initial, maxiters, ordering)
    @test length(index) == length(X)
    @info "searching in the index"
    @show dim, n, m , numcenters, k, centersrecall, ordering
    
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, Q, k)
    @test_call searchbatch(index, Q, k)
    recall = macrorecall(Igold, Ires)
    @info "before optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), index-construction: $indextime"
    
    @info "**** optimizing ParetoRadius() ****"
    opttime = @elapsed optimize!(index, ParetoRadius(); verbose=false)
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, Q, k)
    @test_call searchbatch(index, Q, k)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), optimization-time: $opttime"
    
    @info "**** optimizing ParetoRecall() ****"
    opttime = @elapsed optimize!(index, ParetoRecall(); verbose=false)
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, Q, k)
    @test_call searchbatch(index, Q, k)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), optimization-time: $opttime"
    #@test recall >= min(0.2, minrecall)

    @info "**** optimizing MinRecall(0.95) ****"
    opttime = @elapsed optimize!(index, MinRecall(0.95); verbose=false)
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, Q, k)
    @test_call searchbatch(index, Q, k)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), index: $(tsearchtime * m), optimization-time: $opttime"
    @test recall >= minrecall
    
    @info "********************* generic searches *******************"
    res = KnnResult(10)
    @time search(index, Q[1], res)
    res = reuse!(res)
    @time search(index, Q[2], res)
    F = sort!([length(neighbors(index.invfile.adj, i)) for i in eachindex(index.invfile.adj)], rev=true)
    println(histogram(F, nbins=20))
    println(lineplot(F))

    #=
    @info "********************* SearchGraph ***************"
    neighborhood = SimilaritySearch.Neighborhood(reduce=IdentityNeighborhood(), logbase=2)
    G = SearchGraph(; dist, db=X, neighborhood)
    G.neighborhood.reduce = IdentityNeighborhood()
    index!(G, parallel_block=1000)
    opttime = @elapsed optimize!(G, MinRecall(0.95); verbose=true)
    @time Ires, Dres, tsearchtime = timedsearchbatch(G, Q, k)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(G)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime * m), G: $(tsearchtime * m), optimization-time: $opttime"
    # @test recall >= min(0.7, minrecall)
    =#
end

@testset "KnrIndex" begin
    centersrecall = 0.95
    # DistanceOnTopKOrdering and InternalDistanceOrdering need high `kbuild` and `ksearch` to generate
    # some discrimination power by the internal distance
    # useful for costly distance functions or whenever the dataset is pretty large
    m = 100
    n = 10^4
    dim = 4
    numcenters = 100
    k = 10
    runtest(; dim, n, m, numcenters, k, centersrecall,
            kbuild=5, ksearch=5, ordering=InternalDistanceOrdering(), minrecall=0.2)
    @info "********************* Real search (top-k) *********************"
    # useful for costly distance functions
    
    runtest(; dim, n, m, numcenters, k, centersrecall,
            kbuild=5, ksearch=5, ordering=DistanceOnTopKOrdering(1000), minrecall=0.8)
    @info "********************* Real search *********************"
    # most usages
    runtest(; dim, n, m, numcenters, k, centersrecall,
            kbuild=1, ksearch=1, ordering=DistanceOrdering(), minrecall=0.8)
end
