# This file is a part of NeighborhoodApproximationIndex.jl

using Test, SimilaritySearch, KCenters, InvertedFiles, LinearAlgebra
using SimilaritySearch: neighbors
using JET

function runtest(odist, ordering; dim, n, m,
    numcenters=5ceil(Int, sqrt(n)), k=10, centersrecall=0.95, kbuild=1, ksearch=1,
    minrecall=1.0, initial=:dnet, maxiters=0)
    A = randn(Float32, dim, n)
    B = randn(Float32, dim, m)
    X = MatrixDatabase(A)
    Q = MatrixDatabase(B)
    dist = SqL2Distance()
    seq = ExhaustiveSearch(dist, X)
    @info "================== creating gold standard ===================="
    gsearchtime = @elapsed Igold, Dgold = searchbatch(seq, Q, k)
    @info "creating the KnrIndex"
    refs = ExhaustiveSearch(; dist, db=references(dist, X, 32; initial, maxiters))

    indextime = @elapsed index = if ordering === nothing 
        KnrIndex(X, refs; kbuild, ksearch)
    else
        KnrIndex(X, refs, odist, ordering; kbuild, ksearch)
    end

    @test length(index) == length(X)
    @info "=== searching in the index odist=$odist, ordering=$ordering ==="
    @show dim, n, m , numcenters, k, centersrecall, ordering
    
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, Q, k)
    @test_call searchbatch(index, getcontext(index), Q, k) # test_call fails without context
    recall = macrorecall(Igold, Ires)
    @info "before optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime), index: $(tsearchtime), index-construction: $indextime" 

    @testset "saveindex and loadindex WeightedInvertedFile" begin
        tmpfile = tempname()
        @info "--- load and save!!!"
        saveindex(tmpfile, index; meta=[1, 2, 4, 8], store_db=false)
        let
            G, meta = loadindex(tmpfile, database(index); staticgraph=true)
            @test meta == [1, 2, 4, 8]
            @test G.invfile.adj isa StaticAdjacencyList
            @test 1.0 == macrorecall(Ires, searchbatch(G, Q, k)[1])
        end
    end
    @info "**** optimizing ParetoRadius() ****"
    opttime = @elapsed optimize_index!(index, ParetoRadius(); verbose=false)
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, Q, k)
    @test_call searchbatch(index, getcontext(index), Q, k)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime), index: $(tsearchtime), optimization-time: $opttime"
    
    @info "**** optimizing ParetoRecall() ****"
    opttime = @elapsed optimize_index!(index, ParetoRecall(); verbose=false)
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, Q, k)
    @test_call searchbatch(index, getcontext(index), Q, k)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime), index: $(tsearchtime), optimization-time: $opttime"
    #@test recall >= min(0.2, minrecall)

    @info "**** optimizing MinRecall(0.95) - $index ****"
    opttime = @elapsed optimize_index!(index, getcontext(index), MinRecall(0.95); verbose=false)
    tsearchtime = @elapsed Ires, Dres = searchbatch(index, getcontext(index), Q, k)
    @test_call searchbatch(index, getcontext(index), Q, k)
    recall = macrorecall(Igold, Ires)
    @info "AFTER optimization: $(index)" (recall=recall, qps=1/tsearchtime, gold_qps=1/gsearchtime)
    @info "searchtime: gold: $(gsearchtime), index: $(tsearchtime), optimization-time: $opttime"
    @test recall >= minrecall
    
    @info "********************* generic searches *******************"
    res = KnnResult(10)
    @time search(index, Q[1], res)
    res = reuse!(res)
    @time search(index, Q[2], res)
    F = sort!([length(neighbors(index.invfile.adj, i)) for i in eachindex(index.invfile.adj)], rev=true)

    #=
    @info "********************* SearchGraph ***************"
    neighborhood = SimilaritySearch.Neighborhood(reduce=IdentityNeighborhood(), logbase=2)
    G = SearchGraph(; dist, db=X, neighborhood)
    G.neighborhood.reduce = IdentityNeighborhood()
    index!(G, parallel_block=1000)
    opttime = @elapsed optimize_index!(G, MinRecall(0.95); verbose=true)
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

    @info "********************* Real search *********************"
    # most usages
    runtest(JaccardDistance(), DistanceOnTopKOrdering(300);
            dim, n, m, numcenters, k, centersrecall, kbuild=5, ksearch=5, minrecall=0.6)
    runtest(CosineDistance(), DistanceOnTopKOrdering(300);
            dim, n, m, numcenters, k, centersrecall, kbuild=5, ksearch=5, minrecall=0.2)
    runtest(nothing, nothing; dim, n, m, numcenters, k, centersrecall,
            kbuild=5, ksearch=5, minrecall=0.9)
end
