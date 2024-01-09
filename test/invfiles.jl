# This file is part of InvertedFiles.jl

using InvertedFiles, SimilaritySearch, SimilaritySearch.AdjacencyLists, LinearAlgebra
using Test, JET
using Random
Random.seed!(0)

@testset "WeightedInvertedFile" begin
    A = MatrixDatabase(normalize!(rand(300, 1000)))
    B = VectorDatabase([Dict(enumerate(a)) for a in A])

    # testing with Vector container (map=nothing)
    I = index!(WeightedInvertedFile(300, B))

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
        if i == 1
          @test_call search(I, B[qid], KnnResult(k))
        end
    end

    # testing with Dict container (map != nothing)
    I = index!(WeightedInvertedFile(300, B))

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
        if i == 1
          @test_call search(I, B[qid], KnnResult(k))
        end
    end

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
        if i == 1
          @test_call search(I, B[qid], KnnResult(k))
        end
    end

    ## working on sparse data
    # increasing sparsity of the arrays
    for A_ in A
        t = partialsort(A_, 7, rev=true)
        for i in eachindex(A_)
            A_[i] = A_[i] < t ? 0.0 : A_[i]
        end
        normalize!(A_)
    end

    create_sparse(A_) = Dict([i => a for (i, a) in enumerate(A_) if a > 0.0])

    #B = VectorDatabase([create_sparse(A_) for A_ in A])
    B = VectorDatabase(A)
    I = append_items!(WeightedInvertedFile(300), B)
    k = 1  # the aggresive cut of the attributes need a small k
    @test length(I) == length(B)
    for i in 1:10
        #@info i
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
        #@show recallscore(a.res, b.res)
        if i == 1
          @test_call search(I, B[qid], KnnResult(k))
        end
    end

    I = WeightedInvertedFile(300, B)
    @test length(I) == 0
    index!(I)
    @test length(I) == length(B)
    k = 1  # the aggresive cut of the attributes need a small k
    for i in 1:10
        # @info i
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
        #@show recallscore(a.res, b.res)
    end

    ak = allknn(ExhaustiveSearch(dist=NormalizedCosineDistance(), db=I.db), 3)[1]
    @test 1.0 == macrorecall(ak, allknn(I, 3)[1])
    
    @testset "saveindex and loadindex WeightedInvertedFile" begin
        tmpfile = tempname()
        @info "--- load and save!!!"
        saveindex(tmpfile, I; meta=[1, 2, 4, 8], store_db=false)
        let
            G, meta = loadindex(tmpfile, database(I); staticgraph=true)
            @test meta == [1, 2, 4, 8]
            @test G.adj isa StaticAdjacencyList
            @test 1.0 == macrorecall(ak, allknn(G, 3)[1])
        end
    end
end

@testset "BinaryInvertedFile" begin
    vocsize = 128
    n = 10_000
    m = 100
    len = 10
    k = 10
    db = VectorDatabase([sort!(unique(rand(1:vocsize, len))) for i in 1:n])
    queries = VectorDatabase([sort!(unique(rand(1:vocsize, len))) for i in 1:m])

    #for dist in [JaccardDistance(), DiceDistance(), CosineDistanceSet(), IntersectionDissimilarity()]
    for dist in [JaccardDistance()]
        S = ExhaustiveSearch(; dist, db)
        gI, gD = searchbatch(S, queries, k)

        IF = BinaryInvertedFile(vocsize, dist)
        append_items!(IF, db)
        iI, iD = searchbatch(IF, queries, k)
        ctx = getcontext(IF)
        @time search(IF, queries[1], getknnresult(k, ctx))
        @time search(IF, queries[2], getknnresult(k, ctx))
        @test_call search(IF, queries[2], getknnresult(k, ctx))
        recall = macrorecall(gI, iI)
        @show dist, recall
        @test recall > 0.95  # sets can be tricky since we can expect many similar distances
        err = 0.0
        for i in 1:m
            d = evaluate(L2Distance(), gD[:, i], iD[:, i])
            err += d
            if d > 0.1
                @info dist, i, gD[:, i], iD[:, i]
                @info dist, i, gI[:, i], iI[:, i]
                @info dist, i, queries[i]
            end
        end
        @show dist, err
        @test err < 0.01  # acc. floating point errors


    
    @testset "saveindex and loadindex BinaryInvertedFile" begin
        tmpfile = tempname()
        @info "--- load and save!!!"
        saveindex(tmpfile, IF; meta=[1, 2, 4, 8], store_db=false)
        let
            G, meta = loadindex(tmpfile, database(IF); staticgraph=true)
            @test meta == [1, 2, 4, 8]
            @test G.adj isa StaticAdjacencyList
            Iloaded, _ = searchbatch(G, queries, k)
            recall = macrorecall(gI, Iloaded)
            @test recall > 0.95
        end
    end

    end
end
