# This file is part of InvertedFiles.jl

using InvertedFiles, SimilaritySearch, LinearAlgebra
using Test
using Random
Random.seed!(0)

@testset "WeightedInvertedFile" begin
    A = [normalize!(rand(300)) for i in 1:1000]
    B = VectorDatabase([Dict(enumerate(a)) for a in A])

    # testing with Vector container (map=nothing)
    I = append!(WeightedInvertedFile(300), B)

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
    end

    # testing with Dict container (map != nothing)
    I = append!(WeightedInvertedFile(300), B)

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
    end

    k = 30
    for i in 1:10
        @info i
        qid = rand(1:length(A))
        @time a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        @time b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
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
    I = append!(WeightedInvertedFile(300), B)
    k = 1  # the aggresive cut of the attributes need a small k
    for i in 1:10
        @info i
        qid = rand(1:length(A))
        @time a = search(ExhaustiveSearch(NormalizedCosineDistance(), A), A[qid], KnnResult(k))
        @time b = search(I, B[qid], KnnResult(k))
        @test recallscore(a.res, b.res) == 1.0
        @show recallscore(a.res, b.res)
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
        append!(IF, db)
        iI, iD = searchbatch(IF, queries, k)
        @time search(IF, queries[1], SimilaritySearch.getknnresult(k))
        @time search(IF, queries[2], SimilaritySearch.getknnresult(k))
        
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
    end
end
