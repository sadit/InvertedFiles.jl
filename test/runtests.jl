# This file is part of InvertedFiles.jl

using InvertedFiles, SimilaritySearch, LinearAlgebra
using Test
using Random
Random.seed!(0)

@testset "DVEC" begin
    cmpex(u, v) = abs(u[1] - v[1]) < 1e-3 && u[2] == v[2]

    aL = []
    AL = []
    for i in 1:10
        A = rand(300)
        B = rand(300)
        a = SVEC(k => v for (k, v) in enumerate(A))
        b = SVEC(k => v for (k, v) in enumerate(B))

        @test abs(norm(A) - norm(a)) < 1e-3
        @test abs(norm(B) - norm(b)) < 1e-3
        normalize!(A); normalize!(a)
        normalize!(B); normalize!(b)
        @test abs(norm(a) - 1.0) < 1e-3
        @test abs(norm(b) - 1.0) < 1e-3

        @test abs(dot(a, b) - dot(A, B)) < 1e-3
        @test abs(maximum(a) - maximum(A)) < 1e-3
        @test abs(minimum(a) - minimum(A)) < 1e-3

        
        @test cmpex(findmax(a), findmax(A))
        @test cmpex(findmin(a), findmin(A))

        push!(aL, a)
        push!(AL, A)
    end

    @test (norm(sum(AL)) - norm(sum(aL))) < 1e-3

    adist = AngleDistance()
    cdist = CosineDistance()

    for i in 1:length(aL)-1
        @test abs(evaluate(adist, aL[i], aL[i+1]) - evaluate(adist, AL[i], AL[i+1])) < 1e-3
        @test abs(evaluate(cdist, aL[i], aL[i+1]) - evaluate(cdist, AL[i], AL[i+1])) < 1e-3
    end
end

@testset "WeightedInvertedFile" begin
    A = [normalize!(rand(300)) for i in 1:1000]
    B = VectorDatabase([SVEC(enumerate(a)) for a in A])

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

    create_sparse(A_) = SVEC([i => a for (i, a) in enumerate(A_) if a > 0.0])

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

    for dist in [JaccardDistance(), DiceDistance(), CosineDistanceSet(), IntersectionDissimilarity()]
        S = ExhaustiveSearch(; dist, db)
        gI, gD = searchbatch(S, queries, k)

        IF = BinaryInvertedFile(vocsize, dist)
        append!(IF, db)
        iI, iD = searchbatch(IF, queries, k)
        
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