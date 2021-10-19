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

@testset "InvertedFile" begin
    A = [normalize!(rand(300)) for i in 1:1000]
    B = [SVEC(enumerate(a)) for a in A]
    I = append!(InvertedFile(), B)

    k = 30
    for i in 1:10
        qid = rand(1:length(A))
        Ares = search(ExhaustiveSearch(CosineDistance(), A), A[qid], KnnResult(k))
        Bres = search(I, B[qid], KnnResult(k))
        @test scores(Ares, Bres).recall == 1.0
    end

    k = 30
    for i in 1:10
        @info i
        qid = rand(1:length(A))
        @time Ares = search(ExhaustiveSearch(CosineDistance(), A), A[qid], KnnResult(k))
        @time Bres = search(I, B[qid], KnnResult(k))
        @time Cres = search(I, B[qid], KnnResult(k); intersection=true)
        @test scores(Ares, Bres).recall == 1.0
        @test scores(Ares, Cres).recall == 1.0
    end

    # increasing sparsity of the arrays
    for A_ in A
        t = partialsort(A_, 7, rev=true)
        for i in eachindex(A_)
            A_[i] = A_[i] < t ? 0.0 : A_[i]
        end
    end

    create_sparse(A_) = SVEC([i => a for (i, a) in enumerate(A_) if a > 0.0])

    B = [create_sparse(A_) for A_ in A]
    I = append!(InvertedFile(), B)
    k = 1  # the aggresive cut of the attributes need a small k
    for i in 1:10
        @info i
        qid = rand(1:length(A))
        @time Ares = search(ExhaustiveSearch(CosineDistance(), A), A[qid], KnnResult(k))
        @time Bres = search(I, B[qid], KnnResult(k))
        @time Cres = search(I, B[qid], KnnResult(k); intersection=true)
        @test scores(Ares, Bres).recall == 1.0
        @test scores(Ares, Cres).recall > 0.8
        @show scores(Ares, Bres).recall, scores(Ares, Cres).recall
    end
end
