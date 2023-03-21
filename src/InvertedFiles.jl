# This file is part of InvertedFiles.jl

module InvertedFiles
    using Intersections
    import SimilaritySearch:
        search,  index!, getpools, getknnresult, getminbatch
    using SimilaritySearch.AdjacencyLists
    using Base.Threads: SpinLock
    using Polyester
    
    include("sortedintset.jl")
    include("plists.jl")
    include("invfile.jl")
    include("winvfile.jl")
    include("binvfile.jl")
    include("invfilesearch.jl")
    include("winvfilesearch.jl")
    include("binvfilesearch.jl")

    include("knr.jl")
    include("knrsearch.jl")
    include("knropt.jl")
    include("io.jl")

    function __init__()
        __init__invfile()
    end
end
