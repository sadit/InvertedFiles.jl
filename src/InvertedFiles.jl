# This file is part of InvertedFiles.jl

module InvertedFiles
    using Intersections
    import SimilaritySearch: search, getpools, getknnresult
    
    include("dvec.jl")
    include("sortedintset.jl")
    include("plists.jl")
    include("sort.jl")
    include("invfile.jl")
    include("winvfile.jl")
    include("binvfile.jl")
    include("invfilesearch.jl")
    include("winvfilesearch.jl")
    include("binvfilesearch.jl")
    include("sparseconversions.jl")
end
