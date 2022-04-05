# This file is part of InvertedFiles.jl

module InvertedFiles
    using Intersections

    include("dvec.jl")
    include("sortedintset.jl")
    include("plists.jl")
    include("sort.jl")
    include("invfile.jl")
    include("winvfile.jl")
    include("winvfilesearch.jl")
    include("binvfile.jl")
    include("binvfilesearch.jl")
    include("sparseconversions.jl")
end
