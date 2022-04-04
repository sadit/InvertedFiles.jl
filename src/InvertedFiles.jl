# This file is part of InvertedFiles.jl

module InvertedFiles
    using Intersections
    
    include("dvec.jl")
    include("svecutils.jl")
    include("sort.jl")
    include("winvfile.jl")
    include("winvfilesearch.jl")
    include("binvfile.jl")
    include("binvfilesearch.jl")
end
