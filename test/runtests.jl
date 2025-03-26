# This file is part of InvertedFiles.jl
using InvertedFiles
using Aqua
Aqua.test_all(InvertedFiles, ambiguities=false)
Aqua.test_ambiguities([InvertedFiles])

include("invfiles.jl")
