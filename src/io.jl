# This file is part of InvertedFiles.jl

import SimilaritySearch: serializeindex, restoreindex

function serializeindex(file, parent::String, index::AbstractInvertedFile, meta, options::Dict)
    adj = StaticAdjacencyList(index.adj)
    I = copy(index; adj)
    file[joinpath(parent, "index")] = I
end

"""
    loadindex(...; staticgraph=false, parent="/")
    restoreindex(file, parent::String, index, meta, options::Dict; staticgraph=false)

load the inverted index optionally making the postings lists static or dynamic 
"""
function restoreindex(file, parent::String, index::WeightedInvertedFile, meta, options::Dict; staticgraph=false)
    adj = staticgraph ? index.adj : AdjacencyList(index.adj)
    WeightedInvertedFile(index; adj)
end

function restoreindex(file, parent::String, index::BinaryInvertedFile, meta, options::Dict; staticgraph=false)
    adj = staticgraph ? index.adj : AdjacencyList(index.adj)
    BinaryInvertedFile(index; adj)
end

