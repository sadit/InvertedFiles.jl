# This file is a part of InvertedFiles.jl

struct KnrMergeOutput{QType,KnrType<:KnrIndex}
    idx::KnrType
    q::QType
    res::KnnResult
end

function Intersections.onmatch!(output::KnrMergeOutput, L, P, m::Int)
    @inbounds objID = getkey(L[1].list, P[1])
    d = evaluate(distance(output.idx), output.q, database(output.idx, objID))
    push_item!(output.res, IdWeight(objID, d))
end

"""
    search(idx::KnrIndex, q, res::KnnResult; t=1, ksearch=idx.opt.ksearch, ordering=idx.ordering, pool=getpools(D))

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(idx::KnrIndex, q, res::KnnResult; t=1, pools=getpools(idx), ksearch=idx.opt.ksearch)
    enc = encode_object_res!(idx.encoder, q)
    idx.invfile isa WeightedInvertedFile && knr_as_similarity!(enc)
    Q = select_posting_lists(idx.invfile, enc) do plist
      true
    end
    
    search_(idx, q, enc, Q, res, t, idx.ordering)
end

function search_(idx::KnrIndex, q, _, Q, res::KnnResult, t, ::DistanceOrdering)
    dist = distance(idx)
    P = getcachepositions(length(Q))
    cost = xmerge!(KnrMergeOutput(idx, q, res), Q, P; t)
    SearchResult(res, cost)
end

function search_(idx::KnrIndex, q, enc, Q, res::KnnResult, t, ordering::DistanceOnTopKOrdering)
    enc = encode_object_res!(idx.encoder, q; k=ordering.top)
    pools = getpools(idx.invfile)
    search_invfile(idx.invfile, Q, enc, t, pools)

    dist = distance(idx)
    for item in enc
        @inbounds push_item!(res, item.id, evaluate(dist, q, database(idx, item.id)))
    end

    SearchResult(res, length(enc))
end

function search_(idx::KnrIndex, q, _, Q, res::KnnResult, t, ::InternalDistanceOrdering)
    pools = getpools(idx.invfile)
    search_invfile(idx.invfile, Q, res, t, pools)
end
