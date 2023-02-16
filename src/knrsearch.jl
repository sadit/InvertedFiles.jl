# This file is a part of NeighborhoodApproximationIndex.jl

using InvertedFiles: getcachepositions

"""
    search(idx::KnrIndex, q, res::KnnResult; ksearch=idx.opt.ksearch, ordering=idx.ordering, pools=getpools(D))

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(idx::KnrIndex, q, res::KnnResult; pools=getpools(idx), ksearch=idx.opt.ksearch)
    enc = getencodeknnresult(ksearch, pools)
    search(idx.centers, q, enc)
    ifpools = getpools(idx.invfile)
    Q = prepare_posting_lists_for_querying(idx.invfile, enc, ifpools)
    P = getcachepositions(length(Q), ifpools)
    search_(idx, q, enc, Q, P, res, idx.ordering)
end

function search_(idx::KnrIndex, q, _, Q, P_, res::KnnResult, ::DistanceOrdering)
    dist = idx.dist
    cost = umerge(Q, P_) do L, P, _
        @inbounds objID = _get_key(L[1].list, P[1])
        @inbounds push_item!(res, objID, evaluate(dist, q, database(idx, objID)))
    end

    SearchResult(res, cost)
end

function search_(idx::KnrIndex, q, enc, Q, P_, res::KnnResult, ordering::DistanceOnTopKOrdering)
    enc = reuse!(enc, ordering.top)
    search(idx.invfile, Q, P_, 1) do objID, d
        @inbounds push_item!(enc, objID, d)
    end

    dist = distance(idx)
    for objID in idview(enc)
        @inbounds push_item!(res, objID, evaluate(dist, q, database(idx, objID)))
    end

    SearchResult(res, length(enc))
end

function search_(idx::KnrIndex, q, _, Q, P_, res::KnnResult, ::InternalDistanceOrdering)
    cost = search(idx.invfile, Q, P_, 1) do objID, d
        @inbounds push_item!(res, objID, d)
    end

    SearchResult(res, cost)
end
