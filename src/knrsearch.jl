# This file is a part of InvertedFiles.jl

"""
    search(idx::KnrIndex, q, res::KnnResult; t=1, ksearch=idx.opt.ksearch, ordering=idx.ordering, pool=getpools(D))

Searches nearest neighbors of `q` inside the `index` under the distance function `dist`.
"""
function search(idx::KnrIndex, q, res::KnnResult; t=1, pools=getpools(idx), ksearch=idx.opt.ksearch)
    enc = getencodeknnresult(ksearch, pools)
    search(idx.centers, q, enc)
    Q = select_posting_lists(idx.invfile, enc) do plist
      true
    end
    
    search_(idx, q, enc, Q, res, t, idx.ordering)
end

function search_(idx::KnrIndex, q, _, Q, res::KnnResult, t, ::DistanceOrdering)
    dist = idx.dist
    P_ = getcachepositions(length(Q), idx.invfile)

    cost = xmergefun(Q, P_; t) do L, P, _
        @inbounds objID = _get_key(L[1].list, P[1])
        d = evaluate(dist, q, database(idx, objID))
        push_item!(res, IdWeight(objID, d))
    end

    SearchResult(res, cost)
end

function search_(idx::KnrIndex, q, enc, Q, res::KnnResult, t, ordering::DistanceOnTopKOrdering)
    enc = reuse!(enc, ordering.top)
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
