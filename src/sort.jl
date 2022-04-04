"""
    sortlastpush!(idlist)

Sorts the last push in place. It implements insertion sort that it is efficient due to the expected
distribution of the items being inserted (it is expected to be really near of its sorted position)
"""
function sortlastpush!(id::AbstractVector)
    sp = 1
    pos = N = lastindex(id)
    id_ = id[end]

    @inbounds while pos > sp && id_ < id[pos-1]
        pos -= 1
    end

    @inbounds if pos < N
        while N > pos
            id[N] = id[N-1]
            N -= 1
        end

        id[N] = id_
    end

    nothing
end

function sortlastpush!(id::AbstractVector, dist)
    sp = 1
    pos = N = lastindex(id)
    id_ = id[end]
    dist_ = dist[end]
    
    @inbounds while pos > sp && id_ < id[pos-1]
        pos -= 1
    end

    @inbounds if pos < N
        while N > pos
            id[N] = id[N-1]
            dist[N] = dist[N-1]
            N -= 1
        end

        dist[N] = dist_
        id[N] = id_
    end

    nothing
end