# This file is part of Intersections.jl
export doublingsearch, binarysearch

"""
	binarysearch(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A` in the range `sp:ep`
"""
function binarysearch(A, x, sp=1, ep=length(A))
	while sp < ep
		mid = div(sp + ep, 2)
		@inbounds if x <= A[mid]
			ep = mid
		else
			sp = mid + 1
		end
	end
	
	@inbounds x <= A[sp] ? sp : sp + 1
end

"""
	doublingsearch(A, x, sp=1, ep=length(A))

Finds the insertion position of `x` in `A`, starting at `sp`
"""
function doublingsearch(A, x, sp=1, ep=length(A))
	p = 0
    i = 1

    @inbounds while sp+i <= ep && A[sp+i] < x
		p = i
		i += i
    end

    binarysearch(A, x, sp + p, min(ep, sp+i))
end

