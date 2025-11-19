# src/mi_batched.jl
# Batched, threaded MI over Node objects, using PIDCConfig.
# This file is included inside the NetworkInference module.

using Base.Threads

# Thread-local scratch buffers sized once; reused for all pairs
mutable struct MIScratch
    freq::Matrix{Int}          # (max_b, max_b)
    pxy::Matrix{Float64}       # (max_b, max_b)
    px::Matrix{Float64}        # (max_b, 1)
    py::Matrix{Float64}        # (1, max_b)
end

# Return views with the same rank/layout as the original implementation:
# p_xy :: (nbx, nby)
# p_x  :: (nbx, 1)
# p_y  :: (1, nby)
@inline function view2!(scratch::MIScratch, nbx::Int, nby::Int)
    f   = @view scratch.freq[1:nbx, 1:nby]
    pxy = @view scratch.pxy[1:nbx, 1:nby]
    px  = @view scratch.px[1:nbx, 1:1]    # (nbx, 1)
    py  = @view scratch.py[1:1, 1:nby]    # (1, nby)
    return f, pxy, px, py
end

"""
    compute_mi_batched(nodes; estimator, base, config) -> Matrix{Float64}

Compute pairwise mutual information between all pairs of `nodes` in tiles of size
`config.batch_size_genes`, threaded over tiles. Uses each Node's precomputed
`binned_values` and `number_of_bins`.
"""
function compute_mi_batched(
    nodes::Vector{Node};
    estimator::AbstractString = "maximum_likelihood",
    base::Real = 2,
    config::PIDCConfig = PIDCConfig(),
)::Matrix{Float64}

    n = length(nodes)
    MI = zeros(Float64, n, n)
    if n == 0
        return MI
    end

    # Max number of bins across all nodes
    max_b = maximum(n_.number_of_bins for n_ in nodes)

    # Build list of upper-triangle tiles
    B = max(config.batch_size_genes, 1)
    tile_pairs = Vector{Tuple{UnitRange{Int},UnitRange{Int}}}()
    i0 = 1
    while i0 <= n
        I = i0:min(i0 + B - 1, n)
        j0 = i0
        while j0 <= n
            J = j0:min(j0 + B - 1, n)
            push!(tile_pairs, (I, J))
            j0 += B
        end
        i0 += B
    end

    # Thread over tiles; each thread gets its own scratch
    @threads for t = 1:length(tile_pairs)
        I, J = tile_pairs[t]

        # Allocate scratch once per thread
        scratch = MIScratch(
            zeros(Int,     max_b, max_b),
            zeros(Float64, max_b, max_b),
            zeros(Float64, max_b, 1),   # px: (max_b, 1)
            zeros(Float64, 1, max_b),   # py: (1, max_b)
        )

        for i in I
            # Ensure we only do upper triangle i < j
            jstart = first(J)
            if first(J) <= i <= last(J)
                jstart = i + 1
            end

            for j = jstart:last(J)
                nbx = nodes[i].number_of_bins
                nby = nodes[j].number_of_bins
                binx = nodes[i].binned_values
                biny = nodes[j].binned_values

                f, pxy, px, py = view2!(scratch, nbx, nby)

                # zero contingency
                @inbounds fill!(f, 0)

                # fill contingency table
                @inbounds @simd for k = 1:length(binx)
                    f[binx[k], biny[k]] += 1
                end

                # p(x,y) from estimator; this returns a matrix same size as f
                pxy .= get_probabilities(estimator, f)

                # p(x): sum over columns
                @inbounds for a in 1:nbx
                    s = 0.0
                    @simd for b in 1:nby
                        s += pxy[a, b]
                    end
                    px[a, 1] = s
                end

                # p(y): sum over rows
                @inbounds for b in 1:nby
                    s = 0.0
                    @simd for a in 1:nbx
                        s += pxy[a, b]
                    end
                    py[1, b] = s
                end

                # MI with same shapes the original formula expects
                mi = apply_mutual_information_formula(pxy, px, py, base)

                @inbounds begin
                    MI[i, j] = mi
                    MI[j, i] = mi
                end
            end
        end
    end

    # zero diagonal
    @inbounds for d in 1:n
        MI[d, d] = 0.0
    end

    return MI
end
