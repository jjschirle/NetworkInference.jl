# src/puc_pruned.jl
#
# Pruned PUC: use MI-based k-neighborhoods instead of all triplets.
# This is opt-in via PIDCConfig.triplet_block_k > 0.
#
# Option A (edge-centric, neighbor_mode = :union):
# For each unordered pair (x, z), use K(x,z) = N(x) ∪ N(z) as third-gene
# candidates. When k >= n-1, this reproduces the full PUC exactly (up to FP
# roundoff), because K(x,z) then contains all y ≠ x,z.

function compute_puc_pruned(nodes::Vector{Node};
    estimator::String = "maximum_likelihood",
    base::Int = 2,
    config::PIDCConfig)

    n = length(nodes)
    k = min(config.triplet_block_k, max(n - 1, 0))

    # If k == 0, or we haven't implemented this neighbor_mode yet, fall back.
    if k == 0 || config.neighbor_mode != :union
        return compute_puc_full(nodes; estimator = estimator, base = base)
    end

    # --- build NodePair cache (MI + specific information) ------------

    node_pairs = Array{NodePair}(undef, n, n)

    function get_mi_and_si(node1::Node, node2::Node)
        probabilities, probabilities1, probabilities2 =
        get_joint_probabilities(node1, node2, estimator)

        mi = apply_mutual_information_formula(
        probabilities, probabilities1, probabilities2, base)

        si1 = apply_specific_information_formula(
        probabilities, probabilities1, probabilities2, 1, base)

        si2 = apply_specific_information_formula(
        probabilities, probabilities2, probabilities1, 2, base)

        return mi, si1, si2
    end

    for i in 1:n
        for j in i+1:n
            mi, si1, si2 = get_mi_and_si(nodes[i], nodes[j])
            node_pairs[i, j] = NodePair(mi, si1)  # source = i, target = j
            node_pairs[j, i] = NodePair(mi, si2)  # source = j, target = i
        end
    end

    # --- MI-based neighbor lists for each gene -----------------------

    neighbors = Vector{Vector{Int}}(undef, n)

    for t in 1:n
        mivals = Vector{Tuple{Float64,Int}}()
        sizehint!(mivals, n - 1)

        for u in 1:n
            u == t && continue
            mi_tu = node_pairs[t, u].mi
            push!(mivals, (mi_tu, u))
        end

        sort!(mivals; by = x -> x[1], rev = true)
        k_eff = min(k, length(mivals))
        neighbors[t] = [mivals[i][2] for i in 1:k_eff]
    end

    # --- allocate PUC scores ----------------------------------------

    puc_scores = zeros(Float64, n, n)

    # Same clamping behavior as legacy increment_puc_scores.
    function increment_puc_scores!(x::Int, z::Int, mi::Float64, redundancy::Float64,
            scores::AbstractMatrix{Float64})
        puc_score = (mi - redundancy) / mi
        puc_score = isfinite(puc_score) && puc_score >= 0 ? puc_score : zero(puc_score)
        scores[x, z] += puc_score
        scores[z, x] += puc_score
    end

    # --- neighbor-based triplet loop (Option A: union of neighbors) --

    # We parallelize over x; each (x,z) pair is handled exactly once with z > x,
    # and we always update (x,z) & (z,x) together, so there are no write races.
    Threads.@threads for x in 1:n
        for z in x+1:n
        # Build candidate set K(x,z) = neighbors[x] ∪ neighbors[z]
        # (deduplicated, excluding x and z). We do this via sort + unique
        # to avoid heavy Set allocations, since |neighbors| ≤ 2k.
        cands = Vector{Int}()
        sizehint!(cands, length(neighbors[x]) + length(neighbors[z]))
        append!(cands, neighbors[x])
        append!(cands, neighbors[z])
        sort!(cands)

        last = 0
        for y in cands
            # Skip duplicates and self-pairs
            if y == last || y == x || y == z
                last = y
                continue
            end
            last = y

            # --- Contribution 1: target = z, sources = (x, y) ----------
            # This matches legacy:
            # get_puc(nodes[z], node_pairs[x,z], node_pairs[y,z], x, y, z, puc_scores)
            np_xz = node_pairs[x, z]   # source = x, target = z
            np_yz = node_pairs[y, z]   # source = y, target = z

            Rz = apply_redundancy_formula(
            nodes[z].probabilities,
            np_xz.si,
            np_yz.si,
            base
            )
            increment_puc_scores!(x, z, np_xz.mi, Rz, puc_scores)

            # --- Contribution 2: target = x, sources = (y, z) ----------
            # This matches legacy:
            # get_puc(nodes[x], node_pairs[y,x], node_pairs[z,x], y, z, x, puc_scores)
            # which increments PUC(y,x) with mi(y,x) and PUC(z,x) with mi(z,x).
            # For the pair (x,z), we care about the second one (mi(z,x)).
            np_yx = node_pairs[y, x]   # source = y, target = x
            np_zx = node_pairs[z, x]   # source = z, target = x

            Rx = apply_redundancy_formula(
            nodes[x].probabilities,
            np_yx.si,
            np_zx.si,
            base
            )
            increment_puc_scores!(x, z, np_zx.mi, Rx, puc_scores)
            end
        end
    end

    return puc_scores
end
