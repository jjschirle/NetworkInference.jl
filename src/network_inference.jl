# Network inference algorithms and the InferredNetwork type. The algorithms MI, CLR, PUC and
# PIDC are explained in http://biorxiv.org/content/early/2017/04/26/082099 - along with terms
# such as specific information, proportional unique contribution, context, etc.

# Network inference algorithms
abstract type AbstractNetworkInference end
struct MINetworkInference <: AbstractNetworkInference end
struct CLRNetworkInference <: AbstractNetworkInference end
struct PUCNetworkInference <: AbstractNetworkInference end
struct PIDCNetworkInference <: AbstractNetworkInference end

# Context trait
apply_context(::MINetworkInference) = false
apply_context(::CLRNetworkInference) = true
apply_context(::PUCNetworkInference) = false
apply_context(::PIDCNetworkInference) = true

# PUC trait
get_puc(::MINetworkInference) = false
get_puc(::CLRNetworkInference) = false
get_puc(::PUCNetworkInference) = true
get_puc(::PIDCNetworkInference) = true

# For sorting the edges
get_weight(edge::Edge) = edge.weight

# Gets the joint probability distribution for two Nodes.
function get_joint_probabilities(node1, node2, estimator)

    frequencies = get_frequencies_from_bin_ids(
        node1.binned_values,
        node2.binned_values,
        node1.number_of_bins,
        node2.number_of_bins
    )

    probabilities = get_probabilities(estimator, frequencies)
    # probabilities is already property of a node, but doing this gets correct array shapes.
    # Also, for MI and CLR, it means that we don't assume that the marginal probabilities for
    # a node are always the same, no matter what the second node is, meaning that we can use
    # estimators other than maximum likelihood. (We still can't do this for PUC and PIDC,
    # because we do make that assumption for 3-node joint distributions, in get_puc.)
    probabilities1 = sum(probabilities, dims = 2)
    probabilities2 = sum(probabilities, dims = 1)

    return (probabilities, probabilities1, probabilities2)

end

# Gets the mutual information between all pairs of Nodes.
function get_mi_scores(nodes, number_of_nodes, estimator, base; config::PIDCConfig = PIDCConfig())

    # Legacy path
    function get_mi(node1, node2, i, j, base, mi_scores)
        probabilities, probabilities1, probabilities2 = get_joint_probabilities(node1, node2, estimator)
        mi = apply_mutual_information_formula(probabilities, probabilities1, probabilities2, base)
        mi_scores[i, j] = mi
        mi_scores[j, i] = mi
    end

    mi_scores = SharedArray{Float64}(number_of_nodes, number_of_nodes)

    @sync @distributed for i in 1 : number_of_nodes
        for j in i+1 : number_of_nodes
            get_mi(nodes[i], nodes[j], i, j, base, mi_scores)
        end
    end

    return mi_scores

end

# # Gets the proportional unique contribution between all pairs of Nodes.
# function get_puc_scores(nodes, number_of_nodes, estimator, base;
#     config::PIDCConfig = PIDCConfig())

#     if config.triplet_block_k <= 0
#         # Full legacy PUC
#         return compute_puc_full(nodes; estimator = estimator, base = base)
#     else
#         # Pruned, neighbor-based PUC
#         return compute_puc_pruned(nodes; estimator = estimator, base = base, config = config)
#     end
# end

function get_puc_scores(nodes, number_of_nodes, estimator, base;
    config::PIDCConfig = PIDCConfig())
    if config.triplet_block_k > 0
        # Pruned PUC
        if config.triplet_backend == :distributed
            return compute_puc_pruned_dist(nodes; estimator = estimator, base = base, config = config)
        else
            # default: threads
            return compute_puc_pruned( nodes; estimator = estimator, base = base, config = config)
        end
    else
        # Full PUC (no pruning)
        return compute_puc_full(nodes; estimator = estimator, base = base, config = config)
    end
end


# Applies context to the raw edge weights.
function get_weights(inference, scores, number_of_nodes, nodes)

    # In their respective original implementations, CLR and PIDC applied network context in slightly
    # different ways. Those differences are respected here; in informal tests, they have not been
    # found to make much of a difference.
    function get_weight(::PIDCNetworkInference, i, j, scores, weights, nodes)
        score = scores[i, j]
        scores_i = vcat(scores[1:i-1, i], scores[i+1:end, i])
        scores_j = vcat(scores[1:j-1, j], scores[j+1:end, j])
        try
            weights[i, j] = cdf(fit(Gamma, scores_i), score) + cdf(fit(Gamma, scores_j), score)
        catch
            # println(string("Gamma distribution failed for ", nodes[i].label, " and ", nodes[j].label, "; used normal instead."))
            apply_clr_context(i, j, score, scores_i, scores_j, weights)
        end
    end

    function get_weight(::CLRNetworkInference, i, j, scores, weights, nodes)
        score = scores[i, j]
        scores_i = vcat(scores[1:i-1, i], scores[i+1:end, i])
        scores_j = vcat(scores[1:j-1, j], scores[j+1:end, j])
        apply_clr_context(i, j, score, scores_i, scores_j, weights)
    end

    function apply_clr_context(i, j, score, scores_i, scores_j, weights)
        difference_i = score - mean(scores_i)
        difference_j = score - mean(scores_j)
        weights[i, j] = sqrt(
            (var(scores_i) == 0 || difference_i < 0 ? 0 : difference_i^2 / var(scores_i)) +
            (var(scores_j) == 0 || difference_j < 0 ? 0 : difference_j^2 / var(scores_j))
        )
    end

    weights = SharedArray{Float64}(number_of_nodes, number_of_nodes)

    @sync @distributed for i in 1 : number_of_nodes
        for j in i+1 : number_of_nodes
            get_weight(inference, i, j, scores, weights, nodes)
        end
    end

    return weights

end

"""
InferredNetwork type. Represents a weighted, fully connected network, where an
edges's weight indicates the relative confidence of that edge existing in the true
network.

Fields:
* `nodes`: array of all the nodes, in an arbitrary order
* `edges`: array of all the edges, in descending order of weight
"""
struct InferredNetwork
    nodes::Array{Node}
    edges::Array{Edge}
end

# Constructs an InferredNetwork given a network inference algorithm and an array of
# Nodes.
#
# Keyword arguments:
# - estimator: algorithm for estimating the probability distribution
# (The "maximum_likelihood" estimator is recommended for PUC and PIDC, because speedups
# are made here, based on the assumption that the marginal probability distribution for
# a node, from the joint distribution with any two other nodes is always the same. If
# the joint distributions are estimated using other estimators, this assumption is
# violated for PUC and PIDC in get_puc and get_joint_probabilities.)
# - base: base for the information measures
function InferredNetwork(inference::AbstractNetworkInference, nodes::Array{Node}; estimator = "maximum_likelihood", base = 2, config::PIDCConfig = PIDCConfig())

    # Constants and containers
    number_of_nodes = length(nodes)
    edges = Array{Edge}(undef, binomial(number_of_nodes, 2))

    # # Get the raw scores (Unchanged logic; just forward config)
    # scores = get_puc(inference) ?
    #     get_puc_scores(nodes, number_of_nodes, estimator, base; config = config) :
    #     get_mi_scores(nodes, number_of_nodes, estimator, base; config = config)
    # Get raw scores
    if get_puc(inference)
        mi_scores, scores = get_puc_scores(
            nodes, number_of_nodes, estimator, base; config = config
        )

        # Only dump MI for PIDC (not PUC)
        if isa(inference, PIDCNetworkInference) && config.dump_mi_path !== nothing
            dump_mi_scores(mi_scores, nodes, config)
        end
    else
        scores = get_mi_scores(
            nodes, number_of_nodes, estimator, base; config = config
        )
    end
    
    # Apply context if necessary
    if apply_context(inference)
        weights = get_weights(inference, scores, number_of_nodes, nodes)
    else
        weights = scores
    end

    # Get edges from scores
    index = 0
    for i in 1 : number_of_nodes
        node1 = nodes[i]
        for j in i+1 : number_of_nodes
            index += 1
            node2 = nodes[j]
            edges[index] = Edge(
                [node1, node2],
                weights[i, j]
            )
        end
    end
    sort!(edges; by = get_weight, rev = true)

    return InferredNetwork(nodes, edges)

end
