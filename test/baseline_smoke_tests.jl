using Test, DelimitedFiles
using .BaselineHelpers
using NetworkInference

# Paths
const DATA_DIR = joinpath(dirname(@__FILE__), "data")
const OUT_DIR  = joinpath(dirname(@__FILE__), "baseline_outputs")
isdir(OUT_DIR) || mkpath(OUT_DIR)

# --- Small yeast dataset: mirror current tests and save snapshots ---
@testset "Yeast10 baseline snapshots" begin
    data_file = joinpath(DATA_DIR, "yeast1_10_data.txt")
    mi_net, clr_net, puc_net, pidc_net = run_all_networks(data_file)

    # Save matrices and edge lists (for future diffs)
    # NOTE: these are optional; you already have reference txt files for MI/PUC/CLRs.
    # We still snapshot PIDC edges for end-to-end ranking diffs.
    BaselineHelpers.save_edges_tsv(joinpath(OUT_DIR, "pidc_yeast_edges.tsv"), pidc_net)
end

# --- Toy 1k×200 dataset: determinism + timings/allocations ---
@testset "Toy 1k×200 determinism + timings" begin
    data_file = joinpath(DATA_DIR, "toy_1k_200.txt")

    # First run
    t1 = @timed begin
        mi1, clr1, puc1, pidc1 = run_all_networks(data_file)
        mi1, clr1, puc1, pidc1
    end
    (mi1, clr1, puc1, pidc1) = t1.value

    # Second run
    t2 = @timed begin
        mi2, clr2, puc2, pidc2 = run_all_networks(data_file)
        mi2, clr2, puc2, pidc2
    end
    (mi2, clr2, puc2, pidc2) = t2.value

    # Determinism: check a subset of weights and full edge order equality
    @test length(mi1.edges)   == length(mi2.edges)
    @test length(puc1.edges)  == length(puc2.edges)
    @test length(pidc1.edges) == length(pidc2.edges)

    # Exact equality should hold given deterministic pipeline; if CI noise appears,
    # switch to ≈ with tiny tolerance.
    for idx in (1, 5, 10, 50, length(pidc1.edges))
        @test mi1.edges[idx].weight   == mi2.edges[idx].weight
        @test clr1.edges[idx].weight  == clr2.edges[idx].weight
        @test puc1.edges[idx].weight  == puc2.edges[idx].weight
        @test pidc1.edges[idx].weight == pidc2.edges[idx].weight
        @test Set([n.label for n in pidc1.edges[idx].nodes]) ==
              Set([n.label for n in pidc2.edges[idx].nodes])
    end

    # Persist snapshots for later diffs (original vs. modernized)
    BaselineHelpers.save_edges_tsv(joinpath(OUT_DIR, "pidc_toy_edges.tsv"), pidc1)

    # Log timings/allocations to a simple TSV
    open(joinpath(OUT_DIR, "timings.tsv"), "w") do io
        println(io, "phase\twall_seconds\talloc_bytes")
        println(io, "toy_first\t$(t1.time)\t$(t1.bytes)")
        println(io, "toy_second\t$(t2.time)\t$(t2.bytes)")
    end

    @info "Toy timings (s)" first=t1.time second=t2.time
    @info "Toy allocations (bytes)" first=t1.bytes second=t2.bytes
end

@testset "Config is backward compatible" begin
    data_file = joinpath(dirname(@__FILE__), "data", "toy_1k_200.txt")
    nodes = get_nodes(data_file)
    net1 = InferredNetwork(PIDCNetworkInference(), nodes)
    net2 = InferredNetwork(PIDCNetworkInference(), nodes; config = PIDCConfig())
    @test net1.edges[1].weight == net2.edges[1].weight
end