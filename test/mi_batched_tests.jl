# test/mi_batched_tests.jl
using Test
using DelimitedFiles
using NetworkInference

@testset "Batched MI equals baseline" begin
    data_dir = joinpath(dirname(@__FILE__), "data")
    data_file = joinpath(data_dir, "yeast1_10_data.txt")
    mi_benchmark = readdlm(joinpath(data_dir, "mi.txt"))

    # Force batched path
    cfg = PIDCConfig(batch_size_genes=3)

    nodes = get_nodes(data_file)
    mi_net_batched = InferredNetwork(MINetworkInference(), nodes; config=cfg)

    # Compare a few selected edges (same indices as existing tests)
    for i in (1, 5, 10, 20, 40)
        @test mi_net_batched.edges[i].weight ≈ mi_benchmark[2*i, 3] atol = 1e-9
    end

    # Sanity: diagonal is zero in the MI matrix that produced these edges
    # (not directly exposed, but we ensure network has as many edges as baseline)
    @test length(mi_net_batched.edges) == 45
end
