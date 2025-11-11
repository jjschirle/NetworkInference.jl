#!/usr/bin/env julia
using Profile, ProfileSVG # or StatProfilerHTML if you prefer
using NetworkInference
using .BaselineHelpers

const DATA = joinpath(@__DIR__, "data", "toy_1k_200.txt")

# Warm-up
run_all_networks(DATA)

# Profile PIDC only
Profile.clear()
@profile begin
    _, _, _, pidc = run_all_networks(DATA)
    pidc
end

# Save profile artifact
svg = joinpath(@__DIR__, "baseline_outputs", "pidc_profile.svg")
isdir(dirname(svg)) || mkpath(dirname(svg))
ProfileSVG.save(svg)
println("Saved profile to ", svg)
