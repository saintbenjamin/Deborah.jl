# ============================================================================
# src/Sarah/SeedManager.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SeedManager

import StatsBase
import ..JobLoggerTools

"""
    struct RNGPool

Container for multiple independent RNGs used in bootstrap sampling.

# Fields
- `rng::StatsBase.Xoshiro`: RNG for full set.
- `rng_lb::StatsBase.Xoshiro`: RNG for labeled set.
- `rng_tr::StatsBase.Xoshiro`: RNG for training set.
- `rng_bc::StatsBase.Xoshiro`: RNG for bias correction set.
- `rng_ul::StatsBase.Xoshiro`: RNG for unlabeled set..
"""
struct RNGPool
    rng::StatsBase.Xoshiro    
    rng_lb::StatsBase.Xoshiro
    rng_tr::StatsBase.Xoshiro
    rng_bc::StatsBase.Xoshiro
    rng_ul::StatsBase.Xoshiro
end

"""
    setup_rng(
        ranseed::Int=850528, 
        jobid::Union{Nothing, String}=nothing
    ) -> StatsBase.Xoshiro

Initialize a pseudorandom number generator (PRNG) with a fixed seed.

# Arguments
- `ranseed`: Seed value for the RNG (default = `850528`).
- `jobid`: Optional job identifier for logging.

# Returns
- [`StatsBase.Xoshiro`](https://docs.julialang.org/en/v1/stdlib/Random/#Random.Xoshiro): Seeded PRNG instance.

# Side Effects
Prints the seed and RNG type to `stdout`.
"""
function setup_rng(
    ranseed::Int=850528, 
    jobid::Union{Nothing, String}=nothing
)::StatsBase.Xoshiro
    rng = StatsBase.Xoshiro(ranseed)
    JobLoggerTools.println_benji("Seed for PRNG = $(ranseed)", jobid)
    JobLoggerTools.println_benji("PRNG for StatsBase = $(rng)", jobid)
    return rng
end

"""
    setup_rng_pool(
        ranseed::Int=850528, 
        jobid::Union{Nothing, String}=nothing
    ) -> RNGPool

Initialize an [`RNGPool`](@ref) with distinct but reproducible seeds derived from a base seed.

# Arguments
- `ranseed`: Base seed value (default = `850528`).
- `jobid`: Optional job identifier for logging.

# Returns
- `RNGPool`: Struct containing five independent seeded RNGs.

# Side Effects
Prints the seed initialization information to stdout.
"""
function setup_rng_pool(
    ranseed::Int=850528, 
    jobid::Union{Nothing, String}=nothing
)::RNGPool
    JobLoggerTools.println_benji("Base seed for PRNG_pool = $(ranseed)", jobid)
    return RNGPool(
        StatsBase.Xoshiro(ranseed),
        StatsBase.Xoshiro(ranseed + 1),
        StatsBase.Xoshiro(ranseed + 2),
        StatsBase.Xoshiro(ranseed + 3),
        StatsBase.Xoshiro(ranseed + 4),
    )
end

end  # module SeedManager