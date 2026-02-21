# ============================================================================
# src/Sarah/BlockSizeSuggester.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BlockSizeSuggester

import Printf: @sprintf
import OrderedCollections
import Statistics
import ..DatasetPartitioner
import ..JobLoggerTools
import ..TOMLLogger

"""
    suggest_opt_block_sizes(
        N_cnf::Integer, 
        N_lb::Integer, 
        N_bc::Integer, 
        N_ul::Integer,
        base_block_size::Integer; 
        min_block::Int=1
    ) -> Dict{Symbol,Int}

Return optimal block sizes for each subset as a `Dict{Symbol,Int}` based on a
given optimal block size for the full dataset.

# Arguments
- `N_cnf::Integer`: Total number of configurations (full dataset size).
- `N_lb::Integer`: Number of configurations in the labeled set.
- `N_bc::Integer`: Number of configurations in the bias-correction set.
- `N_ul::Integer`: Number of configurations in the unlabeled set.
- `base_block_size::Integer`: Optimal block size for the full dataset (`:all`).

# Keyword Arguments
- `min_block::Int=1`: Lower bound for any suggested block size.

# Returns
A `Dict{Symbol,Int}` with keys:
- `:all` → `base_block_size`
- `:lb`  → `base * (N_lb / N_cnf)`
- `:bc`  → `base * (N_bc / N_cnf)`
- `:ul`  → `base * (N_ul / N_cnf)`

All scaled values are rounded to the nearest integer and clamped to `min_block`.
"""
function suggest_opt_block_sizes(
    N_cnf::Integer, 
    N_lb::Integer, 
    N_bc::Integer, 
    N_ul::Integer,
    base_block_size::Integer; 
    min_block::Int=1
)
    # Basic sanity checks
    JobLoggerTools.assert_benji(
        N_cnf > 0,
        "N_cnf must be positive"
    )
    JobLoggerTools.assert_benji(
        0 <= N_lb <= N_cnf,
        "N_lb must be in [0, N_cnf]"
    )
    JobLoggerTools.assert_benji(
        0 <= N_bc <= N_cnf,
        "N_bc must be in [0, N_cnf]"
    )
    JobLoggerTools.assert_benji(
        0 <= N_ul <= N_cnf,
        "N_ul must be in [0, N_cnf]"
    )
    JobLoggerTools.assert_benji(
        base_block_size > 0,
        "base_block_size must be positive"
    )
    JobLoggerTools.assert_benji(
        min_block > 0,
        "min_block must be positive"
    )

    # local helper: scale and clamp
    safe_scale(base::Integer, num::Integer, den::Integer) = begin
        JobLoggerTools.assert_benji(
            den > 0,
            "denominator must be positive"
        )
        JobLoggerTools.assert_benji(
            0 <= num <= den,
            "numerator must be in [0, den]"
        )
        val = round(Int, base * (num / den))
        return max(min_block, val)
    end

    blk_all = Int(base_block_size)
    blk_lb  = safe_scale(base_block_size, N_lb, N_cnf)
    blk_bc  = safe_scale(base_block_size, N_bc, N_cnf)
    blk_ul  = safe_scale(base_block_size, N_ul, N_cnf)

    return Dict{Symbol,Int}(
        :all => blk_all,
        :lb  => blk_lb,
        :bc  => blk_bc,
        :ul  => blk_ul,
    )
end

# -----------------------------------------------------------------------------
# Compatibility wrapper: accept a `partition` object and forward to the main API
# (Useful for Deborah/Esther that already construct DatasetPartitionInfo)
# -----------------------------------------------------------------------------
"""
    suggest_opt_block_sizes(
        partition::DatasetPartitioner.DatasetPartitionInfo, 
        base_block_size; 
        min_block::Int=1
    ) -> Dict{Symbol,Int}

Compatibility wrapper that extracts `N_cnf`, `N_lb`, `N_bc`, `N_ul` from `partition`
and forwards to the integer-argument method.
"""
function suggest_opt_block_sizes(
    partition::DatasetPartitioner.DatasetPartitionInfo, 
    base_block_size::Integer; 
    min_block::Int=1
)
    JobLoggerTools.assert_benji(
        hasproperty(partition, :N_cnf),
        "partition must have field N_cnf"
    )
    JobLoggerTools.assert_benji(
        hasproperty(partition, :N_lb),
        "partition must have field N_lb"
    )
    JobLoggerTools.assert_benji(
        hasproperty(partition, :N_bc),
        "partition must have field N_bc"
    )
    JobLoggerTools.assert_benji(
        hasproperty(partition, :N_ul),
        "partition must have field N_ul"
    )

    return suggest_opt_block_sizes(
        getfield(partition, :N_cnf),
        getfield(partition, :N_lb),
        getfield(partition, :N_bc),
        getfield(partition, :N_ul),
        base_block_size; min_block=min_block,
    )
end

end  # module BlockSizeSuggester