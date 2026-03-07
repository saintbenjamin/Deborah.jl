# ============================================================================
# src/Sarah/BootstrapRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BootstrapRunner

import ..Random
import ..Statistics

import ..JobLoggerTools
import ..DatasetPartitioner
import ..SeedManager
import ..Bootstrap

"""
    run_bootstrap!(
        bootstrap_data::Dict{Symbol, Any},
        trace_data::Dict{String, Vector{Vector{T}}},
        partition::DatasetPartitioner.DatasetPartitionInfo,
        N_bs::Int,
        rng_pool::SeedManager.RNGPool,
        opt_blk_size::Dict{Symbol, Int},
        method::String,
        jobid::Union{Nothing, String}=nothing
    ) where T<:Real -> Nothing

Perform block-bootstrap resampling on a single ensemble's trace data and populate
`bootstrap_data[:mean]` in place.

# Arguments
- `bootstrap_data::Dict{Symbol, Any}`  
  Preallocated containers for output statistics (e.g. `:mean`). This function
  writes into `bootstrap_data[:mean][label][bs_idx][trace_idx]`.

- `trace_data::Dict{String, Vector{Vector{T}}}`  
  Input traces organized as `label => [trM1, trM2, ...]`
  (each `trMi::Vector{T}` over configurations).  
  Typical labels include: `"Y_lb"`, `"Y_tr"`, `"Y_bc"`, `"Y_ul"` (and possibly `"Y_info"` which should be ignored).

- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo)  
  Holds configuration counts per label (e.g., `N_lb`, `N_tr`, `N_bc`, `N_ul`), used
  to validate lengths / iterate groups.

- `N_bs::Int`  
  Number of bootstrap replicates to generate.

- [`rng_pool::SeedManager.RNGPool`](@ref Deborah.Sarah.SeedManager.RNGPool)  
  Pool of RNGs (e.g., `Random.Xoshiro`), typically one per thread.

- `opt_blk_size::Dict{Symbol, Int}`  
  Per-label optimal block sizes, keyed by `:lb`, `:tr`, `:bc`, `:ul`.

- `method::String`  
  Block-bootstrap scheme to use (case-sensitive):
  - `"nonoverlapping"` : Nonoverlapping Block Bootstrap (NBB) — resample disjoint
    blocks of length `blk_size`.
  - `"moving"`         : Moving Block Bootstrap (MBB) — candidate blocks are all
    length-`blk_size` sliding windows.
  - `"circular"`       : Circular Block Bootstrap (CBB) — like MBB but windows
    wrap around the end (circular indexing).
  Any other string should raise an error.

- `jobid::Union{Nothing, String}` (optional)  
  Identifier for logging/debugging.

# Behavior
- For each label in `trace_data` (excluding metadata like `"Y_info"`), and for each
  trace index within that label, draw `N_bs` bootstrap replicates using the chosen
  `method` and the label's `blk_size = opt_blk_size[label_as_symbol]`.
- For each replicate, resample blocks with replacement until reaching (or slightly
  exceeding) the original series length; truncate the last block if necessary.
- Compute the sample mean of each resampled series and store it into
  `bootstrap_data[:mean][label][bs_idx][trace_idx]`.

# Returns
- `Nothing` — modifies `bootstrap_data` in place.

# Notes
- Set `blk_size = 1` to recover the i.i.d. bootstrap.
- Ensure the trace length is adequate relative to `blk_size` (especially for MBB/CBB).
- Randomness is sourced from `rng_pool`; per-thread RNG selection is recommended.
- Labels are expected to align with `partition` counts; mismatches should be logged/errored.
"""
function run_bootstrap!(
    bootstrap_data::Dict{Symbol, Any},
    trace_data::Dict{String, Vector{Vector{T}}},
    partition::DatasetPartitioner.DatasetPartitionInfo,
    N_bs::Int,
    rng_pool::SeedManager.RNGPool,
    opt_blk_size::Dict{Symbol, Int},
    method::String,
    jobid::Union{Nothing, String}=nothing
) where T<:Real
    w_lb = partition.N_lb / partition.N_cnf
    w_ul = partition.N_ul / partition.N_cnf

    N_all = partition.N_cnf
    N_lb  = partition.N_lb
    N_tr  = partition.N_tr
    N_bc  = partition.N_bc
    N_ul  = partition.N_ul

    Y_info = haskey(trace_data, "Y_info") && !isempty(trace_data["Y_info"]) ? trace_data["Y_info"][1] : Float64[]
    Y_lb   = haskey(trace_data, "Y_lb")   && !isempty(trace_data["Y_lb"])   ? trace_data["Y_lb"][1]   : Float64[]
    Y_bc   = haskey(trace_data, "Y_bc")   && !isempty(trace_data["Y_bc"])   ? trace_data["Y_bc"][1]   : Float64[]
    Y_ul   = haskey(trace_data, "Y_ul")   && !isempty(trace_data["Y_ul"])   ? trace_data["Y_ul"][1]   : Float64[]
    YP_bc  = haskey(trace_data, "YP_bc")  && !isempty(trace_data["YP_bc"])  ? trace_data["YP_bc"][1]  : Float64[]
    YP_ul  = haskey(trace_data, "YP_ul")  && !isempty(trace_data["YP_ul"])  ? trace_data["YP_ul"][1]  : Float64[]

    mean_Y_info = bootstrap_data[:mean]["Y_info"]
    mean_Y_lb   = bootstrap_data[:mean]["Y_lb"]
    mean_Y_bc   = bootstrap_data[:mean]["Y_bc"]
    mean_Y_ul   = bootstrap_data[:mean]["Y_ul"]
    mean_YP_bc  = bootstrap_data[:mean]["YP_bc"]
    mean_YP_ul  = bootstrap_data[:mean]["YP_ul"]

    mean_YmYP   = bootstrap_data[:mean]["YmYP"]
    mean_Y_P1   = bootstrap_data[:mean]["Y_P1"]
    mean_Y_P2   = bootstrap_data[:mean]["Y_P2"]

    starts_all, nblk_all, last_all = Bootstrap.ensure_plan(nothing, rng_pool.rng,    N_all, opt_blk_size[:all], N_bs; method=method)
    starts_lb,  nblk_lb,  last_lb  = Bootstrap.ensure_plan(nothing, rng_pool.rng_lb, N_lb,  opt_blk_size[:lb],  N_bs; method=method)
    starts_bc,  nblk_bc,  last_bc  = Bootstrap.ensure_plan(nothing, rng_pool.rng_bc, N_bc,  opt_blk_size[:bc],  N_bs; method=method)
    starts_ul,  nblk_ul,  last_ul  = Bootstrap.ensure_plan(nothing, rng_pool.rng_ul, N_ul,  opt_blk_size[:ul],  N_bs; method=method)

    ps_info = (N_all>0) ? Bootstrap.prefix_sums(Y_info) : nothing
    ps_lb   = (N_lb >0) ? Bootstrap.prefix_sums(Y_lb)   : nothing
    ps_bc   = (N_bc >0) ? Bootstrap.prefix_sums(Y_bc)   : nothing
    ps_ul   = (N_ul >0) ? Bootstrap.prefix_sums(Y_ul)   : nothing
    ps_YPbc = (N_bc >0) ? Bootstrap.prefix_sums(YP_bc)  : nothing
    ps_YPul = (N_ul >0) ? Bootstrap.prefix_sums(YP_ul)  : nothing

    JobLoggerTools.log_stage_sub1_benji("Bootstrap resampling loop ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        @inbounds for ibs in 1:N_bs
            mean_Y_info[ibs] = Bootstrap.mean_from_plan(ps_info, Y_info, starts_all, N_all, opt_blk_size[:all], nblk_all, last_all, ibs; method=method)
            mean_Y_lb[ibs]   = (N_lb>0) ? Bootstrap.mean_from_plan(ps_lb, Y_lb, starts_lb, N_lb, opt_blk_size[:lb], nblk_lb, last_lb, ibs; method=method) : 0.0

            if N_bc>0 && N_tr>0
                mean_Y_bc[ibs]  = Bootstrap.mean_from_plan(ps_bc,   Y_bc,   starts_bc, N_bc, opt_blk_size[:bc], nblk_bc, last_bc, ibs; method=method)
                mean_YP_bc[ibs] = Bootstrap.mean_from_plan(ps_YPbc, YP_bc,  starts_bc, N_bc, opt_blk_size[:bc], nblk_bc, last_bc, ibs; method=method)
            else
                mean_Y_bc[ibs]  = 0.0
                mean_YP_bc[ibs] = 0.0
            end

            if N_ul>0 && N_tr>0
                mean_Y_ul[ibs]  = Bootstrap.mean_from_plan(ps_ul,   Y_ul,   starts_ul, N_ul, opt_blk_size[:ul], nblk_ul, last_ul, ibs; method=method)
                mean_YP_ul[ibs] = Bootstrap.mean_from_plan(ps_YPul, YP_ul,  starts_ul, N_ul, opt_blk_size[:ul], nblk_ul, last_ul, ibs; method=method)
            else
                mean_Y_ul[ibs]  = 0.0
                mean_YP_ul[ibs] = 0.0
            end

            if N_tr == 0
                mean_YmYP[ibs] = 0.0
                mean_Y_P1[ibs] = mean_Y_lb[ibs]
                mean_Y_P2[ibs] = mean_Y_lb[ibs]
            else
                mean_YmYP[ibs] = mean_Y_bc[ibs] - mean_YP_bc[ibs]
                mean_Y_P1[ibs] = mean_Y_ul[ibs] + mean_YmYP[ibs]  # YP_UL + (BC-YP_BC)
                mean_Y_P2[ibs] = w_lb * mean_Y_lb[ibs] + w_ul * mean_Y_P1[ibs]
            end
        end
    end
end

"""
    run_bootstrap!(
        bootstrap_data::Dict{Symbol, Any},
        trace_data::Dict{String, Vector{Vector{T}}},
        Q_moment::Dict{String, Vector{T}},
        partition::DatasetPartitioner.DatasetPartitionInfo,
        N_bs::Int,
        rng_pool::SeedManager.RNGPool,
        opt_blk_size::Dict{Symbol, Int},
        method::String,
        jobid::Union{Nothing, String}=nothing
    ) where T<:Real -> Nothing

Perform block-bootstrap resampling for both traces and precomputed
``Q``-moment series, writing bootstrap sample means into `bootstrap_data[:mean]`
in place.

# Arguments
- `bootstrap_data::Dict{Symbol, Any}`  
  Output containers; this routine writes results to `bootstrap_data[:mean]`
  (organized by label and series index).

- `trace_data::Dict{String, Vector{Vector{T}}}`  
  Map from observable label (e.g., `"Y_lb"`, `"Y_tr"`, `"Y_bc"`, `"Y_ul"`) to a
  list of ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` time series. Each value is typically a 4-element vector
  `[TrM1, TrM2, TrM3, TrM4]`, where each `TrMi` is a `Vector{T}` over configurations.
  Labels like `"Y_info"` (metadata) should be ignored.

- `Q_moment::Dict{String, Vector{T}}`  
  Map from label to precomputed Q-moment time series (1D arrays over configurations).
  (Commonly keys are moment- or label-qualified, e.g., `"Q1_Y_lb"`, `"Q2_Y_lb"`, ...)

- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo)  
  Configuration counts per label/group (e.g., `N_lb`, `N_tr`, `N_bc`, `N_ul`);
  used for validation and iteration.

- `N_bs::Int`  
  Number of bootstrap replicates.

- [`rng_pool::SeedManager.RNGPool`](@ref Deborah.Sarah.SeedManager.RNGPool)  
  Pool of RNGs (e.g., `Random.Xoshiro`), typically one per thread.

- `opt_blk_size::Dict{Symbol, Int}`  
  Per-group block lengths, keyed by symbols such as `:lb`, `:tr`, `:bc`, `:ul`.
  The appropriate `blk_size` is selected based on each series' label.

- `method::String`  
  Block-bootstrap scheme (case-sensitive):
  - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB): resample disjoint blocks.
  - `"moving"`        — Moving Block Bootstrap (MBB): resample sliding windows of length `blk_size`.
  - `"circular"`      — Circular Block Bootstrap (CBB): like MBB, but windows wrap around (circular indexing).
  Any other value should raise an error.

- `jobid::Union{Nothing, String}` (optional)  
  Identifier for logging/debugging.

# Behavior
- For each label and each series under that label:
  1. Determine `blk_size = opt_blk_size[label_as_symbol]`.
  2. Draw `N_bs` bootstrap replicates using `method`.
  3. For each replicate, resample blocks with replacement until the original
     length is reached (truncate the final block if overshooting).
  4. Compute the sample mean of the resampled series and store it into
     `bootstrap_data[:mean][label][bs_idx][series_idx]`.
- The same procedure is applied to each series in `Q_moment` (with `series_idx`
  advancing accordingly for that label).

# Returns
- `Nothing` — modifies `bootstrap_data` in place.

# Notes
- Set `blk_size = 1` to recover the i.i.d. bootstrap.
- Ensure each series length is adequate relative to `blk_size` (especially for MBB/CBB).
- RNGs should be drawn from `rng_pool` to avoid cross-thread contention.
- Labels in `trace_data` / `Q_moment` are expected to be consistent with `partition`;
  mismatches should be logged or errored.
"""
function run_bootstrap!(
    bootstrap_data::Dict{Symbol, Any},
    trace_data::Dict{String, Vector{Vector{T}}},
    Q_moment::Dict{String, Vector{T}},
    partition::DatasetPartitioner.DatasetPartitionInfo,
    N_bs::Int,
    rng_pool::SeedManager.RNGPool,
    opt_blk_size::Dict{Symbol, Int},
    method::String,
    jobid::Union{Nothing, String}=nothing
) where T<:Real
    models_trM = ["trM1", "trM2", "trM3", "trM4"]
    models_Q   = ["Q1", "Q2", "Q3", "Q4"]

    w_lb = partition.N_lb / partition.N_cnf
    w_ul = partition.N_ul / partition.N_cnf

    N_all = partition.N_cnf
    N_lb  = partition.N_lb
    N_tr  = partition.N_tr
    N_bc  = partition.N_bc
    N_ul  = partition.N_ul

    # ---------- helpers ----------
    # Safe getters returning empty vectors if missing
    safeget_td(key, i) = (haskey(trace_data, key) && length(trace_data[key]) ≥ i) ? trace_data[key][i] : Float64[]
    safeget_Qv(key)    = haskey(Q_moment, key) ? Q_moment[key] : Float64[]

    # ---------- source arrays ----------
    # traces for trM1..4
    tr_Y_info = [safeget_td("Y_info", i) for i in 1:length(models_trM)]
    tr_Y_lb   = [safeget_td("Y_lb",   i) for i in 1:length(models_trM)]
    tr_Y_bc   = [safeget_td("Y_bc",   i) for i in 1:length(models_trM)]
    tr_Y_ul   = [safeget_td("Y_ul",   i) for i in 1:length(models_trM)]
    tr_YP_bc  = [safeget_td("YP_bc",  i) for i in 1:length(models_trM)]
    tr_YP_ul  = [safeget_td("YP_ul",  i) for i in 1:length(models_trM)]

    # Q1..Q4 traces (already moments-per-config)
    Q_Y_info = [safeget_Qv("$(m):Y_info") for m in models_Q]
    Q_Y_lb   = [safeget_Qv("$(m):Y_lb")   for m in models_Q]
    Q_Y_bc   = [safeget_Qv("$(m):Y_bc")   for m in models_Q]
    Q_Y_ul   = [safeget_Qv("$(m):Y_ul")   for m in models_Q]
    Q_YP_bc  = [safeget_Qv("$(m):YP_bc")  for m in models_Q]
    Q_YP_ul  = [safeget_Qv("$(m):YP_ul")  for m in models_Q]

    # ---------- outputs ----------
    mean_tr_Y_info = [bootstrap_data[:mean]["$(m):Y_info"] for m in models_trM]
    mean_tr_Y_lb   = [bootstrap_data[:mean]["$(m):Y_lb"]   for m in models_trM]
    mean_tr_Y_bc   = [bootstrap_data[:mean]["$(m):Y_bc"]   for m in models_trM]
    mean_tr_Y_ul   = [bootstrap_data[:mean]["$(m):Y_ul"]   for m in models_trM]
    mean_tr_YP_bc  = [bootstrap_data[:mean]["$(m):YP_bc"]  for m in models_trM]
    mean_tr_YP_ul  = [bootstrap_data[:mean]["$(m):YP_ul"]  for m in models_trM]

    mean_tr_YmYP   = [bootstrap_data[:mean]["$(m):YmYP"]   for m in models_trM]
    mean_tr_Y_P1   = [bootstrap_data[:mean]["$(m):Y_P1"]   for m in models_trM]
    mean_tr_Y_P2   = [bootstrap_data[:mean]["$(m):Y_P2"]   for m in models_trM]

    mean_Q_Y_info = [bootstrap_data[:mean]["$(m):Y_info"] for m in models_Q]
    mean_Q_Y_lb   = [bootstrap_data[:mean]["$(m):Y_lb"]   for m in models_Q]
    mean_Q_Y_bc   = [bootstrap_data[:mean]["$(m):Y_bc"]   for m in models_Q]
    mean_Q_Y_ul   = [bootstrap_data[:mean]["$(m):Y_ul"]   for m in models_Q]
    mean_Q_YP_bc  = [bootstrap_data[:mean]["$(m):YP_bc"]  for m in models_Q]
    mean_Q_YP_ul  = [bootstrap_data[:mean]["$(m):YP_ul"]  for m in models_Q]

    mean_Q_YmYP   = [bootstrap_data[:mean]["$(m):YmYP"]   for m in models_Q]
    mean_Q_Y_P1   = [bootstrap_data[:mean]["$(m):Y_P1"]   for m in models_Q]
    mean_Q_Y_P2   = [bootstrap_data[:mean]["$(m):Y_P2"]   for m in models_Q]

    # ---------- plans (reusable across κ) ----------
    starts_all, nblk_all, last_all = Bootstrap.ensure_plan(nothing, rng_pool.rng,    N_all, opt_blk_size[:all], N_bs; method=method)
    starts_lb,  nblk_lb,  last_lb  = Bootstrap.ensure_plan(nothing, rng_pool.rng_lb, N_lb,  opt_blk_size[:lb],  N_bs; method=method)
    starts_bc,  nblk_bc,  last_bc  = Bootstrap.ensure_plan(nothing, rng_pool.rng_bc, N_bc,  opt_blk_size[:bc],  N_bs; method=method)
    starts_ul,  nblk_ul,  last_ul  = Bootstrap.ensure_plan(nothing, rng_pool.rng_ul, N_ul,  opt_blk_size[:ul],  N_bs; method=method)

    # ---------- prefix sums for every series ----------
    # (build arrays-of-ps to match arrays-of-traces)
    ps_tr_Y_info = [ (N_all>0 ? Bootstrap.prefix_sums(v) : nothing) for v in tr_Y_info ]
    ps_tr_Y_lb   = [ (N_lb >0 ? Bootstrap.prefix_sums(v) : nothing) for v in tr_Y_lb ]
    ps_tr_Y_bc   = [ (N_bc >0 ? Bootstrap.prefix_sums(v) : nothing) for v in tr_Y_bc ]
    ps_tr_Y_ul   = [ (N_ul >0 ? Bootstrap.prefix_sums(v) : nothing) for v in tr_Y_ul ]
    ps_tr_YP_bc  = [ (N_bc >0 ? Bootstrap.prefix_sums(v) : nothing) for v in tr_YP_bc ]
    ps_tr_YP_ul  = [ (N_ul >0 ? Bootstrap.prefix_sums(v) : nothing) for v in tr_YP_ul ]

    ps_Q_Y_info = [ (N_all>0 ? Bootstrap.prefix_sums(v) : nothing) for v in Q_Y_info ]
    ps_Q_Y_lb   = [ (N_lb >0 ? Bootstrap.prefix_sums(v) : nothing) for v in Q_Y_lb ]
    ps_Q_Y_bc   = [ (N_bc >0 ? Bootstrap.prefix_sums(v) : nothing) for v in Q_Y_bc ]
    ps_Q_Y_ul   = [ (N_ul >0 ? Bootstrap.prefix_sums(v) : nothing) for v in Q_Y_ul ]
    ps_Q_YP_bc  = [ (N_bc >0 ? Bootstrap.prefix_sums(v) : nothing) for v in Q_YP_bc ]
    ps_Q_YP_ul  = [ (N_ul >0 ? Bootstrap.prefix_sums(v) : nothing) for v in Q_YP_ul ]

    # ---------- main loop ----------
    JobLoggerTools.log_stage_sub1_benji("Bootstrap resampling loop ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        @inbounds for ibs in 1:N_bs
            # trM* means
            for k in eachindex(tr_Y_info)
                mean_tr_Y_info[k][ibs] = Bootstrap.mean_from_plan(ps_tr_Y_info[k], tr_Y_info[k], starts_all, N_all, opt_blk_size[:all], nblk_all, last_all, ibs; method=method)
                mean_tr_Y_lb[k][ibs]   = Bootstrap.mean_from_plan(ps_tr_Y_lb[k],   tr_Y_lb[k],   starts_lb,  N_lb,  opt_blk_size[:lb],  nblk_lb,  last_lb,  ibs; method=method)

                if N_tr>0 && N_bc>0
                    mean_tr_Y_bc[k][ibs]  = Bootstrap.mean_from_plan(ps_tr_Y_bc[k],  tr_Y_bc[k],  starts_bc,  N_bc,  opt_blk_size[:bc],  nblk_bc,  last_bc,  ibs; method=method)
                    mean_tr_YP_bc[k][ibs] = Bootstrap.mean_from_plan(ps_tr_YP_bc[k], tr_YP_bc[k], starts_bc,  N_bc,  opt_blk_size[:bc],  nblk_bc,  last_bc,  ibs; method=method)
                else
                    mean_tr_Y_bc[k][ibs]  = 0.0
                    mean_tr_YP_bc[k][ibs] = 0.0
                end

                if N_tr>0 && N_ul>0
                    mean_tr_Y_ul[k][ibs]  = Bootstrap.mean_from_plan(ps_tr_Y_ul[k],  tr_Y_ul[k],  starts_ul,  N_ul,  opt_blk_size[:ul],  nblk_ul,  last_ul,  ibs; method=method)
                    mean_tr_YP_ul[k][ibs] = Bootstrap.mean_from_plan(ps_tr_YP_ul[k], tr_YP_ul[k], starts_ul,  N_ul,  opt_blk_size[:ul],  nblk_ul,  last_ul,  ibs; method=method)
                else
                    mean_tr_Y_ul[k][ibs]  = 0.0
                    mean_tr_YP_ul[k][ibs] = 0.0
                end
            end

            # Q* means
            for k in eachindex(Q_Y_info)
                mean_Q_Y_info[k][ibs] = Bootstrap.mean_from_plan(ps_Q_Y_info[k], Q_Y_info[k], starts_all, N_all, opt_blk_size[:all], nblk_all, last_all, ibs; method=method)
                mean_Q_Y_lb[k][ibs]   = Bootstrap.mean_from_plan(ps_Q_Y_lb[k],   Q_Y_lb[k],   starts_lb,  N_lb,  opt_blk_size[:lb],  nblk_lb,  last_lb,  ibs; method=method)

                if N_tr>0 && N_bc>0
                    mean_Q_Y_bc[k][ibs]  = Bootstrap.mean_from_plan(ps_Q_Y_bc[k],  Q_Y_bc[k],  starts_bc,  N_bc,  opt_blk_size[:bc],  nblk_bc,  last_bc,  ibs; method=method)
                    mean_Q_YP_bc[k][ibs] = Bootstrap.mean_from_plan(ps_Q_YP_bc[k], Q_YP_bc[k], starts_bc,  N_bc,  opt_blk_size[:bc],  nblk_bc,  last_bc,  ibs; method=method)
                else
                    mean_Q_Y_bc[k][ibs]  = 0.0
                    mean_Q_YP_bc[k][ibs] = 0.0
                end

                if N_tr>0 && N_ul>0
                    mean_Q_Y_ul[k][ibs]  = Bootstrap.mean_from_plan(ps_Q_Y_ul[k],  Q_Y_ul[k],  starts_ul,  N_ul,  opt_blk_size[:ul],  nblk_ul,  last_ul,  ibs; method=method)
                    mean_Q_YP_ul[k][ibs] = Bootstrap.mean_from_plan(ps_Q_YP_ul[k], Q_YP_ul[k], starts_ul,  N_ul,  opt_blk_size[:ul],  nblk_ul,  last_ul,  ibs; method=method)
                else
                    mean_Q_Y_ul[k][ibs]  = 0.0
                    mean_Q_YP_ul[k][ibs] = 0.0
                end
            end

            # combine (P1/P2)
            if N_tr == 0
                for k in eachindex(models_trM)
                    mean_tr_YmYP[k][ibs] = 0.0
                    mean_tr_Y_P1[k][ibs] = mean_tr_Y_lb[k][ibs]
                    mean_tr_Y_P2[k][ibs] = mean_tr_Y_lb[k][ibs]
                end
                for k in eachindex(models_Q)
                    mean_Q_YmYP[k][ibs] = 0.0
                    mean_Q_Y_P1[k][ibs] = mean_Q_Y_lb[k][ibs]
                    mean_Q_Y_P2[k][ibs] = mean_Q_Y_lb[k][ibs]
                end
            else
                for k in eachindex(models_trM)
                    mean_tr_YmYP[k][ibs] = mean_tr_Y_bc[k][ibs]  - mean_tr_YP_bc[k][ibs]
                    mean_tr_Y_P1[k][ibs] = mean_tr_YP_ul[k][ibs] + mean_tr_YmYP[k][ibs]
                    mean_tr_Y_P2[k][ibs] = w_lb * mean_tr_Y_lb[k][ibs] + w_ul * mean_tr_Y_P1[k][ibs]
                end
                for k in eachindex(models_Q)
                    mean_Q_YmYP[k][ibs] = mean_Q_Y_bc[k][ibs]  - mean_Q_YP_bc[k][ibs]
                    mean_Q_Y_P1[k][ibs] = mean_Q_YP_ul[k][ibs] + mean_Q_YmYP[k][ibs]
                    mean_Q_Y_P2[k][ibs] = w_lb * mean_Q_Y_lb[k][ibs] + w_ul * mean_Q_Y_P1[k][ibs]
                end
            end
        end
    end
end

end  # module BootstrapRunner