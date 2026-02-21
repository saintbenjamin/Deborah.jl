# =============================================================================
# src/Miriam/CumulantsBundle/compute_traces_bundle_raw.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# =============================================================================

"""
    compute_traces_bundle_raw(
        N_bs::Int,
        blk_size::Int,
        method::String,
        trMi_all_bundle::Vector{Vector{Vector{T}}},
        ens_bundle::Ensemble.EnsembleArrayBundle{T},
        ens_idx::Int,
        jobid::Union{Nothing, String}=nothing;
        rng_pool::Union{Nothing, SeedManager.RNGPool} = nothing,
        idx_all::Union{Nothing, Matrix{Int}} = nothing,
        idx_lb ::Union{Nothing, Matrix{Int}} = nothing,
        idx_bc ::Union{Nothing, Matrix{Int}} = nothing,
        idx_ul ::Union{Nothing, Matrix{Int}} = nothing
    ) where T -> (
        Vector{Vector{Float64}},
        Vector{Vector{Float64}},
        Vector{Vector{Float64}},
        Dict{Symbol, Union{Nothing, Matrix{Int}}}
    )

Compute bootstrap estimates of ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` for single ensemble. 
This mirrors [`compute_moments_bundle_raw`](@ref) but returns trace components instead of moments.

# Arguments
- `N_bs::Int`: Number of bootstrap resamples.
- `blk_size::Int`: Base block size for block bootstrap (per subset will be optimized).
- `method::String`: Bootstrap method identifier (e.g., `"moving"`, `"circular"`, `"nonoverlapping"`).
- `trMi_all_bundle::Vector{Vector{Vector{T}}}`: For each array in the bundle, a vector of
  per-configuration trace-moment rows (e.g., length-5 `trMi`). Each inner `Vector{T}`
  is processed by [`Deborah.Miriam.Cumulants.calc_trace`](@ref) to obtain ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Bundle containing the ensemble arrays.
- `ens_idx::Int`: Index of the target ensemble array inside `ens_bundle`.
- `jobid`: Optional job identifier for logging.
- [`rng_pool::Union{Nothing, SeedManager.RNGPool}`](@ref Deborah.Sarah.SeedManager.RNGPool): Optional RNG pool (with fields `rng`, `rng_lb`, `rng_bc`, `rng_ul`);
  if `nothing`, a new pool is created via [`Deborah.Sarah.SeedManager.setup_rng_pool`](@ref).
- `idx_all`, `idx_lb`, `idx_bc`, `idx_ul`: Optional bootstrap index plans for the subsets.
  If provided, they are used; otherwise, plans are generated internally.

# Returns
- `(traces_OG_bs, traces_P1_bs, traces_P2_bs, idx_bundle)` where:
  - `traces_OG_bs::Vector{Vector{Float64}}` = `[trM1_ORG_bs, trM2_ORG_bs, trM3_ORG_bs, trM4_ORG_bs]`
  - `traces_P1_bs::Vector{Vector{Float64}}` = `[trM1_P1_bs,  trM2_P1_bs,  trM3_P1_bs,  trM4_P1_bs ]`
  - `traces_P2_bs::Vector{Vector{Float64}}` = `[trM1_P2_bs,  trM2_P2_bs,  trM3_P2_bs,  trM4_P2_bs ]`
    Each `trMk_*_bs` is a length-`N_bs` vector of bootstrap means.
  - `idx_bundle::Dict{Symbol, Union{Nothing, Matrix{Int}}}`:
    `:all`, `:lb`, `:bc`, `:ul` → the index plans used for each subset (or `nothing`).

# Notes
- Subset splitting follows the same tag logic as the cumulant/moment pipeline
  (`ORG`/`Y_LB`/`Y_BC`/`YP_BC`/`Y_UL`/`YP_UL`).
- `P1` is constructed component-wise:
  `Y_P1 = (N_tr == N_lb) ? YP_UL : (YP_UL + (Y_BC - YP_BC))`.
- `P2` is the linear combination:
  `Y_P2 = w_lb * Y_LB + w_ul * Y_P1`, with `w_lb = N_lb/N_all` and `w_ul = N_ul/N_all`.
- If a subset is empty by construction (e.g., `N_lb == 0`), its bootstrap means are set to `0`.
"""
function compute_traces_bundle_raw(
    N_bs::Int,
    blk_size::Int,
    method::String,
    trMi_all_bundle::Vector{Vector{Vector{T}}},
    ens_bundle::Ensemble.EnsembleArrayBundle{T},
    ens_idx::Int, 
    jobid::Union{Nothing, String}=nothing;
    rng_pool::Union{Nothing, SeedManager.RNGPool} = nothing,
    idx_all::Union{Nothing, Matrix{Int}} = nothing,
    idx_lb ::Union{Nothing, Matrix{Int}} = nothing,
    idx_bc ::Union{Nothing, Matrix{Int}} = nothing,
    idx_ul ::Union{Nothing, Matrix{Int}} = nothing
) where T

    # RNG pool
    rng_pool = isnothing(rng_pool) ? SeedManager.setup_rng_pool() : rng_pool

    # === Obtain trM1..trM4 for each array in the bundle (no reweighting factor) ===
    trMi_bundle_raw = Vector{NTuple{4, Vector{T}}}()

    for i in eachindex(ens_bundle.arrays)
        obs   = trMi_all_bundle[i]
        nconf = length(obs)

        trM1 = Vector{T}(undef, nconf)
        trM2 = Vector{T}(undef, nconf)
        trM3 = Vector{T}(undef, nconf)
        trM4 = Vector{T}(undef, nconf)

        for j in 1:nconf
            trM1[j], trM2[j], trM3[j], trM4[j] = Cumulants.calc_trace(obs[j])
        end

        push!(trMi_bundle_raw, (trM1, trM2, trM3, trM4))
    end

    # === Split into subsets; also get "ORG" and convenience groupings ===
    trM_Y_tr, trM_Y_bc, trM_YP_bc, trM_Y_ul, trM_YP_ul, trM_Y_lb, trM_Y_ORG = CumulantsBundleUtils.split_Q_full(nothing, trMi_bundle_raw, ens_bundle, ens_idx, jobid; has_weight=false)

    # Sizes and weights
    N_tr  = length(trM_Y_tr)
    N_bc  = length(trM_Y_bc)
    N_ul  = length(trM_Y_ul)
    N_lb  = N_tr + N_bc
    N_all = N_lb + N_ul

    # Normalization weights
    w_lb = N_lb / N_all
    w_ul = N_ul / N_all

    # === Suggest per-subset block sizes ===
    opt_blk_size = BlockSizeSuggester.suggest_opt_block_sizes(
        N_all, N_lb, N_bc, N_ul, blk_size
    )

    # === Flatten columns to Arrays for fast prefix-sum access ===
    trM1_Y_ORG, trM2_Y_ORG, trM3_Y_ORG, trM4_Y_ORG = CumulantsBundleUtils.flatten_Q4_columns(trM_Y_ORG)
    trM1_Y_lb,  trM2_Y_lb,  trM3_Y_lb,  trM4_Y_lb  = CumulantsBundleUtils.flatten_Q4_columns(trM_Y_lb)
    trM1_Y_bc,  trM2_Y_bc,  trM3_Y_bc,  trM4_Y_bc  = CumulantsBundleUtils.flatten_Q4_columns(trM_Y_bc)
    trM1_YP_bc, trM2_YP_bc, trM3_YP_bc, trM4_YP_bc = CumulantsBundleUtils.flatten_Q4_columns(trM_YP_bc)
    trM1_Y_ul,  trM2_Y_ul,  trM3_Y_ul,  trM4_Y_ul  = CumulantsBundleUtils.flatten_Q4_columns(trM_Y_ul)
    trM1_YP_ul, trM2_YP_ul, trM3_YP_ul, trM4_YP_ul = CumulantsBundleUtils.flatten_Q4_columns(trM_YP_ul)

    # === Allocate output arrays for bootstrap means per subset ===
    trM_mean_Y_ORG  = Matrix{Float64}(undef, 4, N_bs)
    trM_mean_Y_lb   = Matrix{Float64}(undef, 4, N_bs)
    trM_mean_Y_bc   = Matrix{Float64}(undef, 4, N_bs)
    trM_mean_YP_bc  = Matrix{Float64}(undef, 4, N_bs)
    trM_mean_Y_ul   = Matrix{Float64}(undef, 4, N_bs)
    trM_mean_YP_ul  = Matrix{Float64}(undef, 4, N_bs)

    # === Ensure or create bootstrap plans ===
    starts_all, nblk_all, last_all = Bootstrap.ensure_plan(
        (idx_all isa AbstractMatrix{<:Integer}) ? idx_all : nothing,
        rng_pool.rng,  N_all, opt_blk_size[:all], N_bs; method=method)

    starts_lb,  nblk_lb,  last_lb  = Bootstrap.ensure_plan(
        (idx_lb  isa AbstractMatrix{<:Integer}) ? idx_lb  : nothing,
        rng_pool.rng_lb,  N_lb,  opt_blk_size[:lb],  N_bs; method=method)

    starts_bc,  nblk_bc,  last_bc  = Bootstrap.ensure_plan(
        (idx_bc  isa AbstractMatrix{<:Integer}) ? idx_bc  : nothing,
        rng_pool.rng_bc,  N_bc,  opt_blk_size[:bc],  N_bs; method=method)

    starts_ul,  nblk_ul,  last_ul  = Bootstrap.ensure_plan(
        (idx_ul  isa AbstractMatrix{<:Integer}) ? idx_ul  : nothing,
        rng_pool.rng_ul,  N_ul,  opt_blk_size[:ul],  N_bs; method=method)

    # =============================
    # Main bootstrap resampling loop
    # =============================
    JobLoggerTools.@logtime_benji jobid begin
        ps_Y_ORG = (Bootstrap.prefix_sums(trM1_Y_ORG), Bootstrap.prefix_sums(trM2_Y_ORG),
                    Bootstrap.prefix_sums(trM3_Y_ORG), Bootstrap.prefix_sums(trM4_Y_ORG))
        ps_Y_lb  = (Bootstrap.prefix_sums(trM1_Y_lb),  Bootstrap.prefix_sums(trM2_Y_lb),
                    Bootstrap.prefix_sums(trM3_Y_lb),  Bootstrap.prefix_sums(trM4_Y_lb))
        ps_Y_bc  = (Bootstrap.prefix_sums(trM1_Y_bc),  Bootstrap.prefix_sums(trM2_Y_bc),
                    Bootstrap.prefix_sums(trM3_Y_bc),  Bootstrap.prefix_sums(trM4_Y_bc))
        ps_YP_bc = (Bootstrap.prefix_sums(trM1_YP_bc), Bootstrap.prefix_sums(trM2_YP_bc),
                    Bootstrap.prefix_sums(trM3_YP_bc), Bootstrap.prefix_sums(trM4_YP_bc))
        ps_Y_ul  = (Bootstrap.prefix_sums(trM1_Y_ul),  Bootstrap.prefix_sums(trM2_Y_ul),
                    Bootstrap.prefix_sums(trM3_Y_ul),  Bootstrap.prefix_sums(trM4_Y_ul))
        ps_YP_ul = (Bootstrap.prefix_sums(trM1_YP_ul), Bootstrap.prefix_sums(trM2_YP_ul),
                    Bootstrap.prefix_sums(trM3_YP_ul), Bootstrap.prefix_sums(trM4_YP_ul))

        @inbounds for ibs in 1:N_bs
            # ORG over ALL
            Bootstrap.update_mean_from_plan4!((trM1_Y_ORG, trM2_Y_ORG, trM3_Y_ORG, trM4_Y_ORG),
                                    ps_Y_ORG, starts_all, N_all, opt_blk_size[:all], nblk_all, last_all,
                                    trM_mean_Y_ORG, ibs; method=method)

            # LB
            if N_lb > 0
                Bootstrap.update_mean_from_plan4!((trM1_Y_lb, trM2_Y_lb, trM3_Y_lb, trM4_Y_lb),
                                        ps_Y_lb, starts_lb, N_lb, opt_blk_size[:lb], nblk_lb, last_lb,
                                        trM_mean_Y_lb, ibs; method=method)
            else
                @inbounds trM_mean_Y_lb[:, ibs] .= 0.0
            end

            # BC & YP_BC (only when both BC and TR exist)
            if N_bc > 0 && N_tr > 0
                Bootstrap.update_mean_from_plan4!((trM1_Y_bc, trM2_Y_bc, trM3_Y_bc, trM4_Y_bc),
                                        ps_Y_bc, starts_bc, N_bc, opt_blk_size[:bc], nblk_bc, last_bc,
                                        trM_mean_Y_bc, ibs; method=method)
                Bootstrap.update_mean_from_plan4!((trM1_YP_bc, trM2_YP_bc, trM3_YP_bc, trM4_YP_bc),
                                        ps_YP_bc, starts_bc, N_bc, opt_blk_size[:bc], nblk_bc, last_bc,
                                        trM_mean_YP_bc, ibs; method=method)
            else
                @inbounds trM_mean_Y_bc[:,  ibs] .= 0.0
                @inbounds trM_mean_YP_bc[:, ibs] .= 0.0
            end

            # UL & YP_UL (gated on TR as in cumulant pipeline)
            if N_tr > 0
                Bootstrap.update_mean_from_plan4!((trM1_Y_ul, trM2_Y_ul, trM3_Y_ul, trM4_Y_ul),
                                        ps_Y_ul, starts_ul, N_ul, opt_blk_size[:ul], nblk_ul, last_ul,
                                        trM_mean_Y_ul, ibs; method=method)
                Bootstrap.update_mean_from_plan4!((trM1_YP_ul, trM2_YP_ul, trM3_YP_ul, trM4_YP_ul),
                                        ps_YP_ul, starts_ul, N_ul, opt_blk_size[:ul], nblk_ul, last_ul,
                                        trM_mean_YP_ul, ibs; method=method)
            else
                @inbounds trM_mean_Y_ul[:,  ibs] .= 0.0
                @inbounds trM_mean_YP_ul[:, ibs] .= 0.0
            end
        end
    end; flush(stdout); flush(stderr)

    # === Build ORG / P1 / P2 moment sets per bootstrap sample ===
    trM1_ORG_bs = Vector{Float64}(undef, N_bs)
    trM2_ORG_bs = Vector{Float64}(undef, N_bs)
    trM3_ORG_bs = Vector{Float64}(undef, N_bs)
    trM4_ORG_bs = Vector{Float64}(undef, N_bs)

    trM1_P1_bs  = Vector{Float64}(undef, N_bs)
    trM2_P1_bs  = Vector{Float64}(undef, N_bs)
    trM3_P1_bs  = Vector{Float64}(undef, N_bs)
    trM4_P1_bs  = Vector{Float64}(undef, N_bs)

    trM1_P2_bs  = Vector{Float64}(undef, N_bs)
    trM2_P2_bs  = Vector{Float64}(undef, N_bs)
    trM3_P2_bs  = Vector{Float64}(undef, N_bs)
    trM4_P2_bs  = Vector{Float64}(undef, N_bs)

    @inbounds for i in 1:N_bs
        # ORG = means over ALL
        trM1_ORG = trM_mean_Y_ORG[1, i]
        trM2_ORG = trM_mean_Y_ORG[2, i]
        trM3_ORG = trM_mean_Y_ORG[3, i]
        trM4_ORG = trM_mean_Y_ORG[4, i]

        trM1_ORG_bs[i] = trM1_ORG
        trM2_ORG_bs[i] = trM2_ORG
        trM3_ORG_bs[i] = trM3_ORG
        trM4_ORG_bs[i] = trM4_ORG

        # P1 = YP_UL (+ (Y_BC - YP_BC) if N_tr < N_lb)
        if N_tr == 0
            # No training: define P1 = LB
            trM1_P1 = trM_mean_Y_lb[1, i]
            trM2_P1 = trM_mean_Y_lb[2, i]
            trM3_P1 = trM_mean_Y_lb[3, i]
            trM4_P1 = trM_mean_Y_lb[4, i]
        else
            ΔtrM1 = trM_mean_Y_bc[1, i] - trM_mean_YP_bc[1, i]
            ΔtrM2 = trM_mean_Y_bc[2, i] - trM_mean_YP_bc[2, i]
            ΔtrM3 = trM_mean_Y_bc[3, i] - trM_mean_YP_bc[3, i]
            ΔtrM4 = trM_mean_Y_bc[4, i] - trM_mean_YP_bc[4, i]

            if N_tr == N_lb
                # Full pass-through
                trM1_P1 = trM_mean_YP_ul[1, i]
                trM2_P1 = trM_mean_YP_ul[2, i]
                trM3_P1 = trM_mean_YP_ul[3, i]
                trM4_P1 = trM_mean_YP_ul[4, i]
            else
                trM1_P1 = trM_mean_YP_ul[1, i] + ΔtrM1
                trM2_P1 = trM_mean_YP_ul[2, i] + ΔtrM2
                trM3_P1 = trM_mean_YP_ul[3, i] + ΔtrM3
                trM4_P1 = trM_mean_YP_ul[4, i] + ΔtrM4
            end
        end

        trM1_P1_bs[i] = trM1_P1
        trM2_P1_bs[i] = trM2_P1
        trM3_P1_bs[i] = trM3_P1
        trM4_P1_bs[i] = trM4_P1

        # P2 = w_lb * LB + w_ul * P1
        trM1_P2 = w_lb * trM_mean_Y_lb[1, i] + w_ul * trM1_P1
        trM2_P2 = w_lb * trM_mean_Y_lb[2, i] + w_ul * trM2_P1
        trM3_P2 = w_lb * trM_mean_Y_lb[3, i] + w_ul * trM3_P1
        trM4_P2 = w_lb * trM_mean_Y_lb[4, i] + w_ul * trM4_P1

        trM1_P2_bs[i] = trM1_P2
        trM2_P2_bs[i] = trM2_P2
        trM3_P2_bs[i] = trM3_P2
        trM4_P2_bs[i] = trM4_P2
    end

    # === Pack results ===
    traces_OG_bs = [trM1_ORG_bs, trM2_ORG_bs, trM3_ORG_bs, trM4_ORG_bs]
    traces_P1_bs = [trM1_P1_bs,  trM2_P1_bs,  trM3_P1_bs,  trM4_P1_bs ]
    traces_P2_bs = [trM1_P2_bs,  trM2_P2_bs,  trM3_P2_bs,  trM4_P2_bs ]

    idx_bundle = Dict(
        :all => idx_all,
        :lb  => idx_lb,
        :bc  => idx_bc,
        :ul  => idx_ul
    )

    return traces_OG_bs, traces_P1_bs, traces_P2_bs, idx_bundle
end