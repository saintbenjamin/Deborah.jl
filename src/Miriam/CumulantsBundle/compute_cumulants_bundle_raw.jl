# =============================================================================
# src/Miriam/CumulantsBundle/compute_cumulants_bundle_raw.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# =============================================================================

"""
    compute_cumulants_bundle_raw(
        N_bs::Int,
        blk_size::Int,
        method::String,
        V::Int,
        trMi_all_bundle::Vector{Vector{Vector{T}}},
        ens_bundle::Ensemble.EnsembleArrayBundle{T},
        ens_idx::Int, 
        jobid::Union{Nothing, String}=nothing;
        rng_pool::Union{Nothing, SeedManager.RNGPool} = nothing,
        idx_all::Union{Nothing, Vector{Vector{Int}}} = nothing,
        idx_lb::Union{Nothing, Vector{Vector{Int}}} = nothing,
        idx_bc::Union{Nothing, Vector{Vector{Int}}} = nothing,
        idx_ul::Union{Nothing, Vector{Vector{Int}}} = nothing
    ) where T -> (
        Vector{Vector{Float64}},
        Vector{Vector{Float64}},
        Vector{Vector{Float64}},
        Dict{Symbol, Vector{Vector{Int}}}
    )

Compute bootstrap estimates of cumulants for single ensemble.

This function performs block bootstrap resampling and computes cumulants from structured subsets
of trace data -- original, labeled, bias-corrected, and unlabeled -- extracted from a bundle of ensembles.

# Arguments
- `N_bs::Int`: Number of bootstrap resamples.
- `blk_size::Int`: Default block size for bootstrap resampling.
- `method::String`   : Block-bootstrap scheme (case-sensitive):
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB; resample disjoint blocks).
    - `"moving"`        — Moving Block Bootstrap (MBB; resample sliding windows).
    - `"circular"`      — Circular Block Bootstrap (CBB; sliding windows with wrap-around).
- `V::Int`: Lattice volume (e.g., ``N_S^3 \\times N_T``).
- `trMi_all_bundle::Vector{Vector{Vector{T}}}`: Trace moment inputs; `trMi_all_bundle[i][j]` holds the `j`-th configuration of the `i`-th estimator.
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Bundle of ensemble data matching the structure of `trMi_all_bundle`.
- `ens_idx::Int`: Index of the ensemble to process (same across all estimators).
- `jobid::Union{Nothing, String}`: Optional logging context.
- [`rng_pool::Union{Nothing, SeedManager.RNGPool}`](@ref Deborah.Sarah.SeedManager.RNGPool): Optional RNG pool for reproducible sampling.
- `idx_all`: Optional bootstrap indices for original data.
- `idx_lb`: Optional bootstrap indices for labeled set.
- `idx_bc`: Optional bootstrap indices for bias correction set.
- `idx_ul`: Optional bootstrap indices for unlabeled set.

# Returns
- `cumulants_OG_bs::Vector{Vector{Float64}}`: Bootstrap samples of cumulants from the original dataset.
- `cumulants_P1_bs::Vector{Vector{Float64}}`: Bootstrap samples using the `P1` estimator (bias-corrected).
- `cumulants_P2_bs::Vector{Vector{Float64}}`: Bootstrap samples using the `P2` estimator (weighted `LB` + `P1`).
- `idx_bundle::Dict{Symbol, Vector{Vector{Int}}}`: Bundle of bootstrap indices actually used for resampling.

# Notes
- If index arguments are omitted, new bootstrap indices are generated and returned via `idx_bundle`.
"""
function compute_cumulants_bundle_raw(
    N_bs::Int,
    blk_size::Int,
    method::String,
    V::Int,
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

    rng_pool = isnothing(rng_pool) ? SeedManager.setup_rng_pool() : rng_pool

    # Compute and store Q1–Q4 observables and weights from solvers
    Q_bundle_raw = Vector{NTuple{4, Vector{T}}}()

    for i in eachindex(ens_bundle.arrays)
        obs = trMi_all_bundle[i]
        nconf = length(obs)

        Q1 = Vector{T}(undef, nconf)
        Q2 = Vector{T}(undef, nconf)
        Q3 = Vector{T}(undef, nconf)
        Q4 = Vector{T}(undef, nconf)

        for j in 1:nconf
            Q1[j], Q2[j], Q3[j], Q4[j] = Cumulants.calc_Q(obs[j])
        end

        push!(Q_bundle_raw, (Q1, Q2, Q3, Q4))
    end

    # Split observables into source categories
    Q_Y_tr, Q_Y_bc, Q_YP_bc, Q_Y_ul, Q_YP_ul, Q_Y_lb, Q_Y_ORG = CumulantsBundleUtils.split_Q_full(nothing, Q_bundle_raw, ens_bundle, ens_idx, jobid; has_weight=false)

    # Count configuration sizes per category
    N_tr  = length(Q_Y_tr)
    N_bc  = length(Q_Y_bc)
    N_ul  = length(Q_Y_ul)
    N_lb  = N_tr + N_bc
    N_all = N_lb + N_ul

    # Normalization weights
    w_lb = N_lb / N_all
    w_ul = N_ul / N_all

    opt_blk_size = BlockSizeSuggester.suggest_opt_block_sizes(
        N_all, N_lb, N_bc, N_ul, blk_size
    )

    # Flatten tuple-of-arrays into arrays for efficient access
    Q1_Y_ORG, Q2_Y_ORG, Q3_Y_ORG, Q4_Y_ORG = CumulantsBundleUtils.flatten_Q4_columns(Q_Y_ORG)
    Q1_Y_lb,  Q2_Y_lb,  Q3_Y_lb,  Q4_Y_lb  = CumulantsBundleUtils.flatten_Q4_columns(Q_Y_lb)
    Q1_Y_bc,  Q2_Y_bc,  Q3_Y_bc,  Q4_Y_bc  = CumulantsBundleUtils.flatten_Q4_columns(Q_Y_bc)
    Q1_YP_bc, Q2_YP_bc, Q3_YP_bc, Q4_YP_bc = CumulantsBundleUtils.flatten_Q4_columns(Q_YP_bc)
    Q1_Y_ul,  Q2_Y_ul,  Q3_Y_ul,  Q4_Y_ul  = CumulantsBundleUtils.flatten_Q4_columns(Q_Y_ul)
    Q1_YP_ul, Q2_YP_ul, Q3_YP_ul, Q4_YP_ul = CumulantsBundleUtils.flatten_Q4_columns(Q_YP_ul)

    # Allocate output arrays for bootstrap means
    Q_mean_Y_ORG  = Matrix{Float64}(undef, 4, N_bs)
    Q_mean_Y_lb   = Matrix{Float64}(undef, 4, N_bs)
    Q_mean_Y_bc   = Matrix{Float64}(undef, 4, N_bs)
    Q_mean_YP_bc  = Matrix{Float64}(undef, 4, N_bs)
    Q_mean_Y_ul   = Matrix{Float64}(undef, 4, N_bs)
    Q_mean_YP_ul  = Matrix{Float64}(undef, 4, N_bs)

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

        ps_Y_ORG = (Bootstrap.prefix_sums(Q1_Y_ORG), Bootstrap.prefix_sums(Q2_Y_ORG),
                    Bootstrap.prefix_sums(Q3_Y_ORG), Bootstrap.prefix_sums(Q4_Y_ORG))
        ps_Y_lb  = (Bootstrap.prefix_sums(Q1_Y_lb),  Bootstrap.prefix_sums(Q2_Y_lb),
                    Bootstrap.prefix_sums(Q3_Y_lb),  Bootstrap.prefix_sums(Q4_Y_lb))
        ps_Y_bc  = (Bootstrap.prefix_sums(Q1_Y_bc),  Bootstrap.prefix_sums(Q2_Y_bc),
                    Bootstrap.prefix_sums(Q3_Y_bc),  Bootstrap.prefix_sums(Q4_Y_bc))
        ps_YP_bc = (Bootstrap.prefix_sums(Q1_YP_bc), Bootstrap.prefix_sums(Q2_YP_bc),
                    Bootstrap.prefix_sums(Q3_YP_bc), Bootstrap.prefix_sums(Q4_YP_bc))
        ps_Y_ul  = (Bootstrap.prefix_sums(Q1_Y_ul),  Bootstrap.prefix_sums(Q2_Y_ul),
                    Bootstrap.prefix_sums(Q3_Y_ul),  Bootstrap.prefix_sums(Q4_Y_ul))
        ps_YP_ul = (Bootstrap.prefix_sums(Q1_YP_ul), Bootstrap.prefix_sums(Q2_YP_ul),
                    Bootstrap.prefix_sums(Q3_YP_ul), Bootstrap.prefix_sums(Q4_YP_ul))

        @inbounds for ibs in 1:N_bs
            Bootstrap.update_mean_from_plan4!((Q1_Y_ORG, Q2_Y_ORG, Q3_Y_ORG, Q4_Y_ORG),
                                    ps_Y_ORG, starts_all, N_all, opt_blk_size[:all], nblk_all, last_all,
                                    Q_mean_Y_ORG, ibs; method=method)
            if N_lb > 0
                Bootstrap.update_mean_from_plan4!((Q1_Y_lb, Q2_Y_lb, Q3_Y_lb, Q4_Y_lb),
                                        ps_Y_lb, starts_lb, N_lb, opt_blk_size[:lb], nblk_lb, last_lb,
                                        Q_mean_Y_lb, ibs; method=method)
            else
                @inbounds Q_mean_Y_lb[:,ibs] .= 0.0
            end

            if N_bc > 0 && N_tr > 0
                Bootstrap.update_mean_from_plan4!((Q1_Y_bc, Q2_Y_bc, Q3_Y_bc, Q4_Y_bc),
                                        ps_Y_bc, starts_bc, N_bc, opt_blk_size[:bc], nblk_bc, last_bc,
                                        Q_mean_Y_bc, ibs; method=method)
                Bootstrap.update_mean_from_plan4!((Q1_YP_bc, Q2_YP_bc, Q3_YP_bc, Q4_YP_bc),
                                        ps_YP_bc, starts_bc, N_bc, opt_blk_size[:bc], nblk_bc, last_bc,
                                        Q_mean_YP_bc, ibs; method=method)
            else
                @inbounds Q_mean_Y_bc[:,ibs]  .= 0.0
                @inbounds Q_mean_YP_bc[:,ibs] .= 0.0
            end

            if N_tr > 0
                Bootstrap.update_mean_from_plan4!((Q1_Y_ul, Q2_Y_ul, Q3_Y_ul, Q4_Y_ul),
                                        ps_Y_ul, starts_ul, N_ul, opt_blk_size[:ul], nblk_ul, last_ul,
                                        Q_mean_Y_ul, ibs; method=method)
                Bootstrap.update_mean_from_plan4!((Q1_YP_ul, Q2_YP_ul, Q3_YP_ul, Q4_YP_ul),
                                        ps_YP_ul, starts_ul, N_ul, opt_blk_size[:ul], nblk_ul, last_ul,
                                        Q_mean_YP_ul, ibs; method=method)
            else
                @inbounds Q_mean_Y_ul[:,ibs]  .= 0.0
                @inbounds Q_mean_YP_ul[:,ibs] .= 0.0
            end
        end

    end; flush(stdout); flush(stderr)

    # === Compute ORG cumulants using bootstrap samples ===
    cond_ORG_bs = Vector{Float64}(undef, N_bs)
    susp_ORG_bs = Vector{Float64}(undef, N_bs)
    skew_ORG_bs = Vector{Float64}(undef, N_bs)
    kurt_ORG_bs = Vector{Float64}(undef, N_bs)
    bind_ORG_bs = Vector{Float64}(undef, N_bs)

    @inbounds for i in 1:N_bs
        Q1 = Q_mean_Y_ORG[1, i]
        Q2 = Q_mean_Y_ORG[2, i]
        Q3 = Q_mean_Y_ORG[3, i]
        Q4 = Q_mean_Y_ORG[4, i]

        # Compute susceptibility, skewness, kurtosis, Binder cumulant
        susp_tmp, skew_tmp, kurt_tmp, bind_tmp = Cumulants.calc_cumulants(
            float(V), 
            Q1, 
            Q2, 
            Q3, 
            Q4,
            jobid
        )

        cond_ORG_bs[i] = Q1 / V
        susp_ORG_bs[i] = susp_tmp
        skew_ORG_bs[i] = skew_tmp
        kurt_ORG_bs[i] = kurt_tmp
        bind_ORG_bs[i] = bind_tmp
    end

    cond_lb_bs   = Vector{Float64}(undef, N_bs) 
    susp_lb_bs   = Vector{Float64}(undef, N_bs)
    skew_lb_bs   = Vector{Float64}(undef, N_bs) 
    kurt_lb_bs   = Vector{Float64}(undef, N_bs)
    bind_lb_bs   = Vector{Float64}(undef, N_bs)

    Q1_P1_bs  = Vector{Float64}(undef, N_bs)
    Q2_P1_bs  = Vector{Float64}(undef, N_bs)
    Q3_P1_bs  = Vector{Float64}(undef, N_bs)
    Q4_P1_bs  = Vector{Float64}(undef, N_bs)

    Q1_P2_bs  = Vector{Float64}(undef, N_bs)
    Q2_P2_bs  = Vector{Float64}(undef, N_bs)
    Q3_P2_bs  = Vector{Float64}(undef, N_bs)
    Q4_P2_bs  = Vector{Float64}(undef, N_bs)

    @inbounds for i in 1:N_bs
        # LB
        if N_lb != 0
            Q1_Y_lb = Q_mean_Y_lb[1, i]
            Q2_Y_lb = Q_mean_Y_lb[2, i]
            Q3_Y_lb = Q_mean_Y_lb[3, i]
            Q4_Y_lb = Q_mean_Y_lb[4, i]
        else
            Q1_Y_lb = 0.0
            Q2_Y_lb = 0.0
            Q3_Y_lb = 0.0
            Q4_Y_lb = 0.0
        end

        if N_bc != 0 && N_tr != 0
            Q1_Y_bc  = Q_mean_Y_bc[1, i]
            Q2_Y_bc  = Q_mean_Y_bc[2, i]
            Q3_Y_bc  = Q_mean_Y_bc[3, i]
            Q4_Y_bc  = Q_mean_Y_bc[4, i]
            Q1_YP_bc = Q_mean_YP_bc[1, i]
            Q2_YP_bc = Q_mean_YP_bc[2, i]
            Q3_YP_bc = Q_mean_YP_bc[3, i]
            Q4_YP_bc = Q_mean_YP_bc[4, i]
            Q1_YmYP  = Q1_Y_bc - Q1_YP_bc
            Q2_YmYP  = Q2_Y_bc - Q2_YP_bc
            Q3_YmYP  = Q3_Y_bc - Q3_YP_bc
            Q4_YmYP  = Q4_Y_bc - Q4_YP_bc
        else
            Q1_YmYP  = 0.0
            Q2_YmYP  = 0.0
            Q3_YmYP  = 0.0
            Q4_YmYP  = 0.0
        end

        if N_tr != 0
            Q1_YP_ul = Q_mean_YP_ul[1, i]
            Q2_YP_ul = Q_mean_YP_ul[2, i]
            Q3_YP_ul = Q_mean_YP_ul[3, i]
            Q4_YP_ul = Q_mean_YP_ul[4, i]
        else
            Q1_YP_ul = 0.0
            Q2_YP_ul = 0.0
            Q3_YP_ul = 0.0
            Q4_YP_ul = 0.0
            Q1_Y_lb = Q_mean_Y_lb[1,i]
            Q2_Y_lb = Q_mean_Y_lb[2,i]
            Q3_Y_lb = Q_mean_Y_lb[3,i]
            Q4_Y_lb = Q_mean_Y_lb[4,i]
            s, sk, k, b = Cumulants.calc_cumulants(
                float(V), Q1_Y_lb, Q2_Y_lb, Q3_Y_lb, Q4_Y_lb, jobid
            )
            cond_lb_bs[i] = Q1_Y_lb / V
            susp_lb_bs[i] = s
            skew_lb_bs[i] = sk
            kurt_lb_bs[i] = k
            bind_lb_bs[i] = b
        end

        Q1_P1_bs[i] = Q1_YP_ul + Q1_YmYP
        Q2_P1_bs[i] = Q2_YP_ul + Q2_YmYP
        Q3_P1_bs[i] = Q3_YP_ul + Q3_YmYP
        Q4_P1_bs[i] = Q4_YP_ul + Q4_YmYP

        Q1_P2_bs[i] = w_lb * Q1_Y_lb + w_ul * Q1_P1_bs[i]
        Q2_P2_bs[i] = w_lb * Q2_Y_lb + w_ul * Q2_P1_bs[i]
        Q3_P2_bs[i] = w_lb * Q3_Y_lb + w_ul * Q3_P1_bs[i]
        Q4_P2_bs[i] = w_lb * Q4_Y_lb + w_ul * Q4_P1_bs[i]
    end

    # === Derive P1/P2 cumulants from subset cumulants ===
    cond_P1_bs = Vector{Float64}(undef, N_bs)
    susp_P1_bs = Vector{Float64}(undef, N_bs)
    skew_P1_bs = Vector{Float64}(undef, N_bs)
    kurt_P1_bs = Vector{Float64}(undef, N_bs)
    bind_P1_bs = Vector{Float64}(undef, N_bs)

    cond_P2_bs = Vector{Float64}(undef, N_bs)
    susp_P2_bs = Vector{Float64}(undef, N_bs)
    skew_P2_bs = Vector{Float64}(undef, N_bs)
    kurt_P2_bs = Vector{Float64}(undef, N_bs)
    bind_P2_bs = Vector{Float64}(undef, N_bs)

    @inbounds for i in 1:N_bs
        if N_tr == 0
            # No training: P1 = P2 = LB
            cond_P1_bs[i] = cond_lb_bs[i]
            susp_P1_bs[i] = susp_lb_bs[i]
            skew_P1_bs[i] = skew_lb_bs[i]
            kurt_P1_bs[i] = kurt_lb_bs[i]
            bind_P1_bs[i] = bind_lb_bs[i]

            cond_P2_bs[i] = cond_lb_bs[i]
            susp_P2_bs[i] = susp_lb_bs[i]
            skew_P2_bs[i] = skew_lb_bs[i]
            kurt_P2_bs[i] = kurt_lb_bs[i]
            bind_P2_bs[i] = bind_lb_bs[i]
        else
            Q1 = Q1_P1_bs[i]
            Q2 = Q2_P1_bs[i]
            Q3 = Q3_P1_bs[i]
            Q4 = Q4_P1_bs[i]

            susp_tmp, skew_tmp, kurt_tmp, bind_tmp = Cumulants.calc_cumulants(
                float(V), Q1, Q2, Q3, Q4, jobid
            )

            cond_P1_bs[i] = Q1 / V
            susp_P1_bs[i] = susp_tmp
            skew_P1_bs[i] = skew_tmp
            kurt_P1_bs[i] = kurt_tmp
            bind_P1_bs[i] = bind_tmp

            Q1 = Q1_P2_bs[i]
            Q2 = Q2_P2_bs[i]
            Q3 = Q3_P2_bs[i]
            Q4 = Q4_P2_bs[i]

            susp_tmp, skew_tmp, kurt_tmp, bind_tmp = Cumulants.calc_cumulants(
                float(V), Q1, Q2, Q3, Q4, jobid
            )

            cond_P2_bs[i] = Q1 / V
            susp_P2_bs[i] = susp_tmp
            skew_P2_bs[i] = skew_tmp
            kurt_P2_bs[i] = kurt_tmp
            bind_P2_bs[i] = bind_tmp
        end
    end

    # === Pack results ===
    cumulants_OG_bs   = [cond_ORG_bs, susp_ORG_bs, skew_ORG_bs, kurt_ORG_bs, bind_ORG_bs]
    cumulants_P1_bs   = [cond_P1_bs,  susp_P1_bs,  skew_P1_bs,  kurt_P1_bs,  bind_P1_bs ]
    cumulants_P2_bs   = [cond_P2_bs,  susp_P2_bs,  skew_P2_bs,  kurt_P2_bs,  bind_P2_bs ]

    # === Bundle index metadata ===
    idx_bundle = Dict(
        :all => idx_all,
        :lb  => idx_lb,
        :bc  => idx_bc,
        :ul  => idx_ul
    )

    return cumulants_OG_bs, cumulants_P1_bs, cumulants_P2_bs, idx_bundle
end