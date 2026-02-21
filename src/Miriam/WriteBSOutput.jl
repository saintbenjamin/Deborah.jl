# =============================================================================
# src/Miriam/WriteBSOutput.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# =============================================================================

module WriteBSOutput

import Printf: @sprintf
import ..Sarah.Bootstrap
import ..Sarah.SeedManager
import ..Ensemble
import ..EnsembleUtils
import ..CumulantsBundle

"""
    write_bs_cumulants(
        ens_bundle::Ensemble.EnsembleArrayBundle{T},
        N_bs::Int,
        blk_size::Int,
        method::String,
        fname_OG::String,
        fname_P1::String,
        fname_P2::String,
        jobid::Union{Nothing, String}=nothing;
        rng_pool::SeedManager.RNGPool=rng_pool,
    ) where T -> Nothing

Compute and write bootstrap estimates of cumulants for single ensemble.

This function computes three types of cumulants from a given ensemble:
- `OG` : Original data
- `P1` : Bias-corrected estimator
- `P2` : weighted average of `P1` and labeled set

The output includes bootstrap averages and standard errors for each cumulant:
chiral condensate, susceptibility, skewness, kurtosis, and Binder cumulant.

# Arguments
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): The ensemble data container.
- `N_bs::Int`: Number of bootstrap resamples.
- `blk_size::Int`: Default block size for bootstrap resampling.
- `method::String`   : Block-bootstrap scheme (case-sensitive):
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB; resample disjoint blocks).
    - `"moving"`        — Moving Block Bootstrap (MBB; resample sliding windows).
    - `"circular"`      — Circular Block Bootstrap (CBB; sliding windows with wrap-around).
- `fname_OG::String`: Output path for original data.
- `fname_P1::String`: Output path for `P1` estimator.
- `fname_P2::String`: Output path for `P2` estimator.
- `jobid::Union{Nothing, String}`: Optional job ID for contextual logging.
- [`rng_pool::SeedManager.RNGPool`](@ref Deborah.Sarah.SeedManager.RNGPool): Optional RNG pool for bootstrap sampling.

# Output
Three text files written with bootstrap results:
Each line corresponds to a ``\\kappa`` value with 5 cumulants and their errors.

# Notes
- The same bootstrap indices are reused across all estimators for consistent comparison.
- All output files share the same formatting and cumulant ordering.
"""
function write_bs_cumulants(
    ens_bundle::Ensemble.EnsembleArrayBundle{T},
    N_bs::Int,
    blk_size::Int,
    method::String,
    fname_OG::String,
    fname_P1::String,
    fname_P2::String,
    jobid::Union{Nothing, String}=nothing;
    rng_pool::SeedManager.RNGPool=rng_pool,
) where T

    ofs_OG = open(fname_OG, "w")
    ofs_P1 = open(fname_P1, "w")
    ofs_P2 = open(fname_P2, "w")

    # Header for output files (observables and error per cumulant)
    println(ofs_OG,
        "# kappa                                  " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )
    println(ofs_P1,
        "# kappa                                  " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )
    println(ofs_P2,
        "# kappa                                  " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )

    for i in eachindex(ens_bundle.arrays[1].data)

        ens_i = ens_bundle.arrays[1].data[i]

        paramT = Ensemble.Params(
            ens_i.param.ns,
            ens_i.param.nt,
            ens_i.param.nf,
            ens_i.param.beta,
            ens_i.param.csw,
            ens_i.param.kappa
        )
        
        V = paramT.ns^3 * paramT.nt

        trMi_all = Vector{Vector{Vector{Float64}}}(undef, length(ens_bundle.arrays))

        for j in 1:length(ens_bundle.arrays)
            ens = ens_bundle.arrays[j].data[i]
            trMi_all[j] = [EnsembleUtils.trMiT(ens, paramT, iconf) for iconf in 1:ens.nconf]
        end

        cumulants_OG_bs, cumulants_P1_bs, 
        cumulants_P2_bs, _ =
            CumulantsBundle.compute_cumulants_bundle_raw(
                N_bs, blk_size, method, V, trMi_all, ens_bundle, i, jobid; 
                rng_pool=rng_pool
            )

        # --- Bootstrap average and error extraction ---
        cumulants_OG_avgerr = map(Bootstrap.bootstrap_average_error, cumulants_OG_bs)
        cumulants_P1_avgerr = map(Bootstrap.bootstrap_average_error, cumulants_P1_bs)
        cumulants_P2_avgerr = map(Bootstrap.bootstrap_average_error, cumulants_P2_bs)

        cumulants_OG = getindex.(cumulants_OG_avgerr, 1)
        errors_OG    = getindex.(cumulants_OG_avgerr, 2)
        cumulants_P1 = getindex.(cumulants_P1_avgerr, 1)
        errors_P1    = getindex.(cumulants_P1_avgerr, 2)
        cumulants_P2 = getindex.(cumulants_P2_avgerr, 1)
        errors_P2    = getindex.(cumulants_P2_avgerr, 2)

        # --- Write current cumulants and errors to file ---
        println(ofs_OG, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", cumulants_OG[j], errors_OG[j])
                for j in eachindex(cumulants_OG)], ""))

        println(ofs_P1, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", cumulants_P1[j], errors_P1[j])

                for j in eachindex(cumulants_P1)], ""))
        println(ofs_P2, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", cumulants_P2[j], errors_P2[j])
                for j in eachindex(cumulants_P2)], ""))

    end

    close(ofs_OG)
    close(ofs_P1)
    close(ofs_P2)

end

"""
    write_bs_moments(
        ens_bundle::Ensemble.EnsembleArrayBundle{T},
        N_bs::Int,
        blk_size::Int,
        method::String,
        fname_OG::String,
        fname_P1::String,
        fname_P2::String,
        jobid::Union{Nothing, String}=nothing;
        rng_pool::SeedManager.RNGPool=rng_pool,
    ) where T -> Nothing

Compute and write bootstrap estimates for raw moments (``Q_1``, ``Q_2``, ``Q_3``, ``Q_4``) in the single ensemble.

This function computes three moment estimators from a given ensemble:
- `OG` : Original data
- `P1` : Bias-corrected estimator
- `P2` : weighted average of `P1` and labeled set

Each estimator is produced via block bootstrap and summarized by its bootstrap
mean and standard error for ``Q_1``, ``Q_2``, ``Q_3``, ``Q_4``.

# Arguments
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Ensemble data container.
- `N_bs::Int`: Number of bootstrap resamples.
- `blk_size::Int`: Default block size for bootstrap resampling.
- `method::String`: Block-bootstrap scheme (case-sensitive), e.g.:
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB).
    - `"moving"`        — Moving Block Bootstrap (MBB).
    - `"circular"`      — Circular Block Bootstrap (CBB).
- `fname_OG::String`: Output path for original data.
- `fname_P1::String`: Output path for `P1` estimator.
- `fname_P2::String`: Output path for `P2` estimator.
- `jobid::Union{Nothing, String}`: Optional job ID for logging.
- [`rng_pool::SeedManager.RNGPool`](@ref Deborah.Sarah.SeedManager.RNGPool): RNG pool used for bootstrap.

# Output
Three text files written with bootstrap results.  
Each line contains the ``\\kappa`` value followed by ``Q_1 \\pm \\sigma_{Q_1}``, ``Q_2 \\pm \\sigma_{Q_2}``, ``Q_3 \\pm \\sigma_{Q_3}``, ``Q_4 \\pm \\sigma_{Q_4}``.

# Notes
- Internally calls [`Deborah.Miriam.CumulantsBundle.compute_moments_bundle_raw`](@ref) to generate
  bootstrap samples for `OG`/`P1`/`P2` moments.
- The same bootstrap index plans are reused across all estimators for consistency.
"""
function write_bs_moments(
    ens_bundle::Ensemble.EnsembleArrayBundle{T},
    N_bs::Int,
    blk_size::Int,
    method::String,
    fname_OG::String,
    fname_P1::String,
    fname_P2::String,
    jobid::Union{Nothing, String}=nothing;
    rng_pool::SeedManager.RNGPool=rng_pool,
) where T

    ofs_OG = open(fname_OG, "w")
    ofs_P1 = open(fname_P1, "w")
    ofs_P2 = open(fname_P2, "w")

    # Header for output files (observables and error per cumulant)
    println(ofs_OG,
        "# kappa                                  " *
        "Q1                  error                " *
        "Q2                  error                " *
        "Q3                  error                " *
        "Q4                  error                "
    )
    println(ofs_P1,
        "# kappa                                  " *
        "Q1                  error                " *
        "Q2                  error                " *
        "Q3                  error                " *
        "Q4                  error                "
    )
    println(ofs_P2,
        "# kappa                                  " *
        "Q1                  error                " *
        "Q2                  error                " *
        "Q3                  error                " *
        "Q4                  error                "
    )

    for i in eachindex(ens_bundle.arrays[1].data)

        ens_i = ens_bundle.arrays[1].data[i]

        paramT = Ensemble.Params(
            ens_i.param.ns,
            ens_i.param.nt,
            ens_i.param.nf,
            ens_i.param.beta,
            ens_i.param.csw,
            ens_i.param.kappa
        )
        
        trMi_all = Vector{Vector{Vector{Float64}}}(undef, length(ens_bundle.arrays))

        for j in 1:length(ens_bundle.arrays)
            ens = ens_bundle.arrays[j].data[i]
            trMi_all[j] = [EnsembleUtils.trMiT(ens, paramT, iconf) for iconf in 1:ens.nconf]
        end

        moments_OG_bs, moments_P1_bs, 
        moments_P2_bs, _ =
            CumulantsBundle.compute_moments_bundle_raw(
                N_bs, blk_size, method, trMi_all, ens_bundle, i, jobid; 
                rng_pool=rng_pool
            )

        # --- Bootstrap average and error extraction ---
        moments_OG_avgerr = map(Bootstrap.bootstrap_average_error, moments_OG_bs)
        moments_P1_avgerr = map(Bootstrap.bootstrap_average_error, moments_P1_bs)
        moments_P2_avgerr = map(Bootstrap.bootstrap_average_error, moments_P2_bs)

        moments_OG = getindex.(moments_OG_avgerr, 1)
        errors_OG  = getindex.(moments_OG_avgerr, 2)
        moments_P1 = getindex.(moments_P1_avgerr, 1)
        errors_P1  = getindex.(moments_P1_avgerr, 2)
        moments_P2 = getindex.(moments_P2_avgerr, 1)
        errors_P2  = getindex.(moments_P2_avgerr, 2)

        # --- Write current cumulants and errors to file ---
        println(ofs_OG, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", moments_OG[j], errors_OG[j])
                for j in eachindex(moments_OG)], ""))

        println(ofs_P1, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", moments_P1[j], errors_P1[j])

                for j in eachindex(moments_P1)], ""))
        println(ofs_P2, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", moments_P2[j], errors_P2[j])
                for j in eachindex(moments_P2)], ""))

    end

    close(ofs_OG)
    close(ofs_P1)
    close(ofs_P2)

end

"""
    write_bs_traces(
        ens_bundle::Ensemble.EnsembleArrayBundle{T},
        N_bs::Int,
        blk_size::Int,
        method::String,
        fname_OG::String,
        fname_P1::String,
        fname_P2::String,
        jobid::Union{Nothing, String}=nothing;
        rng_pool::SeedManager.RNGPool=rng_pool,
    ) where T -> Nothing

Compute and write bootstrap estimates for raw trace components (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) in the single ensemble.

This function computes three trace estimators from a given ensemble:
- `OG` : Original data
- `P1` : Bias-corrected estimator
- `P2` : weighted average of `P1` and labeled set

Each estimator is produced via block bootstrap and summarized by its bootstrap
mean and standard error for ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.

# Arguments
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Ensemble data container.
- `N_bs::Int`: Number of bootstrap resamples.
- `blk_size::Int`: Default block size for bootstrap resampling.
- `method::String`: Block-bootstrap scheme (case-sensitive), e.g.:
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB).
    - `"moving"`        — Moving Block Bootstrap (MBB).
    - `"circular"`      — Circular Block Bootstrap (CBB).
- `fname_OG::String`: Output path for original data.
- `fname_P1::String`: Output path for `P1` estimator.
- `fname_P2::String`: Output path for `P2` estimator.
- `jobid::Union{Nothing, String}`: Optional job ID for logging.
- [`rng_pool::SeedManager.RNGPool`](@ref Deborah.Sarah.SeedManager.RNGPool): RNG pool used for bootstrap.

# Output
Three text files written with bootstrap results.

# Notes
- Traces are taken from un-rescaled vectors via [`Deborah.Miriam.EnsembleUtils.trMi_rawT`](@ref) before resampling.
- Internally calls [`Deborah.Miriam.CumulantsBundle.compute_traces_bundle_raw`](@ref) to generate
  bootstrap samples for `OG`/`P1`/`P2` traces.
- The same bootstrap index plans are reused across all estimators for consistency.
"""
function write_bs_traces(
    ens_bundle::Ensemble.EnsembleArrayBundle{T},
    N_bs::Int,
    blk_size::Int,
    method::String,
    fname_OG::String,
    fname_P1::String,
    fname_P2::String,
    jobid::Union{Nothing, String}=nothing;
    rng_pool::SeedManager.RNGPool=rng_pool,
) where T

    ofs_OG = open(fname_OG, "w")
    ofs_P1 = open(fname_P1, "w")
    ofs_P2 = open(fname_P2, "w")

    # Header for output files (observables and error per cumulant)
    println(ofs_OG,
        "# kappa                                  " *
        "trM1                error                " *
        "trM2                error                " *
        "trM3                error                " *
        "trM4                error                "
    )
    println(ofs_P1,
        "# kappa                                  " *
        "trM1                error                " *
        "trM2                error                " *
        "trM3                error                " *
        "trM4                error                "
    )
    println(ofs_P2,
        "# kappa                                  " *
        "trM1                error                " *
        "trM2                error                " *
        "trM3                error                " *
        "trM4                error                "
    )

    for i in eachindex(ens_bundle.arrays[1].data)

        ens_i = ens_bundle.arrays[1].data[i]

        paramT = Ensemble.Params(
            ens_i.param.ns,
            ens_i.param.nt,
            ens_i.param.nf,
            ens_i.param.beta,
            ens_i.param.csw,
            ens_i.param.kappa
        )
        
        trMi_all = Vector{Vector{Vector{Float64}}}(undef, length(ens_bundle.arrays))

        for j in 1:length(ens_bundle.arrays)
            ens = ens_bundle.arrays[j].data[i]
            trMi_all[j] = [EnsembleUtils.trMi_rawT(ens, paramT, iconf) for iconf in 1:ens.nconf]
        end

        traces_OG_bs, traces_P1_bs, 
        traces_P2_bs, _ =
            CumulantsBundle.compute_traces_bundle_raw(
                N_bs, blk_size, method, trMi_all, ens_bundle, i, jobid; 
                rng_pool=rng_pool
            )

        # --- Bootstrap average and error extraction ---
        traces_OG_avgerr = map(Bootstrap.bootstrap_average_error, traces_OG_bs)
        traces_P1_avgerr = map(Bootstrap.bootstrap_average_error, traces_P1_bs)
        traces_P2_avgerr = map(Bootstrap.bootstrap_average_error, traces_P2_bs)

        traces_OG = getindex.(traces_OG_avgerr, 1)
        errors_OG = getindex.(traces_OG_avgerr, 2)
        traces_P1 = getindex.(traces_P1_avgerr, 1)
        errors_P1 = getindex.(traces_P1_avgerr, 2)
        traces_P2 = getindex.(traces_P2_avgerr, 1)
        errors_P2 = getindex.(traces_P2_avgerr, 2)

        # --- Write current cumulants and errors to file ---
        println(ofs_OG, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", traces_OG[j], errors_OG[j])
                for j in eachindex(traces_OG)], ""))

        println(ofs_P1, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", traces_P1[j], errors_P1[j])

                for j in eachindex(traces_P1)], ""))
        println(ofs_P2, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", traces_P2[j], errors_P2[j])
                for j in eachindex(traces_P2)], ""))

    end

    close(ofs_OG)
    close(ofs_P1)
    close(ofs_P2)

end

end  # module WriteBSOutput