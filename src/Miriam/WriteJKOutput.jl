# ============================================================================
# src/Miriam/WriteJKOutput.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module WriteJKOutput

import ..Printf: @sprintf

import ..Sarah.Jackknife
import ..Ensemble
import ..EnsembleUtils
import ..Cumulants

"""
    write_jk_cumulants(
        ens_array::Ensemble.EnsembleArray{T},
        bin_size::Int,
        fname::String
    ) where T -> Nothing

Compute jackknife estimates of cumulants for single ensemble and write results to a text file.

This function calculates 5 cumulants -- chiral condensate, susceptibility, skewness,
kurtosis, and Binder -- for each configuration set using jackknife resampling,
and saves the results to the given output file.

# Arguments
- [`ens_array::Ensemble.EnsembleArray{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArray): Array of ensembles to analyze.
- `bin_size::Int`: Number of configurations per jackknife bin.
- `fname::String`: Path to output file.

# Output
- A plain text file written at `fname`, containing ``\\kappa`` values followed by
  each cumulant and its error in order:
  ``\\kappa``, ``\\Sigma \\pm \\sigma_{\\Sigma}``, ``\\chi \\pm \\sigma_{\\chi}``, ``S \\pm \\sigma_{S}``, ``K \\pm \\sigma_{K}``, ``B \\pm \\sigma_{B}``.

# Notes
- The cumulants are computed using [`Deborah.Miriam.Cumulants.compute_cumulants_raw`](@ref).
- Volume factor ``V = N_S^3 \\times N_T`` is used for normalization of chiral condensate and susceptibility.
"""
function write_jk_cumulants(
    ens_array::Ensemble.EnsembleArray{T},
    bin_size::Int,
    fname::String
) where T

    ofs = open(fname, "w")

    println(ofs,
        "# kappa                                  " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )

    for ens in ens_array.data

        paramT = Ensemble.Params(
            ens.param.ns, 
            ens.param.nt, 
            ens.param.nf,
            ens.param.beta, 
            ens.param.csw,
            ens.param.kappa
        )

        V = paramT.ns^3 * paramT.nt

        trMi_tmp = [EnsembleUtils.trMiT(ens, paramT, iconf) for iconf in 1:ens.nconf]

        cumulants_jk = Cumulants.compute_cumulants_raw(bin_size, V, trMi_tmp)

        # --- Jackknife average and error extraction ---
        cumulants_avgerr = map(Jackknife.jackknife_average_error, cumulants_jk)

        cumulants = getindex.(cumulants_avgerr, 1)
        errors    = getindex.(cumulants_avgerr, 2)

        println(ofs, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", cumulants[j], errors[j])
                for j in eachindex(cumulants)], ""))

    end

    close(ofs)

end

"""
    write_jk_moments(
        ens_array::Ensemble.EnsembleArray{T},
        bin_size::Int,
        fname::String
    ) where T -> Nothing

Compute jackknife estimates of the raw moments ``Q_1``, ``Q_2``, ``Q_3``, ``Q_4`` and write results to a text file.

This function evaluates the per-configuration moments (``Q_1``, ``Q_2``, ``Q_3``, ``Q_4``) via
[`Deborah.Miriam.Cumulants.calc_Q`](@ref), applies jackknife resampling to each moment component using the given
`bin_size`, and saves the jackknife mean and error to `fname` for each ensemble.

# Arguments
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Array of ensembles to analyze.
- `bin_size::Int`: Number of configurations per jackknife bin.
- `fname::String`: Path to the output file.

# Output
- A plain text file written at `fname`, containing κ values followed by
  each moment and its error in order:
  ``\\kappa``,  ``Q_1 \\pm \\sigma_{Q_1}``, ``Q_2 \\pm \\sigma_{Q_2}``, ``Q_3 \\pm \\sigma_{Q_3}``, ``Q_4 \\pm \\sigma_{Q_4}``
- A header line is included:
  `# kappa  _Q_1 error  _Q_2 error  _Q_3 error  _Q_4 error`

# Notes
- Moments are produced by [`Deborah.Miriam.Cumulants.compute_moments_raw`](@ref),
  where `trMi_tmp` is typically built from [`Deborah.Miriam.EnsembleUtils.trMiT`](@ref).
"""
function write_jk_moments(
    ens_array::Ensemble.EnsembleArray{T},
    bin_size::Int,
    fname::String
) where T

    ofs = open(fname, "w")

    println(ofs,
        "# kappa                                  " *
        "Q1                  error                " *
        "Q2                  error                " *
        "Q3                  error                " *
        "Q4                  error                "
    )

    for ens in ens_array.data

        paramT = Ensemble.Params(
            ens.param.ns, 
            ens.param.nt, 
            ens.param.nf,
            ens.param.beta, 
            ens.param.csw,
            ens.param.kappa
        )

        trMi_tmp = [EnsembleUtils.trMiT(ens, paramT, iconf) for iconf in 1:ens.nconf]

        moments_jk = Cumulants.compute_moments_raw(bin_size, trMi_tmp)

        # --- Jackknife average and error extraction ---
        moments_avgerr = map(Jackknife.jackknife_average_error, moments_jk)

        moments = getindex.(moments_avgerr, 1)
        errors = getindex.(moments_avgerr, 2)

        println(ofs, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", moments[j], errors[j])
                for j in eachindex(moments)], ""))

    end

    close(ofs)

end

"""
    write_jk_traces(
        ens_array::Ensemble.EnsembleArray{T},
        bin_size::Int,
        fname::String
    ) where T -> Nothing

Compute jackknife estimates of raw trace values (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) and write results to a text file.

This function takes per-configuration trace vectors (with layout like
`[*, trM1, trM2, trM3, trM4]`), applies jackknife resampling to each of ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``,
and saves the jackknife mean and error to `fname` for each ensemble.

# Arguments
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Array of ensembles to analyze.
- `bin_size::Int`: Number of configurations per jackknife bin.
- `fname::String`: Path to the output file.

# Output
- A plain text file written at `fname`.
- A header line is included:
  `# kappa  trM1 error  trM2 error  trM3 error  trM4 error`

# Notes
- The jackknife resamples are constructed by
  [`Deborah.Miriam.Cumulants.compute_traces_raw`](@ref) (a utility that returns
  `[trM1_jk, trM2_jk, trM3_jk, trM4_jk]`), where `trMi_tmp` is obtained
  via [`Deborah.Miriam.EnsembleUtils.trMi_rawT`](@ref).
- No cumulants are computed in this routine; it only processes the raw traces.
"""
function write_jk_traces(
    ens_array::Ensemble.EnsembleArray{T},
    bin_size::Int,
    fname::String
) where T

    ofs = open(fname, "w")

    println(ofs,
        "# kappa                                  " *
        "trM1                error                " *
        "trM2                error                " *
        "trM3                error                " *
        "trM4                error                "
    )

    for ens in ens_array.data

        paramT = Ensemble.Params(
            ens.param.ns, 
            ens.param.nt, 
            ens.param.nf,
            ens.param.beta, 
            ens.param.csw,
            ens.param.kappa
        )

        trMi_tmp = [EnsembleUtils.trMi_rawT(ens, paramT, iconf) for iconf in 1:ens.nconf]

        traces_jk = Cumulants.compute_traces_raw(bin_size, trMi_tmp)

        # --- Jackknife average and error extraction ---
        traces_avgerr = map(Jackknife.jackknife_average_error, traces_jk)

        traces = getindex.(traces_avgerr, 1)
        errors = getindex.(traces_avgerr, 2)

        println(ofs, @sprintf("%.12e                      ", paramT.kappa),
            join([@sprintf("% .12e  %.12e  ", traces[j], errors[j])
                for j in eachindex(traces)], ""))

    end

    close(ofs)

end

end  # module WriteJKOutput