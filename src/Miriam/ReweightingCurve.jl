# ============================================================================
# src/Miriam/ReweightingCurve.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ReweightingCurve

import ..Printf: @sprintf, @printf

import ..Sarah.JobLoggerTools
import ..Sarah.Jackknife
import ..Ensemble
import ..EnsembleUtils
import ..Cumulants
import ..Reweighting
import ..Interpolation

"""
    reweighting_curve!(
        rw::Reweighting.ReweightingSolver{T},
        ens_array::Ensemble.EnsembleArray{T},
        nkappaT::Int,
        bin_size::Int,
        fname::String,
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Compute and export reweighting curves and transition ``\\kappa_T`` estimates based on interpolated cumulants.

This function performs a full sweep of reweighting analysis:
1. Scans `nkappaT` values of ``\\kappa_T`` over the range defined by the ensemble.
2. Computes reweighted observables (chiral condensate, susceptibility, skewness, kurtosis, Binder cumulant) at each point.
3. Tracks the peak (or extremum) ``\\kappa_T`` for each observable.
4. Uses 3-point Lagrange interpolation around the extremum to estimate transition ``\\kappa_T``.
5. Exports full sweep and transition analysis results to a file.

# Arguments
- [`rw::Reweighting.ReweightingSolver{T}`](@ref Deborah.Miriam.Reweighting.ReweightingSolver): Instance holding ensembles and internal buffer for reweighting weights.
- [`ens_array::Ensemble.EnsembleArray{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArray): Ensemble data for scanning ``\\kappa_T``.
- `nkappaT::Int`: Number of ``\\kappa_T`` points to scan across the range.
- `bin_size::Int`: Bin size for Jackknife resampling.
- `fname::String`: Output filename where curve and transition results will be written.
- `jobid::Union{Nothing, String}`: Optional job ID tag for logging.

# Behavior
- Writes header and reweighting data to `fname`.
- Writes estimated transition ``\\kappa_T``s for `susp`, `skew`, and `kurt` via interpolation of jackknife resamples.
- Logs each transition computation phase using [`Deborah.Sarah.JobLoggerTools.log_stage_sub1_benji`](@ref).

# Returns
- `Nothing`: All output is side-effect (written to file, modifies internal weights `rw.w`).
"""
function reweighting_curve!(
    rw::Reweighting.ReweightingSolver{T},
    ens_array::Ensemble.EnsembleArray{T},
    nkappaT::Int,
    bin_size::Int,
    fname::String,
    jobid::Union{Nothing, String}=nothing
) where T

    ofs = open(fname, "w")

    # Header for output files (observables and error per cumulant)
    println(ofs,
        "# kappa                                  " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )

    kappa = [ens.param.kappa for ens in ens_array.data]
    JobLoggerTools.assert_benji(nkappaT >= 2, "At least two κ points are required for reweighting curve", jobid)
    dkappaT = (kappa[end] - kappa[1]) / (nkappaT - 1)

    # Reference parameter set from the first ensemble (for β, ns, nt, etc.)
    param_ref = ens_array.data[1].param
    ns, nt, nf, beta, csw = param_ref.ns, param_ref.nt, param_ref.nf, param_ref.beta, param_ref.csw

    # Allocate trace data holder
    trMi_all = Vector{Vector{Float64}}(undef, rw.nconf_all)

    njk = div(rw.nconf_all, bin_size)
    cumulants_jk_all  = [ [Vector{Float64}(undef, njk) for i in 1:nkappaT] for j in 1:5 ]
    cumulants_avg = [ Vector{Float64}(undef, nkappaT) for j in 1:5 ]

    # --- Step 1: Reweight over κ_T scan ---
    for i in 1:nkappaT
        kappaT = kappa[1] + ( i - 1 ) * dkappaT
        paramT = Ensemble.Params(ns, nt, nf, beta, csw, kappaT)

        Reweighting.calc_w!(rw, paramT)

        V = paramT.ns^3 * paramT.nt

        ii = 1
        for ikappa in eachindex(ens_array.data)
            ens = ens_array.data[ikappa]
            for iconf in 1:ens.nconf
                trMi_all[ii] = EnsembleUtils.trMiT(ens, paramT, iconf)
                ii += 1
            end
        end

        cumulants_jk = Cumulants.compute_cumulants(
            bin_size, 
            V, 
            trMi_all, 
            rw.w
        )

        for j in 1:5
            cumulants_jk_all[j][i] .= cumulants_jk[j]
        end

        # --- Jackknife average and error extraction ---
        cumulants_avgerr = map(Jackknife.jackknife_average_error, cumulants_jk)

        cumulants = getindex.(cumulants_avgerr, 1)
        errors    = getindex.(cumulants_avgerr, 2)

        # Write κ_T step output
        println(ofs, @sprintf("%.12e                      ", kappaT), 
            join([@sprintf("% .12e  %.12e  ", cumulants[j], errors[j]) 
                for j in eachindex(cumulants)], ""))

        # Store jackknife means into cumulants_jk_avg
        for j in 1:5
            cumulants_avg[j][i] = cumulants[j]
        end
    end

    # --- Step 2: Locate transition κ_T via 3-point interpolation ---
    println(ofs,
        "\n\n# kappa_t           error                " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )

    kappa_vec = [kappa[1] + (i - 1) * dkappaT for i in 1:nkappaT]

    L_susp, R_susp, i_susp = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_avg; mode=:susp)
    L_skew, R_skew, i_skew = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_avg; mode=:skew)
    L_kurt, R_kurt, i_kurt = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_avg; mode=:kurt)

    # ---- susceptibility (2) ----
    vec_valids_susp =
        Interpolation.find_transition_cumulants(
            cumulants_jk_all,
            kappa_vec,
            2,
            L_susp, R_susp, i_susp,
            jobid
        )

    out_susp    = Vector{eltype(kappa_vec)}(undef, 6)
    errors_susp = Vector{eltype(kappa_vec)}(undef, 6)
    for i in 1:6
        v = vec_valids_susp[i]
        if isempty(v)
            out_susp[i]    = eltype(kappa_vec)(NaN)
            errors_susp[i] = eltype(kappa_vec)(NaN)
        else
            out_susp[i], errors_susp[i] = Jackknife.jackknife_average_error(v)
        end
    end

    @printf(ofs, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n", 
        out_susp[1], errors_susp[1], out_susp[2], errors_susp[2], 
        out_susp[3], errors_susp[3], out_susp[4], errors_susp[4], 
        out_susp[5], errors_susp[5], out_susp[6], errors_susp[6], "susp")

    # ---- skewness (3) ----
    vec_valids_skew =
        Interpolation.find_transition_cumulants(
            cumulants_jk_all,
            kappa_vec,
            3,
            L_skew, R_skew, i_skew,
            jobid
        )

    out_skew    = Vector{eltype(kappa_vec)}(undef, 6)
    errors_skew = Vector{eltype(kappa_vec)}(undef, 6)
    for i in 1:6
        v = vec_valids_skew[i]
        if isempty(v)
            out_skew[i]    = eltype(kappa_vec)(NaN)
            errors_skew[i] = eltype(kappa_vec)(NaN)
        else
            out_skew[i], errors_skew[i] = Jackknife.jackknife_average_error(v)
        end
    end

    @printf(ofs, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n", 
        out_skew[1], errors_skew[1], out_skew[2], errors_skew[2], 
        out_skew[3], errors_skew[3], out_skew[4], errors_skew[4], 
        out_skew[5], errors_skew[5], out_skew[6], errors_skew[6], "skew")

    # ---- kurtosis (4) ----
    vec_valids_kurt =
        Interpolation.find_transition_cumulants(
            cumulants_jk_all,
            kappa_vec,
            4,
            L_kurt, R_kurt, i_kurt,
            jobid
        )

    out_kurt    = Vector{eltype(kappa_vec)}(undef, 6)
    errors_kurt = Vector{eltype(kappa_vec)}(undef, 6)
    for i in 1:6
        v = vec_valids_kurt[i]
        if isempty(v)
            out_kurt[i]    = eltype(kappa_vec)(NaN)
            errors_kurt[i] = eltype(kappa_vec)(NaN)
        else
            out_kurt[i], errors_kurt[i] = Jackknife.jackknife_average_error(v)
        end
    end

    @printf(ofs, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n", 
        out_kurt[1], errors_kurt[1], out_kurt[2], errors_kurt[2], 
        out_kurt[3], errors_kurt[3], out_kurt[4], errors_kurt[4], 
        out_kurt[5], errors_kurt[5], out_kurt[6], errors_kurt[6], "kurt")

    close(ofs)

end

end