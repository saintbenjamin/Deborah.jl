# ============================================================================
# src/Miriam/ReweightingCurveBundle.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ReweightingCurveBundle

import Printf: @sprintf, @printf
import ..Sarah.JobLoggerTools
import ..Sarah.Bootstrap
import ..Sarah.SeedManager
import ..Ensemble
import ..EnsembleUtils
import ..Cumulants
import ..CumulantsBundle
import ..ReweightingBundle
import ..Interpolation

"""
    reweighting_curve_bundle!(
        rw_bundle::ReweightingBundle.ReweightingSolverBundle,
        ens_bundle::Ensemble.EnsembleArrayBundle{T},
        nkappaT::Int,
        N_bs::Int,
        blk_size::Int,
        method::String,
        fname_OG::String,
        fname_P1::String,
        fname_P2::String,
        jobid::Union{Nothing, String}=nothing;
        rng_pool::SeedManager.RNGPool=rng_pool
    ) where T -> Nothing

Compute reweighting curves and phase transition ``\\kappa`` points using multiple estimators (`OG`, `P1`, `P2`)
with bootstrap resampling for uncertainty quantification.

# Arguments
- [`rw_bundle::ReweightingBundle.ReweightingSolverBundle`](@ref Deborah.Miriam.ReweightingBundle.ReweightingSolverBundle): Bundle of reweighting solvers, each associated with a distinct source tag.
- [`ens_bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Bundle of ensemble arrays containing the trace data used for reweighting.
- `nkappaT::Int`: Number of ``\\kappa`` values to scan along the reweighting trajectory.
- `N_bs::Int`: Number of bootstrap resamples.
- `blk_size::Int`: Block size used for bootstrap resampling (`1` = i.i.d. bootstrap).
- `method::String`   : Block-bootstrap scheme (case-sensitive):
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB; resample disjoint blocks).
    - `"moving"`        — Moving Block Bootstrap (MBB; resample sliding windows).
    - `"circular"`      — Circular Block Bootstrap (CBB; sliding windows with wrap-around).
- `fname_OG::String`: Output filename for the original data.
- `fname_P1::String`: Output filename for the `P1` result, which applies bias correction only.
- `fname_P2::String`: Output filename for the `P2` result, a weighted average between `LB` and `P1`.
- `jobid::Union{Nothing, String}`: Optional job ID for contextual logging.
- [`rng_pool::SeedManager.RNGPool`](@ref Deborah.Sarah.SeedManager.RNGPool): Optional RNG pool for reproducible bootstrap resampling.

# Returns
- `Nothing`; results are written to files and logged.
"""
function reweighting_curve_bundle!(
    rw_bundle::ReweightingBundle.ReweightingSolverBundle,
    ens_bundle::Ensemble.EnsembleArrayBundle{T},
    nkappaT::Int,
    N_bs::Int,
    blk_size::Int,
    method::String,
    fname_OG::String,
    fname_P1::String,
    fname_P2::String,
    jobid::Union{Nothing, String}=nothing;
    rng_pool::SeedManager.RNGPool=rng_pool
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

    # Extract κ list from first ensemble array
    ens_array_first = ens_bundle.arrays[1]
    kappa = [ens.param.kappa for ens in ens_array_first.data]
    JobLoggerTools.assert_benji(nkappaT >= 2, "At least two κ points are required for reweighting curve", jobid)
    dkappaT = (kappa[end] - kappa[1]) / (nkappaT - 1)

    # Reference parameter set from the first ensemble (for β, ns, nt, etc.)
    param_ref = ens_bundle.arrays[1].data[1].param
    ns, nt, nf, beta, csw = param_ref.ns, param_ref.nt, param_ref.nf, param_ref.beta, param_ref.csw

    # Allocate storage for trMi values for all solvers
    trMi_all_bundle = [
        Vector{Vector{Float64}}(undef, rw.nconf_all)
        for rw in rw_bundle.solvers
    ]
    
    cumulants_OG_bs_all  = [ [Vector{Float64}(undef, N_bs) for i in 1:nkappaT] for j in 1:5 ]
    cumulants_P1_bs_all  = [ [Vector{Float64}(undef, N_bs) for i in 1:nkappaT] for j in 1:5 ]
    cumulants_P2_bs_all  = [ [Vector{Float64}(undef, N_bs) for i in 1:nkappaT] for j in 1:5 ]
    cumulants_OG_avg = [ Vector{Float64}(undef, nkappaT) for j in 1:5 ]
    cumulants_P1_avg = [ Vector{Float64}(undef, nkappaT) for j in 1:5 ]
    cumulants_P2_avg = [ Vector{Float64}(undef, nkappaT) for j in 1:5 ]

    idx_bundle_saved = nothing
    for i in 1:nkappaT
        # --- Compute κ_T and build parameter set ---
        kappaT = kappa[1] + ( i - 1 ) * dkappaT
        JobLoggerTools.println_benji("[$i/$nkappaT] kappaT = $kappaT", jobid)
        paramT = Ensemble.Params(ns, nt, nf, beta, csw, kappaT)

        # --- Reweighting factor computation ---
        ReweightingBundle.calc_w_all!(rw_bundle, paramT)

        # --- Volume normalization ---
        V = paramT.ns^3 * paramT.nt

        # --- Calculate trMi for all configurations ---
        for j in eachindex(rw_bundle.solvers)
            solver = rw_bundle.solvers[j]
            ens_array = solver.ens
            trMi_all = trMi_all_bundle[j]

            ii = 1
            for ens in ens_array.data
                for iconf in 1:ens.nconf
                    trMi_all[ii] = EnsembleUtils.trMiT(ens, paramT, iconf)
                    ii += 1
                end
            end
        end

        # --- Compute pbp cumulants ---
        if i == 1
            cumulants_OG_bs, cumulants_P1_bs, 
            cumulants_P2_bs, idx_bundle =
                CumulantsBundle.compute_cumulants_bundle(
                    N_bs, 
                    blk_size, 
                    method, 
                    V, 
                    trMi_all_bundle, 
                    rw_bundle, 
                    jobid; 
                    rng_pool=rng_pool
                )
            idx_bundle_saved = idx_bundle
        else
            cumulants_OG_bs, cumulants_P1_bs, 
            cumulants_P2_bs, _ =
                CumulantsBundle.compute_cumulants_bundle(
                    N_bs, 
                    blk_size, 
                    method, 
                    V, 
                    trMi_all_bundle, 
                    rw_bundle, 
                    jobid;
                    rng_pool=rng_pool,
                    idx_all=idx_bundle_saved[:all],
                    idx_lb=idx_bundle_saved[:lb],
                    idx_bc=idx_bundle_saved[:bc],
                    idx_ul=idx_bundle_saved[:ul]
                )
        end

        for j in 1:5
            cumulants_OG_bs_all[j][i] .= cumulants_OG_bs[j]
            cumulants_P1_bs_all[j][i] .= cumulants_P1_bs[j]
            cumulants_P2_bs_all[j][i] .= cumulants_P2_bs[j]
        end

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
        println(ofs_OG, @sprintf("%.12e                      ", kappaT),
            join([@sprintf("% .12e  %.12e  ", cumulants_OG[j], errors_OG[j])
                for j in eachindex(cumulants_OG)], ""))

        println(ofs_P1, @sprintf("%.12e                      ", kappaT),
            join([@sprintf("% .12e  %.12e  ", cumulants_P1[j], errors_P1[j])
                for j in eachindex(cumulants_P1)], ""))

        println(ofs_P2, @sprintf("%.12e                      ", kappaT),
            join([@sprintf("% .12e  %.12e  ", cumulants_P2[j], errors_P2[j])
                for j in eachindex(cumulants_P2)], ""))

        # Store bootstrap means into cumulants_OG_avg
        for j in 1:5
            cumulants_OG_avg[j][i] = cumulants_OG[j]
            cumulants_P1_avg[j][i] = cumulants_P1[j]
            cumulants_P2_avg[j][i] = cumulants_P2[j]
        end

    end

    # === Finding Transition Points ===
    println(ofs_OG,
        "\n\n# kappa_t           error                " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )
    println(ofs_P1,
        "\n\n# kappa_t           error                " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )
    println(ofs_P2,
        "\n\n# kappa_t           error                " *
        "cond                error                " *
        "susp                error                " *
        "skew                error                " *
        "kurt                error                " *
        "bind                error                "
    )

    kappa_vec = [kappa[1] + (i - 1) * dkappaT for i in 1:nkappaT]

    L_susp_OG, R_susp_OG, i_susp_OG = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_OG_avg; mode=:susp)
    L_skew_OG, R_skew_OG, i_skew_OG = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_OG_avg; mode=:skew)
    L_kurt_OG, R_kurt_OG, i_kurt_OG = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_OG_avg; mode=:kurt)
    L_susp_P1, R_susp_P1, i_susp_P1 = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_P1_avg; mode=:susp)
    L_skew_P1, R_skew_P1, i_skew_P1 = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_P1_avg; mode=:skew)
    L_kurt_P1, R_kurt_P1, i_kurt_P1 = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_P1_avg; mode=:kurt)
    L_susp_P2, R_susp_P2, i_susp_P2 = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_P2_avg; mode=:susp)
    L_skew_P2, R_skew_P2, i_skew_P2 = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_P2_avg; mode=:skew)
    L_kurt_P2, R_kurt_P2, i_kurt_P2 = Interpolation.preprobe_discrete_from_bundle(kappa_vec, cumulants_P2_avg; mode=:kurt)

    # ------------------------------------------------------------------------------------------------------

    JobLoggerTools.println_benji("cumulants_OG_bs_all susceptibility", jobid)

    vec_valids_susp =
        Interpolation.find_transition_cumulants(
            cumulants_OG_bs_all,
            kappa_vec,
            2,
            L_susp_OG, R_susp_OG, i_susp_OG,
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
            out_susp[i], errors_susp[i] = Bootstrap.bootstrap_average_error(v)
        end
    end
    
    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_OG, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_susp[1], errors_susp[1], out_susp[2], errors_susp[2],
        out_susp[3], errors_susp[3], out_susp[4], errors_susp[4],
        out_susp[5], errors_susp[5], out_susp[6], errors_susp[6], "susp")

    # ------------------------------------------------------------------------------------------------------

    vec_valids_skew =
        Interpolation.find_transition_cumulants(
            cumulants_OG_bs_all,
            kappa_vec,
            3,
            L_skew_OG, R_skew_OG, i_skew_OG,
            jobid
        )

    JobLoggerTools.println_benji("cumulants_OG_bs_all skewness", jobid)

    out_skew    = Vector{eltype(kappa_vec)}(undef, 6)
    errors_skew = Vector{eltype(kappa_vec)}(undef, 6)
    for i in 1:6
        v = vec_valids_skew[i]
        if isempty(v)
            out_skew[i]    = eltype(kappa_vec)(NaN)
            errors_skew[i] = eltype(kappa_vec)(NaN)
        else
            out_skew[i], errors_skew[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_OG, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_skew[1], errors_skew[1], out_skew[2], errors_skew[2],
        out_skew[3], errors_skew[3], out_skew[4], errors_skew[4],
        out_skew[5], errors_skew[5], out_skew[6], errors_skew[6], "skew")

    # ------------------------------------------------------------------------------------------------------

    vec_valids_kurt =
        Interpolation.find_transition_cumulants(
            cumulants_OG_bs_all,
            kappa_vec,
            4,
            L_kurt_OG, R_kurt_OG, i_kurt_OG,
            jobid
        )

    JobLoggerTools.println_benji("cumulants_OG_bs_all kurtosis", jobid)

    out_kurt    = Vector{eltype(kappa_vec)}(undef, 6)
    errors_kurt = Vector{eltype(kappa_vec)}(undef, 6)
    for i in 1:6
        v = vec_valids_kurt[i]
        if isempty(v)
            out_kurt[i]    = eltype(kappa_vec)(NaN)
            errors_kurt[i] = eltype(kappa_vec)(NaN)
        else
            out_kurt[i], errors_kurt[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_OG, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_kurt[1], errors_kurt[1], out_kurt[2], errors_kurt[2],
        out_kurt[3], errors_kurt[3], out_kurt[4], errors_kurt[4],
        out_kurt[5], errors_kurt[5], out_kurt[6], errors_kurt[6], "kurt")

    # ------------------------------------------------------------------------------------------------------

    JobLoggerTools.println_benji("cumulants_P1_bs_all susceptibility", jobid)

    vec_valids_susp_P1 =
        Interpolation.find_transition_cumulants(
            cumulants_P1_bs_all,
            kappa_vec,
            2,
            L_susp_P1, R_susp_P1, i_susp_P1,
            jobid
        )

    for i in 1:6
        v = vec_valids_susp_P1[i]
        if isempty(v)
            out_susp[i]    = eltype(kappa_vec)(NaN)
            errors_susp[i] = eltype(kappa_vec)(NaN)
        else
            out_susp[i], errors_susp[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_P1, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_susp[1], errors_susp[1], out_susp[2], errors_susp[2],
        out_susp[3], errors_susp[3], out_susp[4], errors_susp[4],
        out_susp[5], errors_susp[5], out_susp[6], errors_susp[6], "susp")

    # ------------------------------------------------------------------------------------------------------

    JobLoggerTools.println_benji("cumulants_P1_bs_all skewness", jobid)

    vec_valids_skew_P1 =
        Interpolation.find_transition_cumulants(
            cumulants_P1_bs_all,
            kappa_vec,
            3,
            L_skew_P1, R_skew_P1, i_skew_P1,
            jobid
        )

    for i in 1:6
        v = vec_valids_skew_P1[i]
        if isempty(v)
            out_skew[i]    = eltype(kappa_vec)(NaN)
            errors_skew[i] = eltype(kappa_vec)(NaN)
        else
            out_skew[i], errors_skew[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_P1, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_skew[1], errors_skew[1], out_skew[2], errors_skew[2],
        out_skew[3], errors_skew[3], out_skew[4], errors_skew[4],
        out_skew[5], errors_skew[5], out_skew[6], errors_skew[6], "skew")

    # ------------------------------------------------------------------------------------------------------

    JobLoggerTools.println_benji("cumulants_P1_bs_all kurtosis", jobid)

    vec_valids_kurt_P1 =
        Interpolation.find_transition_cumulants(
            cumulants_P1_bs_all,
            kappa_vec,
            4,
            L_kurt_P1, R_kurt_P1, i_kurt_P1,
            jobid
        )

    for i in 1:6
        v = vec_valids_kurt_P1[i]
        if isempty(v)
            out_kurt[i]    = eltype(kappa_vec)(NaN)
            errors_kurt[i] = eltype(kappa_vec)(NaN)
        else
            out_kurt[i], errors_kurt[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_P1, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_kurt[1], errors_kurt[1], out_kurt[2], errors_kurt[2],
        out_kurt[3], errors_kurt[3], out_kurt[4], errors_kurt[4],
        out_kurt[5], errors_kurt[5], out_kurt[6], errors_kurt[6], "kurt")

    # ------------------------------------------------------------------------------------------------------

    JobLoggerTools.println_benji("cumulants_P2_bs_all susceptibility", jobid)

    vec_valids_susp_P2 =
        Interpolation.find_transition_cumulants(
            cumulants_P2_bs_all,
            kappa_vec,
            2,
            L_susp_P2, R_susp_P2, i_susp_P2,
            jobid
        )

    for i in 1:6
        v = vec_valids_susp_P2[i]
        if isempty(v)
            out_susp[i]    = eltype(kappa_vec)(NaN)
            errors_susp[i] = eltype(kappa_vec)(NaN)
        else
            out_susp[i], errors_susp[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_P2, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_susp[1], errors_susp[1], out_susp[2], errors_susp[2],
        out_susp[3], errors_susp[3], out_susp[4], errors_susp[4],
        out_susp[5], errors_susp[5], out_susp[6], errors_susp[6], "susp")

    # ------------------------------------------------------------------------------------------------------

    JobLoggerTools.println_benji("cumulants_P2_bs_all skewness", jobid)

    vec_valids_skew_P2 =
        Interpolation.find_transition_cumulants(
            cumulants_P2_bs_all,
            kappa_vec,
            3,
            L_skew_P2, R_skew_P2, i_skew_P2,
            jobid
        )

    for i in 1:6
        v = vec_valids_skew_P2[i]
        if isempty(v)
            out_skew[i]    = eltype(kappa_vec)(NaN)
            errors_skew[i] = eltype(kappa_vec)(NaN)
        else
            out_skew[i], errors_skew[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_P2, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_skew[1], errors_skew[1], out_skew[2], errors_skew[2],
        out_skew[3], errors_skew[3], out_skew[4], errors_skew[4],
        out_skew[5], errors_skew[5], out_skew[6], errors_skew[6], "skew")

    # ------------------------------------------------------------------------------------------------------

    JobLoggerTools.println_benji("cumulants_P2_bs_all kurtosis", jobid)

    vec_valids_kurt_P2 =
        Interpolation.find_transition_cumulants(
            cumulants_P2_bs_all,
            kappa_vec,
            4,
            L_kurt_P2, R_kurt_P2, i_kurt_P2,
            jobid
        )

    for i in 1:6
        v = vec_valids_kurt_P2[i]
        if isempty(v)
            out_kurt[i]    = eltype(kappa_vec)(NaN)
            errors_kurt[i] = eltype(kappa_vec)(NaN)
        else
            out_kurt[i], errors_kurt[i] = Bootstrap.bootstrap_average_error(v)
        end
    end

    # ------------------------------------------------------------------------------------------------------

    @printf(ofs_P2, "%.12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  % .12e  %.12e  %s\n",
        out_kurt[1], errors_kurt[1], out_kurt[2], errors_kurt[2],
        out_kurt[3], errors_kurt[3], out_kurt[4], errors_kurt[4],
        out_kurt[5], errors_kurt[5], out_kurt[6], errors_kurt[6], "kurt")

    # ------------------------------------------------------------------------------------------------------
        
    close(ofs_OG)
    close(ofs_P1)
    close(ofs_P2)

end

end  # module ReweightingCurveBundle