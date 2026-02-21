# ============================================================================
# src/Miriam/MiriamRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module MiriamRunner

import ..Sarah.JobLoggerTools
import ..Sarah.SeedManager
import ..Sarah.TOMLLogger
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.DataLoader
import ..Sarah.Bootstrap
import ..Sarah.Jackknife
import ..Sarah.Bootstrap
import ..TOMLConfigMiriam
import ..PathConfigBuilderMiriam
import ..MultiEnsembleLoader
import ..Ensemble
import ..EnsembleUtils
import ..FileIO
import ..Cumulants
import ..Interpolation
import ..WriteJKOutput
import ..Reweighting
import ..ReweightingCurve
import ..ReweightingBundle
import ..CumulantsBundleUtils
import ..CumulantsBundle
import ..WriteBSOutput
import ..ReweightingCurveBundle

"""
    run_Miriam(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Main pipeline to run the [`Deborah.Miriam`](@ref) module using configuration from a [`TOML`](https://toml.io/en/) file.

# Arguments
- `toml_path::String`: Path to the configuration `.toml` file, containing information on ensemble directories, solver settings, and observable definitions.
- `jobid::Union{Nothing, String}`: Optional identifier string for logging or parallel job tracking. If `nothing`, no job ID is used.

# Description
Executes the full [`Deborah.Miriam`](@ref) pipeline as follows:

1. Parses the [`TOML`](https://toml.io/en/) config ([`Deborah.Miriam.TOMLConfigMiriam.parse_full_config_Miriam`](@ref), [`Deborah.Miriam.PathConfigBuilderMiriam.build_path_config_Miriam`](@ref)) 
2. Constructs all ensemble sets including `Y_BC`/`Y_UL` variations ([`Deborah.Miriam.FileIO.read_ensemble_array_bundle`](@ref)).
3. Computes jackknife-based cumulants for each observable from the original ensemble ([`Deborah.Miriam.WriteJKOutput.write_jk_traces`](@ref), [`Deborah.Miriam.WriteJKOutput.write_jk_moments`](@ref), [`Deborah.Miriam.WriteJKOutput.write_jk_cumulants`](@ref)).
4. Computes block-bootstrap-based cumulants for each observable from the original ensemble ([`Deborah.Miriam.WriteBSOutput.write_bs_traces`](@ref), [`Deborah.Miriam.WriteBSOutput.write_bs_moments`](@ref), [`Deborah.Miriam.WriteBSOutput.write_bs_cumulants`](@ref)).
5. Runs reweighting solver on the ensemble bundle ([`Deborah.Miriam.ReweightingBundle.calc_f_all!`](@ref)). 
6. Calculate jackknife-based reweighted cumulants ([`Deborah.Miriam.ReweightingCurve.reweighting_curve!`](@ref)).
7. Computes bootstrap cumulants from reweighted data and saves the full result set to disk ([`Deborah.Miriam.ReweightingCurveBundle.reweighting_curve_bundle!`](@ref)).

Returns `nothing`. All results are stored to output files specified in the [`TOML`](https://toml.io/en/) config.
"""
function run_Miriam(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)::Nothing

    JobLoggerTools.log_stage_benji("parse_full_config_Miriam() ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        cfg = TOMLConfigMiriam.parse_full_config_Miriam(
            toml_path, jobid
        )
    end

    JobLoggerTools.log_stage_benji("build_path_config_Miriam() ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        paths = PathConfigBuilderMiriam.build_path_config_Miriam(
            cfg.data, cfg.abbrev, jobid
        )
    end

    JobLoggerTools.log_stage_benji("read_ensemble_array_bundle() ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ens_array_bundle, key_list = FileIO.read_ensemble_array_bundle(
            cfg, paths, jobid
        )
    end

    # Write jackknife trace outputs from original ensemble
    JobLoggerTools.log_stage_benji("write_jk_traces() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        WriteJKOutput.write_jk_traces(
            ens_array_bundle.arrays[1], 
            cfg.jk.bin_size,
            paths.fname.trc_all_jk 
        )
    end

    # Write jackknife moments from original ensemble
    JobLoggerTools.log_stage_benji("write_jk_moments() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        WriteJKOutput.write_jk_moments(
            ens_array_bundle.arrays[1], 
            cfg.jk.bin_size,
            paths.fname.mmt_all_jk 
        )
    end

    # Write jackknife cumulants from original ensemble
    JobLoggerTools.log_stage_benji("write_jk_cumulants() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        WriteJKOutput.write_jk_cumulants(
            ens_array_bundle.arrays[1], 
            cfg.jk.bin_size,
            paths.fname.pnt_all_jk 
        )
    end
   
    JobLoggerTools.log_stage_benji("setup_rng_pool() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        rng_pool = SeedManager.setup_rng_pool(
            cfg.bs.ranseed, jobid
        )
    end

    JobLoggerTools.log_stage_benji("write_bs_traces() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        WriteBSOutput.write_bs_traces(
            ens_array_bundle, 
            cfg.bs.N_bs, cfg.bs.blk_size, cfg.bs.method,
            paths.fname.trc_all_bs,
            paths.fname.trc_P1_bs, 
            paths.fname.trc_P2_bs,
            jobid; 
            rng_pool=rng_pool
        )
    end

    JobLoggerTools.log_stage_benji("write_bs_moments() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        WriteBSOutput.write_bs_moments(
            ens_array_bundle, 
            cfg.bs.N_bs, cfg.bs.blk_size, cfg.bs.method,
            paths.fname.mmt_all_bs,
            paths.fname.mmt_P1_bs, 
            paths.fname.mmt_P2_bs,
            jobid; 
            rng_pool=rng_pool
        )
    end

    JobLoggerTools.log_stage_benji("write_bs_cumulants() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        WriteBSOutput.write_bs_cumulants(
            ens_array_bundle, 
            cfg.bs.N_bs, cfg.bs.blk_size, cfg.bs.method,
            paths.fname.pnt_all_bs,
            paths.fname.pnt_P1_bs, 
            paths.fname.pnt_P2_bs,
            jobid; 
            rng_pool=rng_pool
        )
    end

    # Construct reweighting solver bundle for full three-array bundle
    JobLoggerTools.log_stage_benji("ReweightingSolverBundle() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        rw_bundle = ReweightingBundle.ReweightingSolverBundle(
            ens_array_bundle, 
            cfg.solver.maxiter, cfg.solver.eps
        )
    end

    JobLoggerTools.log_stage_benji("calc_f_all!() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ReweightingBundle.calc_f_all!(
            rw_bundle, paths.info_file, jobid
        )
    end

    # Compute reweighting curve from original ensemble
    JobLoggerTools.log_stage_benji("reweighting_curve!() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ReweightingCurve.reweighting_curve!(
            rw_bundle.solvers[1], 
            ens_array_bundle.arrays[1], 
            cfg.traj.nkappaT, 
            cfg.jk.bin_size,
            paths.fname.rwt_all_jk,
            jobid
        )
    end

    # Write reweighting results for P1 and P2 observables from bundle
    JobLoggerTools.log_stage_benji("reweighting_curve_bundle!() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ReweightingCurveBundle.reweighting_curve_bundle!(
            rw_bundle, 
            ens_array_bundle, 
            cfg.traj.nkappaT, 
            cfg.bs.N_bs, cfg.bs.blk_size, cfg.bs.method,
            paths.fname.rwt_all_bs,
            paths.fname.rwt_P1_bs,  
            paths.fname.rwt_P2_bs,
            jobid; rng_pool=rng_pool
        )
    end

    JobLoggerTools.log_stage_benji("Done", jobid)

end

end  # module MiriamRunner