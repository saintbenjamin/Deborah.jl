# ============================================================================
# src/Esther/EstherRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module EstherRunner

import ..Sarah.JobLoggerTools
import ..Sarah.TOMLLogger
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.SeedManager
import ..Sarah.DatasetPartitioner
import ..Sarah.DataLoader
import ..Sarah.Bootstrap
import ..Sarah.Jackknife
import ..Sarah.BlockSizeSuggester
import ..Sarah.BootstrapDataInit
import ..Sarah.BootstrapRunner
import ..Sarah.AvgErrFormatter
import ..Sarah.SummaryFormatter
import ..Sarah.SummaryCollector
import ..TOMLConfigEsther
import ..DatasetPartitionerEsther
import ..PathConfigBuilderEsther
import ..TraceDataLoader
import ..TraceRescaler
import ..SingleQMoment
import ..SingleCumulant
import ..QMomentCalculator
import ..BootstrapDerivedCalculator
import ..JackknifeRunner
import ..SummaryWriterEsther
import ..ResultPrinterEsther

"""
    run_Esther(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Main execution pipeline for the [`Deborah.Esther`](@ref) module.

This function runs the full workflow of data parsing, trace loading, observable computation, 
resampling (bootstrap and jackknife), and summary output generation based on a [`TOML`](https://toml.io/en/) configuration.

# Arguments
- `toml_path::String`: Path to the configuration file ([`TOML`](https://toml.io/en/) format) containing all metadata for the job.
- `jobid::Union{Nothing, String}`: Optional identifier used for logging and tracking. If `nothing`, simplified logging is used.

# Behavior
- Loads configuration with [`Deborah.Esther.TOMLConfigEsther.parse_full_config_Esther`](@ref) and path settings with [`Deborah.Esther.PathConfigBuilderEsther.build_path_config_Esther`](@ref).
- Loads trace data with [`Deborah.Esther.TraceDataLoader.load_trace_data`](@ref), infers the partition information with [`Deborah.Esther.DatasetPartitionerEsther.infer_partition_info_from_trace`](@ref) and rescales according to lattice parameters with [`Deborah.Esther.TraceRescaler.rescale_all_traces`](@ref).
- Computes moment observables from trace data with [`Deborah.Esther.QMomentCalculator.compute_Q_moments`](@ref).
- Suggests optimal block size for jackknife/bootstrapping with [`Deborah.Sarah.BlockSizeSuggester.suggest_opt_block_sizes`](@ref).
- Initializes bootstrap and jackknife routines with [`Deborah.Sarah.BootstrapDataInit.init_bootstrap_data_cumulant`](@ref) and [`Deborah.Sarah.BootstrapRunner.run_bootstrap!`](@ref).
- Computes derived bootstrap observables with [`Deborah.Esther.BootstrapDerivedCalculator.compute_bootstrap_derived!`](@ref) and jackknife observables with [`Deborah.Esther.JackknifeRunner.compute_jackknife_observables`](@ref).
- Collects statistical summaries with [`Deborah.Sarah.SummaryCollector.collect_summaries_Esther`](@ref) and writes them to disk with [`Deborah.Esther.SummaryWriterEsther.write_summary_file_Esther`](@ref).
- Optionally prints a summary to stdout if `jobid` is `nothing` with [`Deborah.Esther.ResultPrinterEsther.print_summary_results_Esther`](@ref).

# Side Effects
- Writes summary files and logs to disk.
- Modifies internal RNG state and global output structures.

# Notes
This is the central driver of a full [`Deborah.Esther`](@ref) job. All other functions and modules must be correctly configured.
"""
function run_Esther(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)

    JobLoggerTools.log_stage_benji("parse_full_config_Esther() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        cfg = 
        TOMLConfigEsther.parse_full_config_Esther(
            toml_path, 
            jobid
        )
    end

    JobLoggerTools.log_stage_benji("build_path_config_Esther() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        paths = 
        PathConfigBuilderEsther.build_path_config_Esther(
            cfg.data, 
            cfg.abbrev, 
            jobid
        )
    end

    JobLoggerTools.log_stage_benji("load_trace_data() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        trace_data = 
        TraceDataLoader.load_trace_data(
            paths, 
            jobid
        )
    end

    JobLoggerTools.log_stage_benji("infer_partition_info_from_trace() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        partition = 
        DatasetPartitionerEsther.infer_partition_info_from_trace(
            trace_data
        )
    end

    JobLoggerTools.log_stage_benji("rescale_all_traces() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        trace_rscl = 
        TraceRescaler.rescale_all_traces(
            trace_data, 
            cfg.im.kappa, 
            TOMLConfigEsther.get_LatVol(cfg.im)
        )
    end

    JobLoggerTools.log_stage_benji("compute_Q_moments() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        Q_moment = 
        QMomentCalculator.compute_Q_moments(
            trace_rscl, 
            cfg.im.nf
        )
    end

    JobLoggerTools.log_stage_benji("suggest_opt_block_sizes() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        opt_blk_size = BlockSizeSuggester.suggest_opt_block_sizes(
            partition,
            cfg.bs.blk_size; 
            min_block=1,
        )
    end

    JobLoggerTools.log_stage_benji("init_bootstrap_data() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        bootstrap_data = 
        BootstrapDataInit.init_bootstrap_data_cumulant(
            cfg.bs.N_bs
        )
    end

    JobLoggerTools.log_stage_benji("setup_rng_pool() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        rng_pool = 
        SeedManager.setup_rng_pool(
            cfg.bs.ranseed, 
            jobid
        )
    end    

    JobLoggerTools.log_stage_benji("run_bootstrap!() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        BootstrapRunner.run_bootstrap!(
            bootstrap_data, 
            trace_data, 
            Q_moment, 
            partition, 
            cfg.bs.N_bs, 
            rng_pool, 
            opt_blk_size, 
            cfg.bs.method,
            jobid
        )
    end

    JobLoggerTools.log_stage_benji("compute_bootstrap_derived!() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        BootstrapDerivedCalculator.compute_bootstrap_derived!(
            bootstrap_data, 
            TOMLConfigEsther.get_LatVol(cfg.im)
        )
    end

    JobLoggerTools.log_stage_benji("compute_jackknife_observables() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        jackknife_data = Dict{Symbol, Any}()
        jackknife_data[:mean] = 
        JackknifeRunner.compute_jackknife_observables(
            Q_moment, 
            TOMLConfigEsther.get_LatVol(cfg.im), 
            opt_blk_size[:all] # cfg.jk.bin_size
        )
    end

    JobLoggerTools.log_stage_benji("collect_summaries_Esther() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        summary_jackknife, 
        summary_bootstrap = 
        SummaryCollector.collect_summaries_Esther(
            trace_data, 
            Q_moment, 
            jackknife_data, 
            bootstrap_data, 
            opt_blk_size[:all] # cfg.jk.bin_size
        )
    end

    JobLoggerTools.log_stage_benji("write_summary_file_Esther() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        SummaryWriterEsther.write_summary_file_Esther(
            cfg.data, 
            summary_jackknife, 
            summary_bootstrap, 
            paths.overall_name, 
            paths.my_tex_dir,
            jobid
        )
    end

    if isnothing(jobid)
        JobLoggerTools.log_stage_benji("print_summary_results_Esther() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            ResultPrinterEsther.print_summary_results_Esther(
                cfg.data, 
                trace_data, 
                jackknife_data, 
                bootstrap_data, 
                opt_blk_size[:all], # cfg.jk.bin_size
                jobid
            )
        end
    end

    JobLoggerTools.log_stage_benji("Done", jobid)
    
end

end  # module EstherRunner