# ============================================================================
# src/DeborahCore/DeborahRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DeborahRunner

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.TOMLLogger
import ..Sarah.NameParser
import ..Sarah.SeedManager
import ..Sarah.DatasetPartitioner
import ..Sarah.XYInfoGenerator
import ..Sarah.DataLoader
import ..Sarah.Bootstrap
import ..Sarah.Jackknife
import ..Sarah.BlockSizeSuggester
import ..Sarah.AvgErrFormatter
import ..Sarah.BootstrapDataInit
import ..Sarah.BootstrapRunner
import ..Sarah.SummaryFormatter
import ..Sarah.SummaryCollector
import ..TOMLConfigDeborah
import ..PathConfigBuilderDeborah
import ..DatasetPartitionerDeborah
import ..XYMLInfoGenerator
import ..XYMLVectorizer
import ..FeaturePipeline
import ..MLInputPreparer
import ..BaselineSequence
import ..MLSequenceLasso
import ..MLSequenceRidge
import ..MLSequenceLightGBM
import ..MLSequenceMiddleGBM
import ..MLSequencePyCallLightGBM
import ..MLSequence
import ..SummaryWriterDeborah
import ..ResultPrinterDeborah

"""
    run_Deborah(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Top-level function to execute the [`Deborah.DeborahCore`](@ref) pipeline:
1. Parse configuration with [`Deborah.DeborahCore.TOMLConfigDeborah.parse_full_config_Deborah`](@ref)
2. Build path with [`Deborah.DeborahCore.PathConfigBuilderDeborah.build_path_config_Deborah`](@ref) and partition data with [`Deborah.DeborahCore.DatasetPartitionerDeborah.partition_dataset`](@ref)
3. Prepare ML inputs with [`Deborah.DeborahCore.MLInputPreparer.prepare_ML_inputs`](@ref)
4. Suggest optimal block size with [`Deborah.Sarah.BlockSizeSuggester.suggest_opt_block_sizes`](@ref)
5. Run model-specific sequence with [`Deborah.DeborahCore.MLSequence.ml_sequence`](@ref) / [`Deborah.DeborahCore.BaselineSequence.baseline_sequence`](@ref) / [`Deborah.DeborahCore.BaselineSequence.random_sequence`](@ref)
6. Generate trace data with [`Deborah.Sarah.BootstrapDataInit.build_trace_data`](@ref)
7. Initialize and run bootstrap with [`Deborah.Sarah.BootstrapDataInit.init_bootstrap_data`](@ref), [`Deborah.Sarah.SeedManager.setup_rng_pool`](@ref) and [`Deborah.Sarah.BootstrapRunner.run_bootstrap!`](@ref)
8. Collect and save summary with [`Deborah.Sarah.SummaryCollector.collect_summaries_Deborah`](@ref) and [`Deborah.DeborahCore.SummaryWriterDeborah.write_summary_file_Deborah`](@ref)
9. Optionally print result summary with [`Deborah.DeborahCore.ResultPrinterDeborah.print_summary_results_Deborah`](@ref) if `jobid === nothing`.

All steps are logged and timed.
"""
function run_Deborah(
    toml_path::String,
    jobid::Union{Nothing, String}=nothing
)::Nothing

    # -------------------------------------------------------------------------
    # [1] Load and parse full configuration
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("parse_full_config_Deborah() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        cfg = TOMLConfigDeborah.parse_full_config_Deborah(
            toml_path, 
            jobid
        )
    end

    X_list = cfg.data.model == "Baseline" ? [cfg.data.Y] : cfg.data.X

    # -------------------------------------------------------------------------
    # [2] Build file path configuration
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("build_path_config_Deborah() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        paths = PathConfigBuilderDeborah.build_path_config_Deborah(
            cfg.data, 
            cfg.abbrev, 
            X_list, 
            jobid
        )
    end

    # -------------------------------------------------------------------------
    # [3] Partition dataset (load raw data & divide into partitions)
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("partition_dataset() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        partition = DatasetPartitionerDeborah.partition_dataset(
            paths.path, 
            cfg.data, 
            jobid
        )
    end

    # -------------------------------------------------------------------------
    # [4] Prepare machine learning inputs (X, Y, configs)
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("prepare_ML_inputs() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ML_inputs = MLInputPreparer.prepare_ML_inputs(
            partition, 
            X_list, 
            cfg.data.Y, 
            paths;
            jobid=jobid,
            read_column_X=cfg.data.read_column_X,
            read_column_Y=cfg.data.read_column_Y,
            index_column=cfg.data.index_column,
            dump=cfg.data.dump_X
        )
    end

    # -------------------------------------------------------------------------
    # [5] Suggest optimal block size from trace data
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("suggest_opt_block_sizes() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        opt_blk_size = BlockSizeSuggester.suggest_opt_block_sizes(
            partition,
            cfg.bs.blk_size; 
            min_block=1,
        )
    end

    # -------------------------------------------------------------------------
    # [6] Run model-specific sequence and obtain Y_mats
    # -------------------------------------------------------------------------
    if cfg.data.model == "Baseline"
        JobLoggerTools.log_stage_benji("baseline_sequence() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_mats = BaselineSequence.baseline_sequence(
                ML_inputs,
                partition, 
                paths, 
                jobid;
                read_column_Y=cfg.data.read_column_Y,
                dump=cfg.data.dump_X
            )
        end

    elseif cfg.data.model == "Random"
        JobLoggerTools.log_stage_benji("random_sequence() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_mats = BaselineSequence.random_sequence(
                ML_inputs,
                partition, 
                paths, 
                cfg.bs.ranseed, 
                opt_blk_size[:all],
                jobid;
                read_column_Y=cfg.data.read_column_Y,
                dump=cfg.data.dump_X
            )
        end

    elseif cfg.data.model in Set(["LightGBM", "MiddleGBM", "PyGBM", "Lasso", "Ridge"])
        JobLoggerTools.log_stage_benji("ml_sequence() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_mats = MLSequence.ml_sequence(
                cfg.data.model,
                ML_inputs,
                partition,
                paths,
                X_list,
                jobid
            )
        end

    else
        JobLoggerTools.error_benji("Unsupported model type: $(cfg.data.model)", jobid)
        return
    end

    # -------------------------------------------------------------------------
    # [7] Build trace data from predictions
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("build_trace_data() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        trace_data = BootstrapDataInit.build_trace_data(
            Y_mats, 
            ML_inputs.Y_df,
            partition.N_cnf, 
            partition.N_src, 
            cfg.data.read_column_Y
        )
    end

    # -------------------------------------------------------------------------
    # [8] Initialize bootstrap container and RNG pool
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("init_bootstrap_data() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        bootstrap_data = BootstrapDataInit.init_bootstrap_data(
            cfg.bs.N_bs, 
            Float64
        )
    end

    JobLoggerTools.log_stage_benji("setup_rng_pool() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        rng_pool = SeedManager.setup_rng_pool(
            cfg.bs.ranseed, 
            jobid
        )
    end

    # -------------------------------------------------------------------------
    # [9] Run bootstrap resampling
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("run_bootstrap!() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        BootstrapRunner.run_bootstrap!(
            bootstrap_data, 
            trace_data,
            partition, 
            cfg.bs.N_bs, 
            rng_pool,
            opt_blk_size, 
            cfg.bs.method,
            jobid
        )
    end

    # -------------------------------------------------------------------------
    # [10] Collect summary (JK + BS)
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("collect_summaries_Deborah() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        summary_jackknife, summary_bootstrap =
            SummaryCollector.collect_summaries_Deborah(
                trace_data, 
                bootstrap_data, 
                opt_blk_size[:all] # cfg.jk.bin_size
            )
    end

    # -------------------------------------------------------------------------
    # [11] Write summary output files
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("write_summary_file_Deborah() ::", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        SummaryWriterDeborah.write_summary_file_Deborah(
            cfg.data, 
            summary_jackknife, 
            summary_bootstrap,
            paths.overall_name, 
            paths.analysis_dir,
            jobid
        )
    end

    # -------------------------------------------------------------------------
    # [12] Optional: Print summary if no jobid (interactive)
    # -------------------------------------------------------------------------
    if isnothing(jobid)
        JobLoggerTools.log_stage_benji("print_summary_results_Deborah() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            ResultPrinterDeborah.print_summary_results_Deborah(
                cfg.data, 
                trace_data, 
                bootstrap_data,
                opt_blk_size[:all], # cfg.jk.bin_size, 
                jobid
            )
        end
    end

    # -------------------------------------------------------------------------
    # [13] All done
    # -------------------------------------------------------------------------
    JobLoggerTools.log_stage_benji("Done", jobid)
end

end  # module DeborahRunner