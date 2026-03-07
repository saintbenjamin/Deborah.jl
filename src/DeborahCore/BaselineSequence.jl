# ============================================================================
# src/DeborahCore/BaselineSequence.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BaselineSequence

import ..Random
import ..Statistics
import ..Distributions

import ..Sarah.JobLoggerTools
import ..Sarah.SeedManager
import ..Sarah.DatasetPartitioner
import ..Sarah.XYInfoGenerator
import ..Sarah.Jackknife
import ..Sarah.AvgErrFormatter
import ..PathConfigBuilderDeborah
import ..XYMLInfoGenerator
import ..XYMLVectorizer
import ..MLInputPreparer

"""
    baseline_sequence(
        ML_inputs::MLInputPreparer.MLInputBundle,
        partition::DatasetPartitioner.DatasetPartitionInfo, 
        paths::PathConfigBuilderDeborah.DeborahPathConfig, 
        jobid::Union{Nothing, String}=nothing;
        read_column_Y::Int, 
        dump::Bool=true
    ) -> Dict{Symbol, Matrix{T}} where T<:Real

Construct a baseline prediction sequence where inputs and targets are identical,  
i.e., ``X = Y = Y^P``, using the specified observable column from the data.

# Arguments
- [`ML_inputs::MLInputPreparer.MLInputBundle`](@ref Deborah.DeborahCore.MLInputPreparer.MLInputBundle)
    Struct containing:
    - `Y_df::Matrix{T}`: Full observable matrix (``N_\\text{cnf} \\times N_\\text{src}``).
    - `conf_arr::Vector{Int}`: Configuration index array.

- [`partition::DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo)
    Contains partition indices for `:lb`, `:tr`, `:bc`, and `:ul`.

- [`paths::DeborahPathConfig`](@ref Deborah.DeborahCore.PathConfigBuilderDeborah.DeborahPathConfig)
    Contains output directory information for saving results.

- `jobid::Union{Nothing, String}`  
    Optional identifier for logging or batch tracking.

# Keyword Arguments
- `read_column_Y::Int`  
    ``1``-based column index to extract from `Y_df`.

- `dump::Bool = true`  
    If true, saves all generated `Y_*` and `YP_*` vectors to disk.

# Returns
- `Dict{Symbol, Matrix{T}}`  
    Dictionary containing entries like `:Y_bc`, `:YP_bc`, etc.,  
    where each is a ``1``-column matrix (``N \\times 1``) corresponding to a partition.

# Behavior
- For each partition tag (e.g., `:tr`, `:ul`, etc.), generates:
    - `Y_tag`: extracted from `Y_df[:, read_column_Y]`
    - `YP_tag`: identical copy of `Y_tag`
- Does not perform any training or randomization -- this is a deterministic identity baseline.
- Outputs are structurally identical to those of full ML pipelines,  
  making them suitable for direct performance comparisons.

# Notes
- Internally calls [`Deborah.Sarah.XYInfoGenerator.gen_X_info`](@ref), [`Deborah.DeborahCore.XYMLInfoGenerator.gen_XY_ML_info`](@ref),  
  [`Deborah.DeborahCore.XYMLVectorizer.gen_XY_ML`](@ref), and [`Deborah.DeborahCore.XYMLVectorizer.mat_XY_ML`](@ref), using shared logic from ML pipelines.
"""
function baseline_sequence(
    ML_inputs::MLInputPreparer.MLInputBundle,
    partition::DatasetPartitioner.DatasetPartitionInfo, 
    paths::PathConfigBuilderDeborah.DeborahPathConfig, 
    jobid::Union{Nothing, String}=nothing;
    read_column_Y::Int, 
    dump::Bool=true
)

    Y_df = ML_inputs.Y_df
    conf_arr = ML_inputs.conf_arr

    do_bias_correction = !isempty(ML_inputs.Y_tr_vec) && !isempty(ML_inputs.Y_bc_vec)
    no_bias_correction = !isempty(ML_inputs.Y_tr_vec) &&  isempty(ML_inputs.Y_bc_vec)
    skip_all_trainings =  isempty(ML_inputs.Y_tr_vec)

    JobLoggerTools.log_stage_sub1_benji("Baseline Predictor Sequence (X = Y = YP)", jobid)

    Y_info = XYInfoGenerator.gen_X_info(
        Y_df, 
        partition.N_cnf, 
        partition.N_src, 
        read_column_Y
    )

    # Split into sets
    lb_info, tr_info, bc_info, ul_info,
    lb_conf, tr_conf, bc_conf, ul_conf =
        XYMLInfoGenerator.gen_XY_ML_info(
            Y_info, 
            conf_arr, 
            partition.lb_idx, 
            partition.tr_idx, 
            partition.bc_idx, 
            partition.ul_idx,
            partition.N_lb, 
            partition.N_tr, 
            partition.N_bc_persrc, 
            partition.N_ul_persrc,
            "X1_info_", 
            paths.overall_name, 
            paths.analysis_dir, 
            read_column_Y; 
            dump=dump, 
            jobid=jobid
        )

    sets = Dict(
        :lb => (lb_info, lb_conf),
        :tr => (tr_info, tr_conf),
        :bc => (bc_info, bc_conf),
        :ul => (ul_info, ul_conf)
    )

    Y_vecs  = Dict{Symbol, Vector{Real}}()
    YP_vecs = Dict{Symbol, Vector{Real}}()

    JobLoggerTools.println_benji("Saving files in $(paths.analysis_dir)", jobid)

    for (tag, (info, conf)) in sets
        Y_vecs[tag]  = XYMLVectorizer.gen_XY_ML(
            info, 
            read_column_Y, 
            conf, 
            "Y_$tag",  
            paths.overall_name, 
            paths.analysis_dir
        )
    end
    empty_info = Array{Float64, 3}(undef, 0, 0, 0)
    if do_bias_correction
        for (tag, (info, conf)) in sets
            if tag != :lb
                YP_vecs[tag] = XYMLVectorizer.gen_XY_ML(
                    info, 
                    read_column_Y, 
                    conf, 
                    "YP_$tag", 
                    paths.overall_name, 
                    paths.analysis_dir
                )
            else
                YP_vecs[tag] = XYMLVectorizer.gen_XY_ML(
                    empty_info,
                    read_column_Y, 
                    conf, 
                    "YP_$tag", 
                    paths.overall_name, 
                    paths.analysis_dir
                )
            end
        end
    elseif no_bias_correction
        for (tag, (info, conf)) in sets
            if !(tag in (:lb, :bc))
                YP_vecs[tag] = XYMLVectorizer.gen_XY_ML(
                    info, 
                    read_column_Y, 
                    conf, 
                    "YP_$tag", 
                    paths.overall_name, 
                    paths.analysis_dir
                )
            else
                YP_vecs[tag] = XYMLVectorizer.gen_XY_ML(
                    empty_info, 
                    read_column_Y, 
                    conf, 
                    "YP_$tag", 
                    paths.overall_name, 
                    paths.analysis_dir
                )
            end
        end
    elseif skip_all_trainings
        for (tag, (info, conf)) in sets
            YP_vecs[tag] = XYMLVectorizer.gen_XY_ML(
                empty_info, 
                read_column_Y, 
                conf, 
                "YP_$tag", 
                paths.overall_name, 
                paths.analysis_dir
            )
        end
    else
        JobLoggerTools.error_benji("Choose one: machine learning with bias correction, or without?", jobid)
    end


    Y_mats = Dict(
        :Y_tr  => XYMLVectorizer.mat_XY_ML(Y_vecs[:tr],  partition.N_src, partition.N_tr),
        :Y_bc  => XYMLVectorizer.mat_XY_ML(Y_vecs[:bc],  partition.N_src, partition.N_bc_persrc),
        :Y_ul  => XYMLVectorizer.mat_XY_ML(Y_vecs[:ul],  partition.N_src, partition.N_ul_persrc),
        :Y_lb  => XYMLVectorizer.mat_XY_ML(Y_vecs[:lb],  partition.N_src, partition.N_lb),
        :YP_tr => XYMLVectorizer.mat_XY_ML(YP_vecs[:tr], partition.N_src, partition.N_tr),
        :YP_bc => XYMLVectorizer.mat_XY_ML(YP_vecs[:bc], partition.N_src, partition.N_bc_persrc),
        :YP_ul => XYMLVectorizer.mat_XY_ML(YP_vecs[:ul], partition.N_src, partition.N_ul_persrc)
    )    

    JobLoggerTools.println_benji("Baseline sequence completed.", jobid)

    return Dict{Symbol, Matrix}(Y_mats)
end

"""
    random_sequence(
        ML_inputs::MLInputPreparer.MLInputBundle,
        partition::DatasetPartitioner.DatasetPartitionInfo, 
        paths::PathConfigBuilderDeborah.DeborahPathConfig,
        ranseed::Int, 
        jobid::Union{Nothing, String}=nothing; 
        read_column_Y::Int, 
        dump::Bool=true
    ) -> Dict{Symbol, Matrix{T}} where T<:Real

Generate a synthetic prediction sequence using Gaussian noise  
with jackknife-estimated variance, based on the input observable matrix.

This function is used primarily to create random baseline predictions for machine learning workflows,  
especially when training is skipped or for testing stochastic behavior.

# Arguments
- [`ML_inputs::MLInputPreparer.MLInputBundle`](@ref Deborah.DeborahCore.MLInputPreparer.MLInputBundle)  
    Struct containing input matrix `Y_df::Matrix{T}` and configuration array `conf_arr::Vector{Int}`.  
    `Y_df` should be 2D (``N_\\text{cnf} \\times N_\\text{src}``); a single column is selected using `read_column_Y`.

- [`partition::DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo)
    Struct defining data partitions: `lb_idx`, `tr_idx`, `bc_idx`, `ul_idx`.  
    These are used to split the data into labeled, training, bias-correction, and unlabeled sets.

- [`paths::DeborahPathConfig`](@ref Deborah.DeborahCore.PathConfigBuilderDeborah.DeborahPathConfig)
    Configuration for output filenames and directories.

- `ranseed::Int`  
    Seed for RNG to ensure reproducibility.

- `jobid::Union{Nothing, String}`  
    Optional job identifier for logging (e.g., with [`Deborah.Sarah.JobLoggerTools.println_benji`](@ref)).

# Keyword Arguments
- `read_column_Y::Int`  
    ``1``-based column index to select from `Y_df`.

- `dump::Bool=true`  
    If true, generated vectors are saved to disk in standard `Y_*/YP_*` format.

# Behavior
- Computes jackknife-based standard deviation per partition using the selected column from `Y_df`.
- For partitions used in prediction (`YP_*`), generates Gaussian random values with mean zero and matching standard deviation.
- Partitions not involved in prediction (e.g., `:lb`, sometimes `:bc`) receive empty vectors.
- Output is a dictionary with keys `:Y_lb`, `:Y_tr`, ..., `:YP_tr`, `:YP_ul`, etc.

# Returns
- `Dict{Symbol, Matrix{T}}`  
    Dictionary mapping tags like `:Y_bc`, `:YP_bc`, etc., to corresponding 1-column matrices (``N \\times 1``).
"""
function random_sequence(
    ML_inputs::MLInputPreparer.MLInputBundle,
    partition::DatasetPartitioner.DatasetPartitionInfo, 
    paths::PathConfigBuilderDeborah.DeborahPathConfig,
    ranseed::Int, 
    opt_blk_size::Int,
    jobid::Union{Nothing, String}=nothing; 
    read_column_Y::Int, 
    dump::Bool=true
)

    Y_df = ML_inputs.Y_df
    conf_arr = ML_inputs.conf_arr

    do_bias_correction = !isempty(ML_inputs.Y_tr_vec) && !isempty(ML_inputs.Y_bc_vec)
    no_bias_correction = !isempty(ML_inputs.Y_tr_vec) &&  isempty(ML_inputs.Y_bc_vec)
    skip_all_trainings =  isempty(ML_inputs.Y_tr_vec)

    JobLoggerTools.log_stage_sub1_benji("Random Gaussian Prediction Sequence", jobid)

    Y_info = XYInfoGenerator.gen_X_info(
        Y_df, 
        partition.N_cnf, 
        partition.N_src, 
        read_column_Y
    )

    lb_info, tr_info, bc_info, ul_info,
    lb_conf, tr_conf, bc_conf, ul_conf = 
        XYMLInfoGenerator.gen_XY_ML_info(
            Y_info, 
            conf_arr, 
            partition.lb_idx, 
            partition.tr_idx, 
            partition.bc_idx, 
            partition.ul_idx,
            partition.N_lb, 
            partition.N_tr, 
            partition.N_bc_persrc, 
            partition.N_ul_persrc,
            "X1_info_", 
            paths.overall_name, 
            paths.analysis_dir, 
            read_column_Y; 
            dump=dump, 
            jobid=jobid
        )

    sets = Dict(
        :lb => (lb_info, lb_conf),
        :tr => (tr_info, tr_conf),
        :bc => (bc_info, bc_conf),
        :ul => (ul_info, ul_conf)
    )

    Y_vecs  = Dict{Symbol, Vector{Real}}()
    for (tag, (info, conf)) in sets
        Y_vecs[tag]  = XYMLVectorizer.gen_XY_ML(
            info, 
            read_column_Y, 
            conf, 
            "Y_$tag", 
            paths.overall_name, 
            paths.analysis_dir
        )
    end

    Y_info_ORG = [Statistics.mean(Y_info[read_column_Y, iconf, :]) for iconf in 1:partition.N_cnf]

    JobLoggerTools.println_benji("optimal block size is $(opt_blk_size)", jobid)

    m_tmp, jvr_ORG = Jackknife.jackknife_average_error_from_raw(Y_info_ORG, opt_blk_size)
    tmp_str = AvgErrFormatter.avgerr_e2d_from_float(m_tmp, jvr_ORG)
    JobLoggerTools.println_benji("Check Original AVG(ERR) = $(tmp_str)", jobid)

    rgmdist = Distributions.Normal(0.0, jvr_ORG)

    rng = SeedManager.setup_rng(ranseed, jobid)

    function randvec(size) 
        return rand(rng, rgmdist, size)
    end

    if do_bias_correction
        YP_tr_vec = randvec(length(conf_arr[partition.tr_idx]))
        YP_bc_vec = randvec(length(conf_arr[partition.bc_idx]))
        YP_ul_vec = randvec(length(conf_arr[partition.ul_idx]))
    elseif no_bias_correction
        YP_tr_vec = randvec(length(conf_arr[partition.tr_idx]))
        YP_bc_vec = Float64[]
        YP_ul_vec = randvec(length(conf_arr[partition.ul_idx]))
    elseif skip_all_trainings
        YP_tr_vec = Float64[]
        YP_bc_vec = Float64[]
        YP_ul_vec = Float64[]
    else
        JobLoggerTools.error_benji("Choose one: machine learning with bias correction, or without?", jobid)
    end        

    Y_mats = Dict(
        :Y_tr  => XYMLVectorizer.mat_XY_ML(Y_vecs[:tr], partition.N_src, partition.N_tr),
        :Y_bc  => XYMLVectorizer.mat_XY_ML(Y_vecs[:bc], partition.N_src, partition.N_bc_persrc),
        :Y_ul  => XYMLVectorizer.mat_XY_ML(Y_vecs[:ul], partition.N_src, partition.N_ul_persrc),
        :Y_lb  => XYMLVectorizer.mat_XY_ML(Y_vecs[:lb], partition.N_src, partition.N_lb),
        :YP_tr => XYMLVectorizer.mat_XY_ML(YP_tr_vec,   partition.N_src, partition.N_tr),
        :YP_bc => XYMLVectorizer.mat_XY_ML(YP_bc_vec,   partition.N_src, partition.N_bc_persrc),
        :YP_ul => XYMLVectorizer.mat_XY_ML(YP_ul_vec,   partition.N_src, partition.N_ul_persrc)
    )

    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_tr], ML_inputs.tr_conf_arr, "YP_tr", paths.overall_name, paths.analysis_dir)
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_bc], ML_inputs.bc_conf_arr, "YP_bc", paths.overall_name, paths.analysis_dir)
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_ul], ML_inputs.ul_conf_arr, "YP_ul", paths.overall_name, paths.analysis_dir)

    JobLoggerTools.println_benji("Random sequence completed.", jobid)

    return Dict{Symbol, Matrix}(Y_mats)
end

end  # module BaselineSequence