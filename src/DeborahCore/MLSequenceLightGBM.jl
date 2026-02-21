# ============================================================================
# src/DeborahCore/MLSequenceLightGBM.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MLSequenceLightGBM

import Printf: @sprintf
import OrderedCollections
import MLJ
import MLJBase
import LightGBM.MLJInterface: LGBMRegressor; flush(stdout); flush(stderr)
import ..Sarah.JobLoggerTools
import ..Sarah.TOMLLogger
import ..Sarah.DatasetPartitioner
import ..PathConfigBuilderDeborah
import ..FeaturePipeline
import ..XYMLVectorizer

"""
    ml_sequence_LightGBM(; 
        model_tag::String,
        X_data::Dict{String, NamedTuple},
        Y_tr_vec::Vector{T},
        Y_bc_vec::Vector{T},
        Y_ul_vec::Vector{T},
        Y_lb_vec::Vector{T},
        tr_conf_arr::Vector{Int},
        bc_conf_arr::Vector{Int},
        ul_conf_arr::Vector{Int},
        partition::DatasetPartitioner.DatasetPartitionInfo,
        X_list::Vector{String},
        paths::PathConfigBuilderDeborah.DeborahPathConfig,
        jobid::Union{Nothing, String}
    ) -> Tuple{Any, Dict{Symbol, Matrix}} where T<:Real

Train and evaluate a [LightGBM model](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) using the [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) framework.
Generates predicted ``Y`` matrices for training, bias correction, unlabeled, and labeled sets.

# Keyword Arguments
- `model_tag::String` : Short model identifier.
- `X_data::Dict{String, NamedTuple}`: Input feature dictionary. Each key maps to a `NamedTuple` with vectors for `:tr`, `:bc`, `:ul`, `:lb`.
- `Y_tr_vec::Vector{T}` : Target vector for training set.
- `Y_bc_vec::Vector{T}` : Target vector for bias correction set.
- `Y_ul_vec::Vector{T}` : Target vector for unlabeled set
- `Y_lb_vec::Vector{T}` : Target vector for labeled set.
- `tr_conf_arr::Vector{Int}` : Row-wise config index mapping for training set.
- `bc_conf_arr::Vector{Int}` : Row-wise config index mapping for bias correction set.
- `ul_conf_arr::Vector{Int}` : Row-wise config index mapping for unlabeled set.
- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo) : Configuration and counts for dataset partitioning.
- `X_list::Vector{String}` : Ordered list of feature names to be used.
- [`paths::PathConfigBuilderDeborah.DeborahPathConfig`](@ref Deborah.DeborahCore.PathConfigBuilderDeborah.DeborahPathConfig) : Contains path strings for saving result output.
- `jobid::Union{Nothing, String}` : Optional job tag for logging.

# Returns
- `Tuple{Any, Dict{Symbol, Matrix}}`
    - `mach` : Trained `JuliaAI/MLJ.jl` model (machine) using `Y_tr_vec`.
    - `Y_mats`   : Dictionary mapping:
        - `:YP_tr` → predicted ``Y`` matrix on training set
        - `:YP_bc` → predicted ``Y`` matrix on bias correction set
        - `:YP_ul` → predicted ``Y`` matrix on unlabeled set
"""
function ml_sequence_LightGBM(; 
    model_tag::String,
    X_data::Dict{String, NamedTuple},
    Y_tr_vec::Vector{T},
    Y_bc_vec::Vector{T},
    Y_ul_vec::Vector{T},
    Y_lb_vec::Vector{T},
    tr_conf_arr::Vector{Int},
    bc_conf_arr::Vector{Int},
    ul_conf_arr::Vector{Int},
    partition::DatasetPartitioner.DatasetPartitionInfo,
    X_list::Vector{String},
    paths::PathConfigBuilderDeborah.DeborahPathConfig,
    jobid::Union{Nothing, String}
)::Tuple{Any, Dict{Symbol, Matrix}} where T<:Real

    X_tr = FeaturePipeline.build_namedtuple_splitset(X_data, "tr", X_list, jobid)
    X_bc = FeaturePipeline.build_namedtuple_splitset(X_data, "bc", X_list, jobid)
    X_ul = FeaturePipeline.build_namedtuple_splitset(X_data, "ul", X_list, jobid)

    do_bias_correction = !isempty(Y_tr_vec) && !isempty(Y_bc_vec)
    no_bias_correction = !isempty(Y_tr_vec) &&  isempty(Y_bc_vec)
    skip_all_trainings =  isempty(Y_tr_vec)

    best_num_iterations = 40
    best_learning_rate = 0.1
    best_min_data_in_leaf = 20

    model_kwargs = Dict(
        :boosting => "gbdt",
        :num_iterations => best_num_iterations,
        :learning_rate => best_learning_rate,
        :num_leaves => 31,
        :max_depth => 3,

        :tree_learner => "serial",

        :histogram_pool_size => -1.0,
        :bin_construct_sample_cnt => 200000,

        :min_data_in_leaf => best_min_data_in_leaf,
        :min_sum_hessian_in_leaf => 0.001,
        :min_gain_to_split => 0.0,

        :bagging_fraction => 0.7,
        :bagging_freq => 0,
        :feature_fraction => 1.0,

        :lambda_l1 => 0.0,
        :lambda_l2 => 0.0,

        :max_bin => 255,
        :drop_rate => 0.1,
        :max_drop => 50,
        :skip_drop => 0.5,
        :xgboost_dart_mode => false,
        :uniform_drop => false,

        :top_rate => 0.2,
        :other_rate => 0.1,

        :min_data_per_group => 100,
        :max_cat_threshold => 32,
        :cat_l2 => 10.0,
        :cat_smooth => 10.0,

        :objective => "regression",
        :metric => ["l2"],

        :data_random_seed => 1,
        :is_unbalance => false,
        :boost_from_average => true,
        :use_missing => true,

        :num_threads => 1,

        :max_delta_step => 0.0,
        :feature_fraction_bynode => 1.0,
        :feature_fraction_seed => 2,
        :bagging_seed => 3,
        :extra_trees => false,
        :extra_seed => 6,
        :drop_seed => 4,
        :alpha => 0.9,
        :device_type => "cpu",
        :gpu_platform_id => -1,
        :gpu_device_id => -1,
        :force_col_wise => false,
        :force_row_wise => true,

        :early_stopping_round => 0,

        :linear_tree => false,
        :feature_pre_filter => true,
        :metric_freq => 1,
        :num_machines => 1,
        :local_listen_port => 12400,
        :time_out => 120,
        :gpu_use_dp => false,
        :num_gpu => 1,
        :categorical_feature => Int[],
        :truncate_booster => true
    )

    model = LGBMRegressor(; model_kwargs...);

    TOMLLogger.append_section_to_toml(paths.info_file, "hyper_parameters", OrderedCollections.OrderedDict(
        "model_name" => "LGBMRegressor",
        (string(k) => repr(v) for (k, v) in model_kwargs)...
    ))

    if do_bias_correction

        JobLoggerTools.log_stage_sub1_benji("Training sequence   (with bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("MLJ.machine() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            mach = MLJ.machine(model, X_tr, Y_tr_vec); flush(stdout); flush(stderr)
        end    
        JobLoggerTools.log_stage_sub1_benji("MLJBase.fit!() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            MLJBase.fit!(mach, force=true); flush(stdout); flush(stderr)
        end

        JobLoggerTools.log_stage_sub1_benji("Prediction sequence (with bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("MLJBase.predict() using X_tr_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_tr_pred = MLJBase.predict(mach, X_tr)
        end
        JobLoggerTools.log_stage_sub1_benji("MLJBase.predict() using X_bc_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_bc_pred = MLJBase.predict(mach, X_bc)
        end
        JobLoggerTools.log_stage_sub1_benji("MLJBase.predict() using X_ul_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_ul_pred = MLJBase.predict(mach, X_ul)
        end

        TOMLLogger.append_section_to_toml(paths.info_file, "L2_score", OrderedCollections.OrderedDict(
            "TR" => @sprintf("%.12e", MLJ.l2(Y_tr_pred, Y_tr_vec)),
            "BC" => @sprintf("%.12e", MLJ.l2(Y_bc_pred, Y_bc_vec)),
            "UL" => @sprintf("%.12e", MLJ.l2(Y_ul_pred, Y_ul_vec)),
        ))

    elseif no_bias_correction

        JobLoggerTools.log_stage_sub1_benji("Training sequence   (without bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("MLJ.machine() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            mach = MLJ.machine(model, X_tr, Y_tr_vec); flush(stdout); flush(stderr)
        end
        JobLoggerTools.log_stage_sub1_benji("MLJBase.fit!() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            MLJBase.fit!(mach, force=true); flush(stdout); flush(stderr)
        end

        JobLoggerTools.log_stage_sub1_benji("Prediction sequence (without bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("MLJBase.predict() using X_tr_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_tr_pred = MLJBase.predict(mach, X_tr)
        end
        Y_bc_pred = Float64[]
        JobLoggerTools.log_stage_sub1_benji("MLJBase.predict() using X_ul_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_ul_pred = MLJBase.predict(mach, X_ul)
        end

        TOMLLogger.append_section_to_toml(paths.info_file, "L2_score", OrderedCollections.OrderedDict(
            "TR" => @sprintf("%.12e", MLJ.l2(Y_tr_pred, Y_tr_vec)),
            "BC" => @sprintf("%.12e", 0.0),
            "UL" => @sprintf("%.12e", MLJ.l2(Y_ul_pred, Y_ul_vec)),
        ))

    elseif skip_all_trainings

        JobLoggerTools.log_stage_sub1_benji("Skip all trainings (TRP = 0)", jobid)

        mach = nothing
        Y_tr_pred = fill(zero(eltype(Y_tr_vec)), length(Y_tr_vec))
        Y_bc_pred = fill(zero(eltype(Y_bc_vec)), length(Y_bc_vec))
        Y_ul_pred = fill(zero(eltype(Y_ul_vec)), length(Y_ul_vec))

    else

        JobLoggerTools.error_benji("Choose one: machine learning with bias correction, or without?", jobid)

    end

    # --- Convert to matrix form ---
    Y_mats = Dict(
        :Y_tr  => XYMLVectorizer.mat_XY_ML(Y_tr_vec,  partition.N_src, partition.N_tr),
        :Y_bc  => XYMLVectorizer.mat_XY_ML(Y_bc_vec,  partition.N_src, partition.N_bc_persrc),
        :Y_ul  => XYMLVectorizer.mat_XY_ML(Y_ul_vec,  partition.N_src, partition.N_ul_persrc),
        :Y_lb  => XYMLVectorizer.mat_XY_ML(Y_lb_vec,  partition.N_src, partition.N_lb),
        :YP_tr => XYMLVectorizer.mat_XY_ML(Y_tr_pred, partition.N_src, partition.N_tr),
        :YP_bc => XYMLVectorizer.mat_XY_ML(Y_bc_pred, partition.N_src, partition.N_bc_persrc),
        :YP_ul => XYMLVectorizer.mat_XY_ML(Y_ul_pred, partition.N_src, partition.N_ul_persrc),
    )

    # Save predicted outputs
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_tr], tr_conf_arr, "YP_tr", paths.overall_name, paths.analysis_dir)
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_bc], bc_conf_arr, "YP_bc", paths.overall_name, paths.analysis_dir)
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_ul], ul_conf_arr, "YP_ul", paths.overall_name, paths.analysis_dir)

    return (mach, Y_mats)
end

end  # MLSequenceLightGBM