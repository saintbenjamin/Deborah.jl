# ============================================================================
# src/DeborahCore/MLSequencePyCallLightGBM.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MLSequencePyCallLightGBM

import Printf: @sprintf
import OrderedCollections
import PyCall
import PyPlot
import Statistics
import DataFrames
import ..Sarah.JobLoggerTools
import ..Sarah.TOMLLogger
import ..Sarah.DatasetPartitioner
import ..PathConfigBuilderDeborah
import ..FeaturePipeline
import ..XYMLVectorizer

"""
    flush_py() -> Nothing

Flush Python's stdout and stderr streams using embedded [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl).

# Behavior
- Calls `sys.stdout.flush()` and `sys.stderr.flush()` via [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl) to ensure Python-side output is printed immediately.
- Useful when embedding Python code that prints output not immediately visible in Julia.

# Returns
- `Nothing`: This function performs side effects only.
"""
function flush_py()
    PyCall.py"""import sys
    sys.stdout.flush() 
    sys.stderr.flush()"""
end

"""
    l2(
        ŷ::Vector{Float64}, 
        y::Vector{Float64}
    ) -> Float64

Compute the L2 loss (mean squared error) between predicted and true values.

# Formula

```math
L_2(\\hat{\\mathbf{y}}, \\mathbf{y}) = \\frac{1}{N}\\sum_{i=1}^{N}\\left(\\hat{y}_i - y_i\\right)^2
```

# Arguments
- `ŷ`: Vector of predicted values.
- `y`: Vector of ground truth values.

# Returns
- `Float64`: Mean squared error between `ŷ` and `y`.
"""
function l2(
    ŷ::Vector{Float64}, 
    y::Vector{Float64}
)::Float64
    return Statistics.mean((ŷ .- y).^2)
end

"""
    ml_sequence_PyCallLightGBM(; 
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
    ) -> Dict{Symbol, Matrix} where T<:Real

Train and evaluate a LightGBM model using the [Python LightGBM API](https://github.com/microsoft/LightGBM) via [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl).
Generates predicted ``Y`` matrices for training, bias correction, unlabeled, and labeled sets.
This variant enables direct access to native [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) features and uses Python-based training
pipelines to support cross-language comparison or advanced tuning not yet available in [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/).

All predictions are then reshaped back into ``(N_\\text{cnf}, N_\\text{src})`` matrix form.

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
- `Dict{Symbol, Matrix}`
    - `Y_mats`   : Dictionary mapping:
        - `:YP_tr` → predicted ``Y`` matrix on training set
        - `:YP_bc` → predicted ``Y`` matrix on bias correction set
        - `:YP_ul` → predicted ``Y`` matrix on unlabeled set

# Notes
- [Python LightGBM](https://github.com/microsoft/LightGBM) must be available in the environment via [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl).
"""
function ml_sequence_PyCallLightGBM(; 
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
)::Dict{Symbol, Matrix} where T<:Real

    X_tr = FeaturePipeline.build_namedtuple_splitset(X_data, "tr", X_list, jobid)
    X_bc = FeaturePipeline.build_namedtuple_splitset(X_data, "bc", X_list, jobid)
    X_ul = FeaturePipeline.build_namedtuple_splitset(X_data, "ul", X_list, jobid)
    X_tr_PYM_vec = Matrix(DataFrames.DataFrame(X_tr))
    X_bc_PYM_vec = Matrix(DataFrames.DataFrame(X_bc))
    X_ul_PYM_vec = Matrix(DataFrames.DataFrame(X_ul))

    do_bias_correction = !isempty(Y_tr_vec) && !isempty(Y_bc_vec)
    no_bias_correction = !isempty(Y_tr_vec) &&  isempty(Y_bc_vec)
    skip_all_trainings =  isempty(Y_tr_vec)

    best_num_iterations = 40
    best_learning_rate = 0.1
    best_min_data_in_leaf = 20

    JobLoggerTools.@logtime_benji jobid begin
        PyCallLGBMRegressor = PyCall.pyimport("lightgbm"); flush(stdout); flush(stderr)
    end

    model_kwargs = Dict(
        :boosting_type => "gbdt",
        :n_estimators => best_num_iterations,
        :learning_rate => best_learning_rate,
        :num_leaves => 31,
        :max_depth => 3,

        # histogram_pool_size         => automatic in PyCall

        :subsample_for_bin => 200000,

        :min_child_samples => best_min_data_in_leaf,  # = min_data_in_leaf
        :min_child_weight => 0.001,                   # = min_sum_hessian_in_leaf
        :min_split_gain => 0.0,                       # = min_gain_to_split

        :subsample => 0.7,                            # = bagging_fraction
        :subsample_freq => 0,                         # = bagging_freq
        :colsample_bytree => 1.0,                     # = feature_fraction

        :reg_alpha => 0.0,                            # = lambda_l1
        :reg_lambda => 0.0,                           # = lambda_l2

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
        :metric => "l2",                               # metric = ["l2"] → string OK

        :importance_type => "split",                  # N/A in MLJ.jl

        :random_state => 1,                           # = data_random_seed
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

        :early_stopping_round => 0,                   # early_stopping_round in MLJ.jl

        :linear_tree => false,                        # N/A in PyCall
        :feature_pre_filter => true,                  # N/A in PyCall
        :metric_freq => 1,                            # N/A in PyCall
        :num_machines => 1,                           # N/A in PyCall
        :local_listen_port => 12400,                  # N/A in PyCall
        :time_out => 120,                             # N/A in PyCall
        :gpu_use_dp => false,                         # N/A in PyCall
        :num_gpu => 1                                 # N/A in PyCall

        # :categorical_feature => Int[],              # N/A in PyCall
        # :truncate_booster => true                   # N/A in PyCall
    )

    model = PyCallLGBMRegressor.LGBMRegressor(; model_kwargs...)

    TOMLLogger.append_section_to_toml(paths.info_file, "hyper_parameters", OrderedCollections.OrderedDict(
        "model_name" => "PyCall.pyimport(\"lightgbm\")",
        (string(k) => repr(v) for (k, v) in model_kwargs)...
    ))

    if do_bias_correction

        JobLoggerTools.log_stage_sub1_benji("Training sequence   (with bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("model.fit() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            model.fit(X_tr_PYM_vec,Y_tr_vec, eval_metric="l2", eval_set=[(X_bc_PYM_vec, Y_bc_vec), (X_tr_PYM_vec, Y_tr_vec)]); flush(stdout); flush(stderr)
            flush_py()
        end

        # ---------------------------------------------------------------------------------------------

        # infotree=[
        #     # "split_gain", 
        #     "internal_value", 
        #     "internal_count", 
        #     # "internal_weight", 
        #     "leaf_count", 
        #     # "leaf_weight", 
        #     "data_percentage"
        #     ]

        # # ax = Main.ModelFactory.PyCallLGBMRegressor.plot_tree(model, tree_index=0, figsize=(20, 20))
        # # Main.ModelFactory.PyCallLGBMRegressor.plot_importance(model, height=0.2, title="Feature importance", xlabel="Feature importance", ylabel="Features", importance_type="split", ignore_zero=true, grid=true)
        # ModelFactory.PyCallLGBMRegressor.plot_metric(model, metric="l2", title="Metric during training", xlabel="Iterations", ylabel="auto", figsize=(10,10), grid=true)
        # PyPlot.savefig("treetest.png")
        # PyPlot.close()

        # graph = ModelFactory.PyCallLGBMRegressor.create_tree_digraph(model, tree_index=0, show_info=infotree, precision=4, orientation="vertical", max_category_values=10, format="png", name="Tree0")
        # graph.render(view=true)
        # # graph = Main.ModelFactory.PyCallLGBMRegressor.create_tree_digraph(model, tree_index=1, show_info=infotree, precision=4, orientation="vertical", max_category_values=10, format="png", name="Tree1")
        # # graph.render(view=true)
        # # graph = Main.ModelFactory.PyCallLGBMRegressor.create_tree_digraph(model, tree_index=2, show_info=infotree, precision=4, orientation="vertical", max_category_values=10, format="png", name="Tree2")
        # # graph.render(view=true)

        # # savefig("testtree.png")

        # ---------------------------------------------------------------------------------------------

        JobLoggerTools.log_stage_sub1_benji("Prediction sequence (with bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("model.predict() using X_tr_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_tr_pred = Vector{Float64}(model.predict(X_tr_PYM_vec))
        end
        JobLoggerTools.log_stage_sub1_benji("model.predict() using X_bc_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_bc_pred = Vector{Float64}(model.predict(X_bc_PYM_vec))
        end
        JobLoggerTools.log_stage_sub1_benji("model.predict() using X_ul_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_ul_pred = Vector{Float64}(model.predict(X_ul_PYM_vec))
        end

        TOMLLogger.append_section_to_toml(paths.info_file, "L2_score", OrderedCollections.OrderedDict(
            "TR" => @sprintf("%.12e", l2(Y_tr_pred, Y_tr_vec)),
            "BC" => @sprintf("%.12e", l2(Y_bc_pred, Y_bc_vec)),
            "UL" => @sprintf("%.12e", l2(Y_ul_pred, Y_ul_vec)),
        ))

    elseif no_bias_correction

        JobLoggerTools.log_stage_sub1_benji("Training sequence   (without bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("model.fit() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            model.fit(X_tr_PYM_vec,Y_tr_vec, eval_metric="l2", eval_set=[(X_tr_PYM_vec, Y_tr_vec)]); flush(stdout); flush(stderr)
            flush_py()
        end

        JobLoggerTools.log_stage_sub1_benji("Prediction sequence (without bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("model.predict() using X_tr_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_tr_pred = Vector{Float64}(model.predict(X_tr_PYM_vec))
        end
        Y_bc_pred = Float64[]
        JobLoggerTools.log_stage_sub1_benji("model.predict() using X_ul_" * model_tag * "_vec ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Y_ul_pred = Vector{Float64}(model.predict(X_ul_PYM_vec))
        end

        TOMLLogger.append_section_to_toml(paths.info_file, "L2_score", OrderedCollections.OrderedDict(
            "TR" => @sprintf("%.12e", l2(Y_tr_pred, Y_tr_vec)),
            "BC" => @sprintf("%.12e", 0.0),
            "UL" => @sprintf("%.12e", l2(Y_ul_pred, Y_ul_vec)),
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
        :YP_ul => XYMLVectorizer.mat_XY_ML(Y_ul_pred, partition.N_src, partition.N_ul_persrc)
    )

    # Save predicted outputs
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_tr], tr_conf_arr, "YP_tr", paths.overall_name, paths.analysis_dir)
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_bc], bc_conf_arr, "YP_bc", paths.overall_name, paths.analysis_dir)
    _ = XYMLVectorizer.gen_XY_ML(Y_mats[:YP_ul], ul_conf_arr, "YP_ul", paths.overall_name, paths.analysis_dir)

    return Y_mats
end

end  # module MLSequencePyCallLightGBM