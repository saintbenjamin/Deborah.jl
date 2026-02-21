# ============================================================================
# src/DeborahCore/MLSequenceMiddleGBM.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MLSequenceMiddleGBM

import Printf: @sprintf
import OrderedCollections
import MLJ
import MLJBase
import LightGBM.MLJInterface: LGBMRegressor; flush(stdout); flush(stderr)
import PyPlot
import ..Rebekah.PyPlotLaTeX
import ..Sarah.JobLoggerTools
import ..Sarah.TOMLLogger
import ..Sarah.DatasetPartitioner
import ..PathConfigBuilderDeborah
import ..FeaturePipeline
import ..XYMLVectorizer

"""
    ml_sequence_MiddleGBM(; 
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

Train and evaluate a [LightGBM model](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM)
using the [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) framework, with additional hyperparameter tuning and
diagnostics compared to [`Deborah.DeborahCore.MLSequenceLightGBM.ml_sequence_LightGBM`](@ref).

This `MiddleGBM` variant:
- scans simple learning curves on the training set,
- performs random-search tuning over `num_iterations`, `min_data_in_leaf`, and `learning_rate`,
- logs the best-found hyperparameters and L2 scores to a [`TOML`](https://toml.io/en/) info file, and
- optionally saves residual plots for the training, bias-correction, and unlabeled sets when `jobid === nothing`.

As in the [base LightGBM pipeline](@ref Deborah.DeborahCore.MLSequenceLightGBM), it generates predicted ``Y`` matrices for training, bias correction, unlabeled, and
labeled sets, which can be used for downstream bias-corrected ML estimation and multi-ensemble analyses.

# Keyword Arguments
- `model_tag::String` : Short model identifier.
- `X_data::Dict{String, NamedTuple}` : Input feature dictionary. Each key maps to a `NamedTuple` with vectors for `:tr`, `:bc`, `:ul`, `:lb`.
- `Y_tr_vec::Vector{T}` : Target vector for the training set.
- `Y_bc_vec::Vector{T}` : Target vector for the bias-correction set.
- `Y_ul_vec::Vector{T}` : Target vector for the unlabeled set.
- `Y_lb_vec::Vector{T}` : Target vector for the full labeled set.
- `tr_conf_arr::Vector{Int}` : Row-wise configuration index mapping for the training set.
- `bc_conf_arr::Vector{Int}` : Row-wise configuration index mapping for the bias-correction set.
- `ul_conf_arr::Vector{Int}` : Row-wise configuration index mapping for the unlabeled set.
- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo) :
  Configuration and counts for dataset partitioning.
- `X_list::Vector{String}` : Ordered list of feature names to be used.
- [`paths::PathConfigBuilderDeborah.DeborahPathConfig`](@ref Deborah.DeborahCore.PathConfigBuilderDeborah.DeborahPathConfig) :
  Contains path strings for saving logs and analysis output.
- `jobid::Union{Nothing, String}` : Optional job tag for logging. If `nothing`, additional diagnostic plots are written.

# Returns
- `Tuple{Any, Dict{Symbol, Matrix}}`
    - `mach`  : Trained [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) machine wrapping the tuned [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) model (or `nothing` if training is skipped).
    - `Y_mats` : Dictionary of dense matrices with keys:
        - `:Y_tr`  → true ``Y`` on the training set
        - `:Y_bc`  → true ``Y`` on the bias-correction set
        - `:Y_ul`  → true ``Y`` on the unlabeled set
        - `:Y_lb`  → true ``Y`` on the full labeled set
        - `:YP_tr` → predicted ``Y`` on the training set
        - `:YP_bc` → predicted ``Y`` on the bias-correction set
        - `:YP_ul` → predicted ``Y`` on the unlabeled set
"""
function ml_sequence_MiddleGBM(; 
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

        JobLoggerTools.log_stage_sub1_benji("[MiddleGBM] Hyperparameter tuning with training set", jobid)

        r1 = MLJBase.range(model, :num_iterations,   lower=2,    upper=100)
        r2 = MLJBase.range(model, :min_data_in_leaf, lower=2,    upper=20)
        r3 = MLJBase.range(model, :learning_rate,    lower=1e-1, upper=1.0)

        JobLoggerTools.@logtime_benji jobid begin
            run_MiddleGBM_learning_curves(
                model, X_tr, Y_tr_vec, 
                paths.analysis_dir, "TR_"*paths.overall_name,
            jobid)
        end
        tuned_model = MLJ.TunedModel(
            model,
            tuning=MLJ.RandomSearch(),
            resampling=MLJ.Holdout(),
            ranges=[r1, r2, r3],
            measure=MLJ.l2,
            n=100,
        )

        JobLoggerTools.log_stage_sub1_benji("Training sequence   (with bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("MLJ.machine() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            mach = MLJ.machine(tuned_model, X_tr, Y_tr_vec); flush(stdout); flush(stderr)
        end
        JobLoggerTools.log_stage_sub1_benji("MLJBase.fit!() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            MLJBase.fit!(mach, force=true); flush(stdout); flush(stderr)
        end

        best_model = MLJBase.fitted_params(mach).best_model

        TOMLLogger.append_section_to_toml(paths.info_file, "best_hyperparameters", OrderedCollections.OrderedDict(
            "learning_rate"     => @sprintf("%.12e", best_model.learning_rate),
            "min_data_in_leaf"  => string(best_model.min_data_in_leaf),
            "num_iterations"    => string(best_model.num_iterations),
        ))

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

        if isnothing(jobid)
            save_MiddleGBM_plot(Y_tr_pred, Y_tr_vec, "resid_YP_tr_" * paths.overall_name * ".pdf", paths.analysis_dir)
            save_MiddleGBM_plot(Y_bc_pred, Y_bc_vec, "resid_YP_bc_" * paths.overall_name * ".pdf", paths.analysis_dir)
            save_MiddleGBM_plot(Y_ul_pred, Y_ul_vec, "resid_YP_ul_" * paths.overall_name * ".pdf", paths.analysis_dir)
        end

    elseif no_bias_correction

        JobLoggerTools.log_stage_sub1_benji("[MiddleGBM] Hyperparameter tuning with labeled set", jobid)

        r1 = MLJBase.range(model, :num_iterations,   lower=2,    upper=100)
        r2 = MLJBase.range(model, :min_data_in_leaf, lower=2,    upper=20)
        r3 = MLJBase.range(model, :learning_rate,    lower=1e-1, upper=1.0)

        JobLoggerTools.@logtime_benji jobid begin
            run_MiddleGBM_learning_curves(
                model, X_tr, Y_tr_vec, 
                paths.analysis_dir, "TR_"*paths.overall_name, 
                jobid
            )
        end
        tuned_model = MLJ.TunedModel(
            model,
            tuning=MLJ.RandomSearch(),
            resampling=MLJ.Holdout(),
            ranges=[r1, r2, r3],
            measure=MLJ.l2,
            n=100,
        )

        JobLoggerTools.log_stage_sub1_benji("Training sequence   (without bias correction)", jobid)

        JobLoggerTools.log_stage_sub1_benji("MLJ.machine() ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            mach = MLJ.machine(tuned_model, X_tr, Y_tr_vec); flush(stdout); flush(stderr)
        end
        JobLoggerTools.log_stage_sub1_benji("MLJBase.fit! () ::", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            MLJBase.fit!(mach, force=true); flush(stdout); flush(stderr)
        end

        best_model = MLJBase.fitted_params(mach).best_model

        TOMLLogger.append_section_to_toml(paths.info_file, "best_hyperparameters", OrderedCollections.OrderedDict(
            "learning_rate"     => @sprintf("%.12e", best_model.learning_rate),
            "min_data_in_leaf"  => string(best_model.min_data_in_leaf),
            "num_iterations"    => string(best_model.num_iterations),
        ))

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

        if isnothing(jobid)
            save_MiddleGBM_plot(Y_tr_pred, Y_tr_vec, "resid_YP_tr_" * paths.overall_name * ".pdf", paths.analysis_dir)
            save_MiddleGBM_plot(Y_ul_pred, Y_ul_vec, "resid_YP_ul_" * paths.overall_name * ".pdf", paths.analysis_dir)
        end

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

    JobLoggerTools.log_stage_sub1_benji("[MiddleGBM] Done.", jobid)

    return (mach, Y_mats)
end

"""
    run_MiddleGBM_learning_curves(
        model::LGBMRegressor,
        features::NamedTuple,
        targets::Vector{Float64},
        res_dir::String,
        outsuffix::String,
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Generate and save learning curve plots for the given [LightGBM model](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM)
with hyperparameter sweeps for `num_iterations`, `learning_rate`, and `min_data_in_leaf`.

This function evaluates model performance via cross-validation across a range
of hyperparameter values, and saves the results as cropped PDF plots in `res_dir`.

# Arguments
- `model::LGBMRegressor`          : `JuliaAI/MLJ.jl`-compatible model object.
- `features::NamedTuple`          : Feature table (`NamedTuple`) used for model training.
- `targets::Vector{Float64}`      : Corresponding target values for regression.
- `res_dir::String`               : Directory in which to save the resulting plots.
- `outsuffix::String`             : Filename suffix to differentiate output.
- `jobid::Union{Nothing, String}` : Optional job ID for structured logging.

# Output Files
- PDF plots are saved to `res_dir` with filenames:
    - `optimal_boosting_stage_\$(outsuffix).pdf`
    - `optimal_learning_rate_\$(outsuffix).pdf`
    - `optimal_min_data_in_leaf_\$(outsuffix).pdf`

# Notes
- Plotting is performed only when `jobid === nothing`.
- When plotting, [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) + [``\\LaTeX``](https://www.latex-project.org/) style is configured via:
  [`Deborah.Rebekah.PyPlotLaTeX.set_pyplot_latex_style`](@ref).

# Returns
- `Nothing`
"""
function run_MiddleGBM_learning_curves(
    model::LGBMRegressor,
    features::NamedTuple,
    targets::Vector{Float64},
    res_dir::String,
    outsuffix::String,
    jobid::Union{Nothing, String}=nothing
)

    # --- local plotting setup ---
    if isnothing(jobid)
        # You said you want this called up-front for PyPlot-based figures.
        PyPlotLaTeX.set_pyplot_latex_style()

        # Optional: if you want a specific DPI/size behavior independent of rcParams,
        # you can control it per-figure below. We'll keep it explicit like other plots.
        mkpath(res_dir)
    end

    # ---- small helper ----
    _save_and_crop_pdf!(pdfpath::String) = begin
        PyPlot.savefig(pdfpath)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $pdfpath`)
            cropped = replace(pdfpath, ".pdf" => "-crop.pdf")
            mv(cropped, pdfpath; force=true)
        end
        return nothing
    end

    # =========================
    # 1) num_iterations curve
    # =========================
    JobLoggerTools.log_stage_sub1_benji("Running learning curve for num_iterations", jobid)
    curve = MLJ.learning_curve(
        MLJ.machine(model, features, targets),
        resampling=MLJ.CV(nfolds=5),
        range=MLJBase.range(model, :num_iterations, lower=2, upper=60),
        resolution=60,
        measure=MLJ.l2,
    ); flush(stdout); flush(stderr)

    if isnothing(jobid)
        fig, ax = PyPlot.subplots(figsize=(6, 3), dpi=500)
        ax.plot(curve.parameter_values, curve.measurements, linewidth=2)
        ax.set_xlabel("boosting stages (iteration)")
        ax.set_ylabel("L2 loss")
        ax.grid(true)
        fig.tight_layout()

        pdfpath = joinpath(res_dir, "optimal_boosting_stage_$(outsuffix).pdf")
        _save_and_crop_pdf!(pdfpath)
        PyPlot.close(fig)
    end

    # ======================
    # 2) learning_rate curve
    # ======================
    JobLoggerTools.log_stage_sub1_benji("Running learning curve for learning_rate...", jobid)
    curve = MLJ.learning_curve(
        MLJ.machine(model, features, targets),
        resampling=MLJ.CV(nfolds=5),
        range=MLJBase.range(model, :learning_rate, lower=1e-3, upper=2e-1, scale=:log),
        resolution=60,
        measure=MLJ.l2,
    ); flush(stdout); flush(stderr)

    if isnothing(jobid)
        fig, ax = PyPlot.subplots(figsize=(6, 3), dpi=500)
        ax.plot(curve.parameter_values, curve.measurements, linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("L2 loss")
        ax.grid(true)
        fig.tight_layout()

        pdfpath = joinpath(res_dir, "optimal_learning_rate_$(outsuffix).pdf")
        _save_and_crop_pdf!(pdfpath)
        PyPlot.close(fig)
    end

    # ==========================
    # 3) min_data_in_leaf curve
    # ==========================
    JobLoggerTools.log_stage_sub1_benji("Running learning curve for min_data_in_leaf...", jobid)
    curve = MLJ.learning_curve(
        MLJ.machine(model, features, targets),
        resampling=MLJ.CV(nfolds=5),
        range=MLJBase.range(model, :min_data_in_leaf, lower=2, upper=40),
        resolution=40,
        measure=MLJ.l2,
    ); flush(stdout); flush(stderr)

    if isnothing(jobid)
        fig, ax = PyPlot.subplots(figsize=(6, 3), dpi=500)
        ax.plot(curve.parameter_values, curve.measurements, linewidth=2)
        ax.set_xlabel("Min data in leaf")
        ax.set_ylabel("L2 loss")
        ax.grid(true)
        fig.tight_layout()

        pdfpath = joinpath(res_dir, "optimal_min_data_in_leaf_$(outsuffix).pdf")
        _save_and_crop_pdf!(pdfpath)
        PyPlot.close(fig)
    end

    JobLoggerTools.log_stage_sub1_benji("Learning curves saved.", jobid)
end

"""
    save_MiddleGBM_plot(
        predictions::AbstractVector, 
        ground_truth::AbstractVector, 
        filepath::String,
        res_dir::String
    ) -> Nothing

Generate and save a residual plot comparing predictions to ground truth values.

This function computes the relative residuals:
```math
    \\frac{\\text{prediction} - \\text{truth}}{\\text{truth}}
```
for each configuration index and visualizes the error trend.
The resulting plot is saved at the given `filepath`. If the file is a PDF,
[`pdfcrop`](https://ctan.org/pkg/pdfcrop) is automatically applied to remove extra margins.

# Arguments
- `predictions::AbstractVector`     : Model predictions for a given target set.
- `ground_truth::AbstractVector`    : True target values corresponding to the predictions.
- `filepath::String`                : Full file path (including filename and extension) to save the plot.
                                     Should typically end with `.pdf` or `.png`.
- `res_dir::String`                 : Directory where auxiliary output/logging may be stored.

# Output
- A residual plot saved to the specified `filepath`.

# Notes
- Uses [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) for plotting.
- If `filepath` ends in `.pdf`, [`pdfcrop`](https://ctan.org/pkg/pdfcrop) is run automatically to trim whitespace.
- If the output directory (`res_dir`) does not exist, it should be created before calling this function.

# Returns
- `Nothing`
"""
function save_MiddleGBM_plot(
    predictions::AbstractVector,
    ground_truth::AbstractVector,
    filepath::String,
    res_dir::String
)

    # Apply your LaTeX plotting style
    PyPlotLaTeX.set_pyplot_latex_style()

    residual = (predictions .- ground_truth) ./ (ground_truth .+ 1e-12)
    residual[.!isfinite.(residual)] .= NaN
    xvals = collect(1:length(residual))

    fig, ax = PyPlot.subplots(figsize=(6, 3), dpi=500)
    ax.plot(xvals, residual, linewidth=2, label="Relative Residual")
    ax.axhline(0.0, linestyle="--", color="black", label="Zero Line")

    ax.set_xlabel("Configuration Index")
    ax.set_ylabel("(Prediction - Truth) / Truth")
    ax.grid(true)
    ax.legend(frameon=false)
    fig.tight_layout()

    mkpath(res_dir)
    outpath = joinpath(res_dir, filepath)
    PyPlot.savefig(outpath)

    if endswith(filepath, ".pdf") && Sys.which("pdfcrop") !== nothing
        run(`pdfcrop $outpath`)
        cropped = replace(outpath, ".pdf" => "-crop.pdf")
        mv(cropped, outpath; force=true)
    end

    PyPlot.close(fig)
end

end  # module MLSequenceMiddleGBM