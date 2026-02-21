# ============================================================================
# src/DeborahCore/MLInputPreparer.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MLInputPreparer

import ..Sarah.DatasetPartitioner
import ..Sarah.DataLoader
import ..Sarah.XYInfoGenerator
import ..PathConfigBuilderDeborah
import ..FeaturePipeline
import ..XYMLInfoGenerator
import ..XYMLVectorizer

"""
    struct MLInputBundle{T<:Real}

Container for machine learning input and target data used in the [`Deborah.DeborahCore`](@ref)
pipeline.

This struct stores the full feature set `X_data`, target vectors `Y_*_vec` for
each partition (training set, bias correction set, unlabeled set, and labeled
set), the raw data matrix `Y_df`, and the corresponding configuration index
arrays used to assemble them.

# Type Parameters
- `T<:Real` : Element type of all target vectors and matrices (typically
  `Float64`).

# Fields
- `X_data::Dict{String, NamedTuple}`   : Preprocessed feature dictionary, keyed
  by filename.
- `Y_df::Matrix{T}`                    : Original raw ``Y`` matrix (``N_{\\text{cfg}} \\times N_{\\text{src}}``).
- `Y_tr_vec::Vector{T}`                : Flattened ``Y`` vector for training set.
- `Y_bc_vec::Vector{T}`                : Flattened ``Y`` vector for bias-correction
  set.
- `Y_ul_vec::Vector{T}`                : Flattened ``Y`` vector for unlabeled set.
- `Y_lb_vec::Vector{T}`                : Flattened ``Y`` vector for labeled set.
- `conf_arr::Vector{Int}`              : Mapping from global row index to
  configuration index.
- `tr_conf_arr::Vector{Int}`           : Row indices used for training set ``Y``.
- `bc_conf_arr::Vector{Int}`           : Row indices used for bias-correction set ``Y``.
- `ul_conf_arr::Vector{Int}`           : Row indices used for unlabeled set ``Y``.
- `lb_conf_arr::Vector{Int}`           : Row indices used for labeled set ``Y``.
"""
struct MLInputBundle{T<:Real}
    X_data::Dict{String, NamedTuple}
    Y_df::Matrix{T}
    Y_tr_vec::Vector{T}
    Y_bc_vec::Vector{T}
    Y_ul_vec::Vector{T}
    Y_lb_vec::Vector{T}
    conf_arr::Vector{Int}
    tr_conf_arr::Vector{Int}
    bc_conf_arr::Vector{Int}
    ul_conf_arr::Vector{Int}
    lb_conf_arr::Vector{Int}
end

"""
    prepare_ML_inputs(
        partition::DatasetPartitioner.DatasetPartitionInfo, 
        X_file_list::Vector{String}, 
        Y_file::String, 
        paths::PathConfigBuilderDeborah.DeborahPathConfig; 
        jobid::Union{Nothing, String}=nothing,
        dump::Bool=false, 
        read_column_X::Vector{Int},
        read_column_Y::Int,
        index_column::Int
    ) -> MLInputBundle

Load and organize all machine learning input data from raw `.dat` files.

This function loads the target data (`Y_file`) and a list of input feature files (`X_file_list`),  
extracts specific columns from each using `read_column_X` and `read_column_Y`,  
applies dataset partitioning according to the `partition` object,  
and returns the labeled and unlabeled splits of features and targets  
in a structured format suitable for training and evaluation.

# Arguments
- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo)
    Struct defining how configuration indices are split into `lb`, `tr`, `bc`, and `ul` sets.

- `X_file_list::Vector{String}`  
    List of feature file names, e.g., `["plaq.dat", "rect.dat"]`.

- `Y_file::String`  
    Name of the target file to be used as `Y`.

- [`paths::PathConfigBuilderDeborah.DeborahPathConfig`](@ref Deborah.DeborahCore.PathConfigBuilderDeborah.DeborahPathConfig)
    Struct with directory and filename conventions used for reading/writing data.

- `jobid::Union{Nothing, String}`  
    Optional identifier used for structured logging or job tracking.

- `dump::Bool = false`  
    Whether to save preprocessed `X` feature vectors into disk files.

- `read_column_X::Vector{Int}`  
    A vector of ``1``-based column indices, one for each feature file in `X_file_list`.  
    `read_column_X[i]` is used to select a column from file `X_file_list[i]`.

- `read_column_Y::Int`  
    ``1``-based column index to extract from the `Y_file`.

- `index_column::Int`  
    ``1``-based column index from which to read the configuration indices in the `Y_file`.  
    If set to `0`, configuration indices will be auto-generated as `1:N_cnf`.

# Returns
- [`Deborah.DeborahCore.MLInputPreparer.MLInputBundle`](@ref)
    Composite struct containing:
    - `X_dict::Dict{String, NamedTuple}` → partitioned input feature vectors (`:lb`, `:tr`, `:bc`, `:ul`)
    - `Y_lb`, `Y_tr`, `Y_bc`, `Y_ul` → target label vectors
    - configuration index arrays for each group
"""
function prepare_ML_inputs(
    partition::DatasetPartitioner.DatasetPartitionInfo, 
    X_file_list::Vector{String}, 
    Y_file::String, 
    paths::PathConfigBuilderDeborah.DeborahPathConfig; 
    jobid::Union{Nothing, String}=nothing,
    dump::Bool=false, 
    read_column_X::Vector{Int},
    read_column_Y::Int,
    index_column::Int
)::MLInputBundle

    # Load full Y file
    Y_df     = DataLoader.load_data_file(paths.path, Y_file, jobid)

    # Generate full configuration index array
    conf_arr = XYInfoGenerator.gen_conf_from_Y(Y_df, partition.N_cnf, partition.N_src, index_column)
    conf_arr = Int.(conf_arr)

    # Process and combine X features
    X_data = FeaturePipeline.run_feature_pipeline(
        read_column_X,
        X_file_list,
        paths.path,
        conf_arr,
        partition,
        paths;
        dump=dump,
        jobid=jobid
    )

    # Parse metadata from Y file
    Y_info = XYInfoGenerator.gen_X_info(Y_df, partition.N_cnf, partition.N_src, read_column_Y)

    # Split metadata + index arrays into subsets
    Y_lb_info, Y_tr_info, Y_bc_info, Y_ul_info,
    lb_conf_arr, tr_conf_arr, bc_conf_arr, ul_conf_arr =
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
            "Y_info_",
            paths.overall_name,
            paths.analysis_dir, 
            read_column_Y;
            jobid=jobid
        )

    # Extract actual Y vectors for each subset
    Y_lb_vec = XYMLVectorizer.gen_XY_ML(Y_lb_info, read_column_Y, lb_conf_arr, "Y_lb", paths.overall_name, paths.analysis_dir)
    Y_tr_vec = XYMLVectorizer.gen_XY_ML(Y_tr_info, read_column_Y, tr_conf_arr, "Y_tr", paths.overall_name, paths.analysis_dir)
    Y_bc_vec = XYMLVectorizer.gen_XY_ML(Y_bc_info, read_column_Y, bc_conf_arr, "Y_bc", paths.overall_name, paths.analysis_dir)
    Y_ul_vec = XYMLVectorizer.gen_XY_ML(Y_ul_info, read_column_Y, ul_conf_arr, "Y_ul", paths.overall_name, paths.analysis_dir)

    return MLInputBundle(
        X_data, Y_df,
        Y_tr_vec, Y_bc_vec, Y_ul_vec, Y_lb_vec,
        conf_arr,
        tr_conf_arr, bc_conf_arr, ul_conf_arr, lb_conf_arr
    )
end

end  # module MLInputPreparer