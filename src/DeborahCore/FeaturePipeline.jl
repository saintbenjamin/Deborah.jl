# ============================================================================
# src/DeborahCore/FeaturePipeline.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module FeaturePipeline

import ..Sarah.JobLoggerTools
import ..Sarah.DataLoader
import ..Sarah.DatasetPartitioner
import ..Sarah.XYInfoGenerator
import ..PathConfigBuilderDeborah
import ..XYMLInfoGenerator
import ..XYMLVectorizer

"""
    run_feature_pipeline(
        read_column_X::Vector{Int},
        keys::Vector{String},
        path::String,
        conf_arr::Vector{Int},
        partition::DatasetPartitioner.DatasetPartitionInfo,
        paths::PathConfigBuilderDeborah.DeborahPathConfig;
        dump::Bool=true,
        jobid::Union{Nothing, String}=nothing
    ) -> Dict{String, NamedTuple{(:lb, :tr, :bc, :ul), NTuple{4, Vector{Float64}}}}

Run the feature preprocessing pipeline across multiple input feature files.

For each feature key (e.g., `"plaq.dat"`, `"rect.dat"`), this function:
1. Loads the corresponding raw `.dat` file as a matrix,
2. Extracts the column specified in `read_column_X[i]`,
3. Partitions the data into four groups (`lb`, `tr`, `bc`, `ul`),
4. Optionally dumps each partition to disk.

This version allows specifying a separate column index for each input feature file.

# Arguments
- `read_column_X::Vector{Int}`  
    A vector of ``1``-based column indices, one for each feature in `keys`.  
    `read_column_X[i]` is used to extract a column from the feature file `keys[i]`.  
    Each column is treated as an independent scalar input feature.

- `keys::Vector{String}`  
    List of feature file base names, such as `["plaq.dat", "rect.dat"]`.

- `path::String`  
    Directory path containing the raw `.dat` feature files.

- `conf_arr::Vector{Int}`  
    Configuration indices associated with rows in the feature files.

- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo)  
    Struct containing index vectors that define the four partitions:  
    - `lb`: labeled set
    - `tr`: training set
    - `bc`: bias correction set
    - `ul`: unlabeled set

- [`paths::PathConfigBuilderDeborah.DeborahPathConfig`](@ref Deborah.DeborahCore.PathConfigBuilderDeborah.DeborahPathConfig)
    Struct containing global path settings, such as `.analysis_dir`, `.overall_name`.

- `dump::Bool = true`  
    If true, each split feature vector is saved to disk as `.dat` files.

- `jobid::Union{Nothing, String}`  
    Optional identifier string for logging or debugging purposes.

# Returns
- `Dict{String, NamedTuple{(:lb, :tr, :bc, :ul), NTuple{4, Vector{Float64}}}}`  
    Dictionary mapping each feature name to a `NamedTuple` of split vectors.
"""
function run_feature_pipeline(
    read_column_X::Vector{Int},
    keys::Vector{String},
    path::String,
    conf_arr::Vector{Int},
    partition::DatasetPartitioner.DatasetPartitionInfo,
    paths::PathConfigBuilderDeborah.DeborahPathConfig;
    dump::Bool = true, 
    jobid::Union{Nothing, String} = nothing
)::Dict{String, NamedTuple}

    result = Dict{String, NamedTuple}()

    for (i, key) in enumerate(keys)
        raw_data = DataLoader.load_data_file(path, key, jobid)
        info = XYInfoGenerator.gen_X_info(
            raw_data, 
            partition.N_cnf, 
            partition.N_src, 
            read_column_X[i]
        )

        input_prefix = "X$(i)"

        lb_info, tr_info, bc_info, ul_info,
        lb_conf, tr_conf, bc_conf, ul_conf =
            XYMLInfoGenerator.gen_XY_ML_info(
                info, 
                conf_arr,
                partition.lb_idx, 
                partition.tr_idx,
                partition.bc_idx, 
                partition.ul_idx,
                partition.N_lb, 
                partition.N_tr,
                partition.N_bc_persrc, 
                partition.N_ul_persrc,
                input_prefix * "_info_",
                paths.overall_name,
                paths.analysis_dir,
                read_column_X[i];
                dump = dump,
                jobid = jobid
            )

        result[key] = (
            lb = XYMLVectorizer.gen_XY_ML(
                lb_info, 
                read_column_X[i], 
                lb_conf, 
                input_prefix * "_lb", 
                paths.overall_name, 
                paths.analysis_dir; 
                dump = dump
            ),
            tr = XYMLVectorizer.gen_XY_ML(
                tr_info, 
                read_column_X[i], 
                tr_conf, 
                input_prefix * "_tr", 
                paths.overall_name, 
                paths.analysis_dir; 
                dump = dump
            ),
            bc = XYMLVectorizer.gen_XY_ML(
                bc_info, 
                read_column_X[i], 
                bc_conf, 
                input_prefix * "_bc", 
                paths.overall_name, 
                paths.analysis_dir; 
                dump = dump
            ),
            ul = XYMLVectorizer.gen_XY_ML(
                ul_info, 
                read_column_X[i], 
                ul_conf, 
                input_prefix * "_ul", 
                paths.overall_name, 
                paths.analysis_dir; 
                dump = dump
            )
        )
    end

    return result
end

"""
    build_namedtuple_splitset(
        X_data::Dict{String, NamedTuple},
        split::String,
        key_order::Vector{String},
        jobid::Union{Nothing, String} = nothing
    ) -> NamedTuple

Construct a `NamedTuple` of feature vectors corresponding to a given data split.

This function extracts the specified data split (`tr`, `bc`, `ul`, and `lb`)
from each entry in the input feature dictionary and combines them into a
column-indexed `NamedTuple`, suitable for [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) model input.

# Arguments
- `X_data::Dict{String, NamedTuple}`: A dictionary mapping input keys to 4-way
  split feature tuples (`lb`, `tr`, `bc`, `ul`).
- `split::String`: Which dataset split to extract (`tr`, `bc`, `ul`,
  `lb`).
- `key_order::Vector{String}`: The order of keys to use when building columns.
- `jobid::Union{Nothing, String}`: Optional identifier string for logging or
  debugging purposes.

# Returns
- `NamedTuple`: A tuple with keys `:Column1`, `:Column2`, ... containing the
  feature vectors for the specified split.
"""
function build_namedtuple_splitset(
    X_data::Dict{String, NamedTuple},
    split::String,
    key_order::Vector{String},
    jobid::Union{Nothing, String} = nothing
)::NamedTuple
    if split ∉ ("tr", "bc", "ul", "lb")
        JobLoggerTools.error_benji("Invalid split keyword: $split. Must be one of: tr, bc, ul, lb.", jobid)
    end
    split_sym = Symbol(split)
    values = [getproperty(X_data[k], split_sym) for k in key_order]
    col_syms = Symbol.("Column" .* string.(1:length(key_order)))
    return NamedTuple{Tuple(col_syms)}(values)
end

end  # module FeaturePipeline