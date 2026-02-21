# ============================================================================
# src/Esther/TraceDataLoader.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module TraceDataLoader

import ..Sarah.JobLoggerTools
import ..Sarah.DataLoader
import ..PathConfigBuilderEsther

"""
    arr_XY_maker(
        X_label::String,
        my_tex_dir::String,
        overall_name::String,
        jobid::Union{Nothing, String}=nothing
    ) -> Vector{Float64}

Load a processed trace file named `<X_label>_<overall_name>.dat` from the given directory.

# Arguments
- `X_label::String`: Label prefix (e.g., `Y_lb`, `YP_tr`).
- `my_tex_dir::String`: Path to the directory containing `.dat` files.
- `overall_name::String`: Base name of the file (e.g., analysis descriptor).
- `jobid::Union{Nothing, String}`: Optional job identifier for logging context.

# Returns
- `Vector{Float64}`: First column of the file contents.

# Errors
- Exits with code `1` if the target file does not exist.
"""
function arr_XY_maker(
    X_label::String,
    my_tex_dir::String,
    overall_name::String, 
    jobid::Union{Nothing, String}=nothing
)::Vector{Float64}
    X_dat_file = my_tex_dir * "/" * X_label * "_" * overall_name * ".dat"

    if !isfile(X_dat_file)
        JobLoggerTools.error_benji("Cannot find file: $(X_dat_file)", jobid)
    end

    X_df = DataLoader.try_multi_readdlm(X_dat_file)

    if size(X_df, 2) == 0
        return Float64[]
    end

    return X_df[:, 1]
end

"""
    load_trace_data(
        paths::PathConfigBuilderEsther.EstherPathConfig,
        jobid::Union{Nothing, String}=nothing
    ) -> Dict{String, Vector{Vector{Float64}}}

Load all required trace datasets from disk using path metadata.

This function reads both `Y_*` and `YP_*` targets (true and predicted)
for each of the four trace moments (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``), and stores them in
a dictionary grouped by label.

# Arguments
- `paths::PathConfigBuilderEsther.EstherPathConfig`: Precomputed path configuration struct.
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `Dict{String, Vector{Vector{Float64}}}`: Dictionary mapping each label 
  (e.g., `Y_lb`, `YP_bc`) to a vector of 4 datasets, one per ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
"""
function load_trace_data(
    paths::PathConfigBuilderEsther.EstherPathConfig, 
    jobid::Union{Nothing, String}=nothing
)::Dict{String, Vector{Vector{Float64}}}
    labels = ["Y_info", "Y_tr", "Y_bc", "Y_ul", "Y_lb", "YP_tr", "YP_bc", "YP_ul"]

    trace_data = Dict{String, Vector{Vector{Float64}}}()

    # Load other labels using arr_XY_maker
    for label in labels
        trace_data[label] = Vector{Vector{Float64}}()
        for i in 1:4
            dat = arr_XY_maker(
                label, paths.tr_tex_dirs[i],
                paths.traceM_names[i],
                jobid
            )
            push!(trace_data[label], dat)
        end
    end

    return trace_data
end

end  # module TraceDataLoader