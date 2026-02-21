# ============================================================================
# src/Sarah/NameParser.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module NameParser

import ..JobLoggerTools

"""
    model_suffix(
        model_name::String,
        jobid::Union{Nothing, String}=nothing
    ) -> String

Return a short uppercase string used as a model tag in filenames or logs.

# Arguments
- `model_name`: Full model name identifier (e.g., `"LightGBM"`, `"Ridge"`).
- `jobid::Union{Nothing, String}` : Optional job identifier for structured logging.

# Returns
- Short tag (e.g., `"GBM"`, `"RID"`, `"LAS"`).

# Supported Models
- `"LightGBM"`  → `"GBM"`
- `"Lasso"`     → `"LAS"`
- `"Ridge"`     → `"RID"`
- `"Baseline"`  → `"BAS"`
- `"Random"`    → `"RGM"`
- `"MiddleGBM"` → `"MDG"`
- `"PyGBM"`     → `"PYG"`

# Throws
- `ArgumentError` if the `model_name` is not recognized.
"""
function model_suffix(
    model_name::String,
    jobid::Union{Nothing, String}=nothing
)::String
    if model_name == "LightGBM"
        return "GBM"
    elseif model_name == "Lasso"
        return "LAS"
    elseif model_name == "Ridge"
        return "RID"
    elseif model_name == "Baseline"
        return "BAS"
    elseif model_name == "Random"
        return "RGM"
    elseif model_name == "MiddleGBM"
        return "MDG"
    elseif model_name == "PyGBM"
        return "PYG"
    else
        JobLoggerTools.error_benji("Unknown model name: $model_name", jobid)
    end
end

"""
    make_X_Y(
        X_list::Vector{String}, 
        Y::String
    ) -> String

Generate a string representation of input → output relation, useful for filenames.

# Arguments
- `X_list`: Vector of input feature names.
- `Y`: Output target name.

# Returns
- If single input equals output, returns `Y`.
- Else, returns `"X1_X2_Y"` format string.
"""
function make_X_Y(
    X_list::Vector{String}, 
    Y::String
)::String
    if length(X_list) == 1 && X_list[1] == Y
        return Y
    else
        return join(X_list, "_") * "_" * Y
    end
end

"""
    build_trace_name(
        ensemble::String,
        io_decode::String,
        X_str::Vector{String},
        Y_str::String,
        N_lb_str::String,
        N_tr_str::String,
        model_tag::String
    ) -> String

Construct a standardized trace name used for storing results or logs.

# Arguments
- `ensemble`   : Ensemble name.
- `io_decode`  : Encoded or decoded input/output string.
- `X_str`      : Vector of input feature names.
- `Y_str`      : Output target name.
- `N_lb_str`   : `LB` set size string.
- `N_tr_str`   : `TR` set size string.
- `model_tag`  : Short model tag (e.g., `"GBM"`, `"RID"`).

# Returns
- Formatted name string of the form: `ensemble_io_decode_MODEL_LBP_x_TRP_y`
"""
function build_trace_name(
    ensemble::String,
    io_decode::String,
    X_str::Vector{String},
    Y_str::String,
    N_lb_str::String,
    N_tr_str::String,
    model_tag::String
)::String
    suffix = (length(X_str) == 1 && X_str[1] == Y_str) ? "_BAS" : "_" * model_tag
    return ensemble * "_" * io_decode * suffix * "_LBP_" * N_lb_str * "_TRP_" * N_tr_str
end

end  # module NameParser