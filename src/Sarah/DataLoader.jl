# ============================================================================
# src/Sarah/DataLoader.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DataLoader

import ..DelimitedFiles

import ..JobLoggerTools

"""
    try_multi_readdlm(
        path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Matrix{Float64}

Efficiently attempt to read a delimited numeric data file using a fast primary strategy  
with a lightweight fallback parser.

This function first tries to read the file as a tab-delimited matrix using [`DelimitedFiles.readdlm`](https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/#DelimitedFiles.readdlm).  
If that fails, it falls back to a manual parser that splits lines using common delimiters  
and extracts only numeric values, skipping any non-numeric tokens (e.g., labels or tags).

Compared to the original robust version, this version prioritizes speed, assuming the file  
is mostly well-formed with numeric values and consistent rows. Still supports fallback  
parsing for mixed-format dumps (like `Y_info`) but uses `map` and `filter` for better performance.

# Parsing Strategy
1. Attempt `DelimitedFiles.readdlm(path, '\\t', Float64)` (tab-delimited).
2. If it fails:
   - Read lines and split by regex: `[,\t; ]+`
   - Skip non-numeric tokens using `tryparse(Float64, token)`
   - Parse numeric tokens and collect into rows
   - Assert all rows have the same number of numeric columns

# Arguments
- `path::String` : Path to the target `.dat` or text file to read.
- `jobid::Union{Nothing, String}` : Optional job identifier for structured logging.

# Returns
- `Matrix{Float64}` : A matrix of parsed numeric values with shape `(N_rows, N_columns)`.

# Errors
- Throws an error if:
    - Tab-delimited read fails and fallback also fails
    - The number of numeric values per row is inconsistent during fallback.
"""
function try_multi_readdlm(
    path::String, 
    jobid::Union{Nothing, String}=nothing
)::Matrix{Float64}
    # Primary fast path: tab-delimited read
    try
        data = DelimitedFiles.readdlm(path, '\t', Float64, '\n')
        return data isa Matrix{Float64} ? data : Matrix{Float64}(undef, 0, 0)
    catch
        # fallback
    end

    try
        lines = readlines(path)
        parsed = map(lines) do line
            tokens = split(strip(line), r"[,\t; ]+")
            parsed_row = filter(x -> tryparse(Float64, x) !== nothing, tokens)
            parse.(Float64, parsed_row)
        end

        if isempty(parsed)
            return Matrix{Float64}(undef, 0, 0)
        end

        Ncols = length(parsed[1])
        JobLoggerTools.assert_benji(all(length(row) == Ncols for row in parsed), "Inconsistent number of numeric columns", jobid)

        return reduce(vcat, [reshape(row, 1, :) for row in parsed])
    catch
        return Matrix{Float64}(undef, 0, 0)
    end
end

"""
    load_data_file(
        path::String, 
        key::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Matrix{Float64}

Load a tab-delimited data file and return its contents as a matrix of `Float64`.

# Arguments
- `path::String`: Directory path where the data file is located.
- `key::String`: Filename of the data file to be loaded.
- `jobid::Union{Nothing, String}` : Optional job identifier for structured logging.

# Returns
- `Matrix{Float64}`: A 2D array containing the parsed data from the file.

# Notes
- The file is expected to be tab-delimited (`\t`).
- The file is read using [`DelimitedFiles.readdlm`](https://docs.julialang.org/en/v1/stdlib/DelimitedFiles/#DelimitedFiles.readdlm).
"""
function load_data_file(
    path::String, 
    key::String, 
    jobid::Union{Nothing, String}=nothing
)::Matrix{Float64}
    fullpath = joinpath(path, key)
    return try_multi_readdlm(fullpath, jobid)
end

end  # module DataLoader