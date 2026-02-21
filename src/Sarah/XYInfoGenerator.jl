# ============================================================================
# src/Sarah/XYInfoGenerator.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module XYInfoGenerator

"""
    gen_conf_from_Y(
        Y_df::AbstractArray{T,2}, 
        N_cnf::Int, 
        N_src::Int, 
        index_column::Int
    ) -> Vector{Int}

Extract configuration indices from a specific column in a 2D array representing `Y` data.

This function scans the `Y` file data and retrieves configuration numbers at fixed intervals,
typically from every `N_src`-th row. If `index_column > 0`, values are extracted from that
specific column. If `index_column == 0`, default configuration numbers `1:N_cnf` are assigned.

# Arguments
- `Y_df::AbstractArray{T,2}`  
    Raw data array loaded from a `Y` file. Must have at least `N_cnf * N_src` rows and enough columns.  
    Expected shape: `(N_cnf` ``\\times`` `N_src,` ``\\ge`` `index_column)`.

- `N_cnf::Int`  
    Number of unique configurations (i.e., the number of groups in the `Y` data).

- `N_src::Int`  
    Number of sources (rows) per configuration.

- `index_column::Int`  
    ``1``-based column index from which to extract configuration numbers.  
    If set to `0`, configuration numbers are auto-generated as `1:N_cnf`.

# Returns
- `Vector{Int}`  
    Vector of configuration indices of length `N_cnf`.

# Examples
```julia
Y = rand(5500, 3)
conf = gen_conf_from_Y(Y, 1100, 5, 3)  # extract from 3rd column
```
"""
function gen_conf_from_Y(
    Y_df::AbstractArray{T,2}, 
    N_cnf::Int, 
    N_src::Int,
    index_column::Int
)::Vector{Int} where {T<:Real}

    conf_arr = Vector{Int}(undef, N_cnf)

    if index_column > 0
        for iconf in 1:N_cnf
            row_idx = (iconf - 1) * N_src + 1
            conf_arr[iconf] = Int(Y_df[row_idx, index_column])
        end
    else
        # Generate default config IDs: 1, 2, ..., N_cnf
        for iconf in 1:N_cnf
            conf_arr[iconf] = iconf
        end
    end

    return conf_arr
end

"""
    gen_X_info(
        X_df::AbstractArray{T,2}, 
        N_cnf::Int, 
        N_src::Int, 
        read_column::Int
    ) -> Array{T,2}

Extract a single component (e.g., real or imaginary) from a specific column of the `X` data matrix,
and reshape it into a 2D array of shape `(N_cnf, N_src)`.

This function ignores multi-channel assumptions (such as real/imaginary separation)
and simply uses the specified `read_column`.

# Arguments
- `X_df::AbstractArray{T,2}`  
    Raw input data matrix from an `X` file. Shape must be `(N_cnf` ``\\times`` `N_src,` ``\\ge`` `read_column)`.

- `N_cnf::Int`  
    Number of configurations (i.e., groups of sources in `X`).

- `N_src::Int`  
    Number of sources per configuration.

- `read_column::Int`  
    ``1``-based column index to extract values from.

# Returns
- `Array{T,2}`  
    A 2D array of shape `(N_cnf, N_src)` containing values from the selected column only.
"""
function gen_X_info(
    X_df::AbstractArray{T,2},
    N_cnf::Int,
    N_src::Int,
    read_column::Int
)::Array{T,3} where {T<:Real}

    X_info = zeros(1, N_cnf, N_src)

    for iconf in 1:N_cnf
        for jsrc in 1:N_src
            idx = (iconf - 1) * N_src + jsrc
            X_info[read_column, iconf, jsrc] = X_df[idx, read_column]
        end
    end

    return X_info
end

end  # module XYInfoGenerator