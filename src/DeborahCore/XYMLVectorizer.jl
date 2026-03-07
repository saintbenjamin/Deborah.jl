# ============================================================================
# src/DeborahCore/XYMLVectorizer.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module XYMLVectorizer

import ..Printf: @printf

"""
    mat_XY_ML(
        X_vec::AbstractVector{T}, 
        N_src::Int, 
        N_set::Int
    ) -> Matrix{T}

Reshape a flattened ML input vector into a 2D matrix of shape ``(N_\\text{set}, N_\\text{src})``.

# Arguments
- `X_vec::AbstractVector{T}`: Flattened 1D array (``\\text{length} = N_\\text{set} \\times N_\\text{src}``)
- `N_src::Int`: Number of sources (columns in output matrix)
- `N_set::Int`: Number of configurations (rows in output matrix)

# Returns
- `Matrix{T}`: Reconstructed matrix of shape ``(N_\\text{set}, N_\\text{src})``
"""
function mat_XY_ML(
    X_vec::AbstractVector{T},
    N_src::Int,
    N_set::Int
)::Matrix{T} where T<:Real
    if N_src == 0 || N_set == 0 || isempty(X_vec)
        return Matrix{T}(undef, 0, 0)
    end
    X_mat = Matrix{T}(undef, N_set, N_src)
    for iconf in 1:N_set
        for jsrc in 1:N_src
            X_mat[iconf, jsrc] = X_vec[(iconf - 1) * N_src + jsrc]
        end
    end
    return X_mat
end

"""
    gen_XY_ML(
        X_mat::AbstractArray{T,3},
        read_column::Int,
        conf_arr::AbstractVector{Int},
        X_label::String,
        overall_name::String,
        analysis_dir::String;
        use_avg::Bool = true,
        dump::Bool = true
    ) -> Vector{T}

Flatten a slice of a 3D input array into a 1D vector and optionally write it to a `.dat` file.

# Arguments
- `X_mat::AbstractArray{T,3}`: Input data of shape ``(\\texttt{column\\_idx}, N_\\text{set}, N_\\text{src})``
- `read_column::Int`: Index to select along the 1st dimension
- `conf_arr::AbstractVector{Int}`: Configuration number array, ``\\text{length} = N_\\text{set}``
- `X_label::String`: Prefix for output filename
- `overall_name::String`: Suffix for output filename
- `use_avg::Bool`: If true, treat as averaged source data (`jsrc = -1`)
- `dump::Bool`: If true, write a .dat file

# Returns
- `Vector{T}`: Flattened vector of selected component from `X_mat`
"""
function gen_XY_ML(
    X_mat::AbstractArray{T,3},
    read_column::Int,
    conf_arr::AbstractVector{Int},
    X_label::String,
    overall_name::String,
    analysis_dir::String;
    use_avg::Bool = true,
    dump::Bool = true
)::Vector{T} where T<:Real

    N_set = size(X_mat, 2)
    N_src = size(X_mat, 3)

    X_vec = Vector{T}(undef, N_set * N_src)

    if dump
        mkpath(analysis_dir)
        X_dat_file = analysis_dir*"/"*X_label * "_" * overall_name * ".dat"
        open(X_dat_file, "w") do io_X
            for iconf in 1:N_set
                for jsrc in 1:N_src
                    val = X_mat[read_column, iconf, jsrc]
                    X_vec[(iconf - 1) * N_src + jsrc] = val
                    jval = use_avg ? "A" : string(jsrc - 1)
                    @printf(io_X, "%.14e\t%d\t%s\t%d\n", val, conf_arr[iconf], jval, iconf)
                end
            end
        end
    else
        for iconf in 1:N_set
            for jsrc in 1:N_src
                X_vec[(iconf - 1) * N_src + jsrc] = X_mat[read_column, iconf, jsrc]
            end
        end
    end

    return X_vec
end

"""
    gen_XY_ML(
        X_mat::AbstractArray{T,2}, 
        conf_arr::AbstractVector{Int}, 
        X_label::String, 
        overall_name::String,
        analysis_dir::String; 
        use_avg::Bool=true, 
        dump::Bool=true
    ) -> Vector{T}

Convert a 2D input matrix into a flattened vector and optionally dump it to a `.dat` file for ML processing.

# Arguments
- `X_mat::AbstractArray{T,2}`: Input matrix of shape ``(N_\\text{set}, N_\\text{src})``.
- `conf_arr::AbstractVector{Int}`: Array of configuration indices, length ``N_\\text{set}``.
- `X_label::String`: Label prefix for the output file.
- `overall_name::String`: Suffix for the output file name.
- `use_avg::Bool=true`: If true, assumes averaged source and marks `jsrc` as `-1`.
- `dump::Bool=true`: Whether to write the output to a `.dat` file.

# Returns
- `Vector{T}`: Flattened 1D array of input values.
"""
function gen_XY_ML(
    X_mat::AbstractArray{T,2},
    conf_arr::AbstractVector{Int},
    X_label::String,
    overall_name::String,
    analysis_dir::String;
    use_avg::Bool = true,
    dump::Bool = true
)::Vector{T} where T<:Real

    N_set, N_src = size(X_mat)
    X_vec = Vector{T}(undef, N_set * N_src)

    if dump
        mkpath(analysis_dir)
        X_dat_file = analysis_dir*"/"*X_label * "_" * overall_name * ".dat"
        open(X_dat_file, "w") do io_X
            is_explicit_zero = all(x -> x == 0.0, X_mat)
            for iconf in 1:N_set
                for jsrc in 1:N_src
                    idx = (iconf - 1) * N_src + jsrc
                    X_vec[idx] = X_mat[iconf, jsrc]
                    jval = use_avg ? "A" : string(jsrc - 1)
                    if !is_explicit_zero
                        @printf(io_X, "%.14e\t%d\t%s\t%d\n", X_mat[iconf, jsrc], conf_arr[iconf], jval, iconf)
                    end
                end
            end
        end
    else
        for iconf in 1:N_set
            for jsrc in 1:N_src
                idx = (iconf - 1) * N_src + jsrc
                X_vec[idx] = X_mat[iconf, jsrc]
            end
        end
    end

    return X_vec
end

end  # module XYMLVectorizer