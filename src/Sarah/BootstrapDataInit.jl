# ============================================================================
# src/Sarah/BootstrapDataInit.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BootstrapDataInit

import ..XYInfoGenerator

"""
    convert_matrix_to_vec_list(
        mat::Matrix{T}
    ) -> Vector{Vector{T}} where T<:Real

Convert a matrix of shape (`N_conf`, `N_src`) to a list of vectors per source.

Each vector in the output corresponds to a source.

# Arguments
- `mat`: `Matrix{Float64}` of shape (`N_conf`, `N_src`)

# Returns
- `Vector{Vector{Float64}}`: list of `N_src` vectors, each of length `N_conf`
"""
function convert_matrix_to_vec_list(
    mat::Matrix{T}
)::Vector{Vector{T}} where T<:Real
    N_conf, N_src = size(mat)
    vec_list = Vector{Vector{T}}(undef, N_src)
    for jsrc in 1:N_src
        vec_list[jsrc] = mat[:, jsrc]
    end
    return vec_list
end

"""
    build_trace_data(
        Y_mats::Dict{Symbol, Matrix{Float64}}, 
        Y_df::Matrix{Float64}, 
        N_cnf::Int, 
        N_src::Int, 
        read_column_Y::Int
    ) -> Dict{String, Vector{Vector{Float64}}}

Constructs trace data vectors from raw measurement matrices and metadata.

# Arguments
- `Y_mats::Dict{Symbol, Matrix{Float64}}`: Dictionary with keys like `:Y_lb`, `:Y_bc`, `:Y_ul`, `:YP_bc`, `:YP_ul`, each mapped to a `(N_bs` ``\\times`` `N_cnf)` matrix of measurements.
- `Y_df::Matrix{Float64}`: Raw metadata matrix from which `Y_info` is generated.
- `N_cnf::Int`: Number of configurations.
- `N_src::Int`: Number of source points per configuration.
- `read_column_Y::Int`: Index of the column in `Y_df` to extract as label information.

# Returns
- `Dict{String, Vector{Vector{Float64}}}`: Dictionary where each key (e.g., `"Y_bc"`) maps to a list of vectors of size `N_bs`, one per configuration. Includes an additional key `"Y_info"` for label metadata.

# Notes
Each `Y_mats[label]` is converted using [`convert_matrix_to_vec_list`](@ref), which splits the `(N_bs` ``\\times`` `N_cnf)` matrix into a vector of length `N_cnf`, each holding a bootstrap vector of size `N_bs`.
"""
function build_trace_data(
    Y_mats::Dict{Symbol, Matrix}, 
    Y_df::Matrix{Float64}, 
    N_cnf::Int, 
    N_src::Int, 
    read_column_Y::Int
)
    trace_data = Dict{String, Vector{Vector{Float64}}}()

    labels = ["Y_lb", "Y_bc", "Y_ul", "YP_bc", "YP_ul"]
    for label in labels
        trace_data[label] = convert_matrix_to_vec_list(Y_mats[Symbol(label)])
    end

    Y_info_mat = XYInfoGenerator.gen_X_info(Y_df, N_cnf, N_src, read_column_Y)
    trace_data["Y_info"] = [vec(Y_info_mat[1, :, jsrc]) for jsrc in 1:N_src]

    return trace_data
end

"""
    init_bootstrap_data(
        N_bs::Int, 
        ::Type{T}
    ) where T<:Real -> Dict{Symbol, Any}

Initialize a dictionary for bootstrap data with preallocated `:mean` field.

# Arguments
- `N_bs::Int`: Number of bootstrap samples.
- `T<:Real`: Numeric type for stored values (e.g., `Float64`).

# Returns
- A dictionary with key `:mean` mapping to another dictionary of label → zeroed vector of length `N_bs`.
"""
function init_bootstrap_data(
    N_bs::Int, 
    ::Type{T}
)::Dict{Symbol, Any} where T<:Real
    base_labels = ["Y_info","Y_lb", "Y_bc", "Y_ul", "YP_bc", "YP_ul"]
    derived_labels = ["YmYP", "Y_P1", "Y_P2"]

    bd = Dict{Symbol, Any}()
    bd[:mean] = Dict{String, Vector{T}}()

    for label in vcat(base_labels, derived_labels)
        bd[:mean][label] = zeros(T, N_bs)
    end

    return bd
end

"""
    init_bootstrap_data_cumulant(
        N_bs::Int
    ) -> Dict{Symbol, Any}

Initialize a dictionary for cumulant-style bootstrap data with `:mean` field
preallocated for all ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` and ``Q``-moment variants.

# Arguments
- `N_bs::Int`: Number of bootstrap resamples.

# Returns
- A dictionary with key `:mean` mapping to another dictionary of
  label → zeroed vector of length `N_bs`. The labels are prefixed with
  `trM1-4:` and `Q1-4:` for each observable group.
"""
function init_bootstrap_data_cumulant(
    N_bs::Int
)::Dict{Symbol, Any}
    base_labels = ["Y_info", "Y_lb", "Y_bc", "Y_ul", "YP_bc", "YP_ul"]
    derived_labels = ["YmYP", "Y_P1", "Y_P2"]
    models_trM = ["trM1", "trM2", "trM3", "trM4"]

    bd = Dict{Symbol, Any}()
    bd[:mean] = Dict{String, Vector{Float64}}()

    for model in models_trM
        for label in base_labels
            bd[:mean]["$(model):$(label)"] = zeros(N_bs)
        end
        for label in derived_labels
            bd[:mean]["$(model):$(label)"] = zeros(N_bs)
        end
    end

    for q in 1:4
        for label in base_labels
            bd[:mean]["Q$(q):$(label)"] = zeros(N_bs)
        end
        for label in derived_labels
            bd[:mean]["Q$(q):$(label)"] = zeros(N_bs)
        end
    end

    return bd
end

end  # module BootstrapDataInit