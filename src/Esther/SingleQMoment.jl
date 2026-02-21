# ============================================================================
# src/Esther/SingleQMoment.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SingleQMoment

"""
    calc_Q1(
        trM1::AbstractArray, 
        Nf::Int
    ) -> Vector{Float64}

Compute the first-order quark cumulant ``Q_1`` using:

```math
Q_1 = N_{\\text{f}} \\, \\text{Tr} \\, M^{-1}
```

# Arguments
- `trM1::AbstractArray`: ``\\text{Tr} \\, M^{-1}`` for each configuration index.
- `Nf::Int`: Number of quark flavors.

# Returns
- `Vector{Float64}`: ``Q_1`` values for all configuration indices.
"""
function calc_Q1(
    trM1::AbstractArray, 
    Nf::Int
)
    if isempty(trM1)
        return Float64[]
    end
    N_cnf = length(trM1)
    X_out = zeros(Float64, N_cnf)
    for idx in 1:N_cnf
        X_out[idx] = Nf * trM1[idx]
    end
    return X_out
end

"""
    calc_Q2(
        trM1::AbstractArray, 
        trM2::AbstractArray, 
        Nf::Int
    ) -> Vector{Float64}

Compute the second-order quark cumulant ``Q_2`` using:

```math
Q_2 = - N_{\\text{f}} \\, \\text{Tr} \\, M^{-2} + \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\right)^2
```

# Arguments
- `trM1::AbstractArray`: ``\\text{Tr} \\, M^{-1}`` for each configuration index.
- `trM2::AbstractArray`: ``\\text{Tr} \\, M^{-2}`` for each configuration index.
- `Nf::Int`: Number of quark flavors.

# Returns
- `Vector{Float64}`: ``Q_2`` values for all configuration indices.
"""
function calc_Q2(
    trM1::AbstractArray, 
    trM2::AbstractArray, 
    Nf::Int
)
    if isempty(trM1) || isempty(trM2)
        return Float64[]
    end
    N_cnf = length(trM1)
    X_out = zeros(Float64, N_cnf)
    for idx in 1:N_cnf
        X_out[idx] = -Nf * trM2[idx] + (Nf * trM1[idx])^2
    end
    return X_out
end

"""
    calc_Q3(
        trM1::AbstractArray, 
        trM2::AbstractArray, 
        trM3::AbstractArray, 
        Nf::Int
    ) -> Vector{Float64}

Compute the third-order quark cumulant ``Q_3`` using:

```math
Q_3 = 2 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-3}
   - 3 \\, N_{\\text{f}}^{2} \\, \\text{Tr} \\, M^{-2} \\, \\text{Tr}\\, M^{-1}
   + \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\right)^{3}
```

# Arguments
- `trM1::AbstractArray`: ``\\text{Tr} \\, M^{-1}`` for each configuration index.
- `trM2::AbstractArray`: ``\\text{Tr} \\, M^{-2}`` for each configuration index.
- `trM3::AbstractArray`: ``\\text{Tr} \\, M^{-3}`` for each configuration index.
- `Nf::Int`: Number of quark flavors.

# Returns
- `Vector{Float64}`: ``Q_3`` values for all configuration indices.
"""
function calc_Q3(
    trM1::AbstractArray, 
    trM2::AbstractArray, 
    trM3::AbstractArray, 
    Nf::Int
)
    if isempty(trM1) || isempty(trM2) || isempty(trM3)
        return Float64[]
    end
    N_cnf = length(trM1)
    X_out = zeros(Float64, N_cnf)
    for idx in 1:N_cnf
        X_out[idx] = 2.0 * Nf * trM3[idx] - 3.0 * Nf * trM2[idx] * Nf * trM1[idx] + (Nf * trM1[idx])^3
    end
    return X_out
end

"""
    calc_Q4(
        trM1::AbstractArray, 
        trM2::AbstractArray, 
        trM3::AbstractArray, 
        trM4::AbstractArray, 
        Nf::Int
    ) -> Vector{Float64}

Compute the fourth-order quark cumulant Q₄ using:

```math
Q_4 = - 6 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-4}
   + 8 \\, N_{\\text{f}}^{2} \\, \\text{Tr} \\, M^{-3} \\, \\text{Tr} \\, M^{-1} 
   + 3 \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-2} \\right)^{2}
   - 6 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-2} \\, \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\right)^{2}
   + \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\right)^{4}
```

# Arguments
- `trM1::AbstractArray`: ``\\text{Tr} \\, M^{-1}`` for each configuration index.
- `trM2::AbstractArray`: ``\\text{Tr} \\, M^{-2}`` for each configuration index.
- `trM3::AbstractArray`: ``\\text{Tr} \\, M^{-3}`` for each configuration index.
- `trM4::AbstractArray`: ``\\text{Tr} \\, M^{-4}`` for each configuration index.
- `Nf::Int`: Number of quark flavors.

# Returns
- `Vector{Float64}`: ``Q_4`` values for all configuration indices.
"""
function calc_Q4(
    trM1::AbstractArray, 
    trM2::AbstractArray, 
    trM3::AbstractArray, 
    trM4::AbstractArray, 
    Nf::Int
)
    if isempty(trM1) || isempty(trM2) || isempty(trM3) || isempty(trM4)
        return Float64[]
    end
    N_cnf = length(trM1)
    X_out = zeros(Float64, N_cnf)
    for idx in 1:N_cnf
        X_out[idx] = -6.0 * Nf * trM4[idx] +
                     8.0 * Nf * trM3[idx] * Nf * trM1[idx] +
                     3.0 * (Nf * trM2[idx])^2 -
                     6.0 * Nf * trM2[idx] * (Nf * trM1[idx])^2 +
                     (Nf * trM1[idx])^4
    end
    return X_out
end

end  # module SingleQMoment