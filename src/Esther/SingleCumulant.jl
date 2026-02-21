# ============================================================================
# src/Esther/SingleCumulant.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SingleCumulant

"""
    calc_quark_condensate(
        Q1::AbstractArray, 
        LatVol::Int
    ) -> Vector{Float64}

Compute the chiral condensate ``\\left\\langle \\bar{\\psi} \\psi \\right\\rangle`` using ``Q_1`` values by normalizing with the lattice volume.

# Formula
```math
\\Sigma = \\frac{\\left\\langle Q_1 \\right\\rangle}{V}
```

# Arguments
- `Q1::AbstractArray`: Vector of ``Q_1`` values for each bootstrap resample.
- `LatVol::Int`: Lattice volume (``V = N_{S}^3 \\times N_{T}``).

# Returns
- `Vector{Float64}`: Quark condensate values per resample, normalized by volume.
"""
function calc_quark_condensate(
    Q1::AbstractArray, 
    LatVol::Int
)
    N_bs = length(Q1)
    X_out = zeros(Float64, N_bs)
    for ibs in 1:N_bs
        X_out[ibs] = Q1[ibs] / Float64(LatVol)
    end
    return X_out
end

"""
    calc_susceptibility(
        Q1::AbstractArray, 
        Q2::AbstractArray, 
        LatVol::Int
    ) -> Vector{Float64}

Compute the chiral susceptibility using ``Q_1`` and ``Q_2`` values by normalizing with the lattice volume.

# Formula
```math
\\chi = \\frac{\\left\\langle Q_2 \\right\\rangle - \\left\\langle Q_1 \\right\\rangle^2}{V}
```

# Arguments
- `Q1::AbstractArray`: Vector of ``Q_1`` values for each bootstrap resample.
- `Q2::AbstractArray`: Vector of ``Q_2`` values for each bootstrap resample.
- `LatVol::Int`: Lattice volume (``V = N_{S}^3 \\times N_{T}``).

# Returns
- `Vector{Float64}`: Susceptibility values per resample.
"""
function calc_susceptibility(
    Q1::AbstractArray, 
    Q2::AbstractArray, 
    LatVol::Int
)
    N_bs = length(Q1)
    X_out = zeros(Float64, N_bs)
    for ibs in 1:N_bs
        sigma = Q2[ibs] - (Q1[ibs])^2
        X_out[ibs] = sigma / Float64(LatVol)
    end
    return X_out
end

"""
    calc_skewness(
        Q1::AbstractArray, 
        Q2::AbstractArray, 
        Q3::AbstractArray
    ) -> Vector{Float64}

Compute the skewness of chiral condensate using ``Q_1``, ``Q_2`` and ``Q_3`` values.

# Formula
```math
S = \\frac{  
    \\left\\langle Q_3 \\right\\rangle 
    - 3 \\, \\left\\langle Q_2 \\right\\rangle \\, \\left\\langle Q_1 \\right\\rangle
    + 2 \\, \\left\\langle Q_1 \\right\\rangle^3
}{\\left( \\left\\langle Q_2 \\right\\rangle - \\left\\langle Q_1 \\right\\rangle^2 \\right)^{\\frac{3}{2}}}
```

# Arguments
- `Q1::AbstractArray`: Vector of ``Q_1`` values for each bootstrap resample.
- `Q2::AbstractArray`: Vector of ``Q_2`` values for each bootstrap resample.
- `Q3::AbstractArray`: Vector of ``Q_3`` values for each bootstrap resample.

# Returns
- `Vector{Float64}`: Skewness values per resample.
"""
function calc_skewness(Q1::AbstractArray, Q2::AbstractArray, Q3::AbstractArray)
    N_bs = length(Q1)
    X_out = zeros(Float64, N_bs)
    for ibs in 1:N_bs
        sigma = abs(Q2[ibs] - (Q1[ibs])^2)
        X_out[ibs] = (Q3[ibs] - 3.0 * Q2[ibs] * Q1[ibs] + 2.0 * (Q1[ibs])^3) / sigma^(1.5)
    end
    return X_out
end

"""
    calc_kurtosis(
        Q1::AbstractArray, 
        Q2::AbstractArray, 
        Q3::AbstractArray, 
        Q4::AbstractArray
    ) -> Vector{Float64}

Compute the kurtosis of chiral condensate using ``Q_1``, ``Q_2``, ``Q_3`` and ``Q_4`` values.

# Formula
```math
K = \\frac{  
    \\left\\langle Q_4 \\right\\rangle 
    - 4 \\, \\left\\langle Q_3 \\right\\rangle \\, \\left\\langle Q_1 \\right\\rangle
    - 3 \\, \\left\\langle Q_2 \\right\\rangle^2
    + 12 \\, \\left\\langle Q_2 \\right\\rangle \\, \\left\\langle Q_1 \\right\\rangle^2
    - 6 \\, \\left\\langle Q_1 \\right\\rangle^4    
}{\\left( \\left\\langle Q_2 \\right\\rangle - \\left\\langle Q_1 \\right\\rangle^2 \\right)^{2}}
```

# Arguments
- `Q1::AbstractArray`: Vector of ``Q_1`` values for each bootstrap resample.
- `Q2::AbstractArray`: Vector of ``Q_2`` values for each bootstrap resample.
- `Q3::AbstractArray`: Vector of ``Q_3`` values for each bootstrap resample.
- `Q4::AbstractArray`: Vector of ``Q_4`` values for each bootstrap resample.

# Returns
- `Vector{Float64}`: Kurtosis values per resample.
"""
function calc_kurtosis(
    Q1::AbstractArray, 
    Q2::AbstractArray, 
    Q3::AbstractArray, 
    Q4::AbstractArray
)
    N_bs = length(Q1)
    X_out = zeros(Float64, N_bs)
    for ibs in 1:N_bs
        sigma = Q2[ibs] - (Q1[ibs])^2
        X_out[ibs] = (
            Q4[ibs]
            - 4.0 * Q3[ibs] * Q1[ibs]
            - 3.0 * (Q2[ibs])^2
            + 12.0 * Q2[ibs] * (Q1[ibs])^2
            - 6.0 * (Q1[ibs])^4
        ) / sigma^2
    end
    return X_out
end

end  # module SingleCumulant