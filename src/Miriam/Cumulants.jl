# ============================================================================
# src/Miriam/Cumulants.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module Cumulants

import ..Sarah.JobLoggerTools
import ..Sarah.Jackknife

"""
    calc_trace(
        trMi::Vector{T}
    ) where T -> (real(T), real(T), real(T), real(T))

Extract the first four raw trace components (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) from a trace vector.

# Arguments
- `trMi::Vector{T}`: Trace vector where indices `2..5` contain
  the four trace components:
  - `trMi[2]` is ``\\text{Tr} \\, M^{-1}``
  - `trMi[3]` is ``\\text{Tr} \\, M^{-2}``
  - `trMi[4]` is ``\\text{Tr} \\, M^{-3}``
  - `trMi[5]` is ``\\text{Tr} \\, M^{-4}``

# Returns
- `trM1::real(T)`, `trM2::real(T)`, `trM3::real(T)`, `trM4::real(T)`

# Notes
- Requires `length(trMi)` ``\\ge 5``.
"""
function calc_trace(
    trMi::Vector{T}
) where T
    trM1_out = real(trMi[2])
    trM2_out = real(trMi[3])
    trM3_out = real(trMi[4])
    trM4_out = real(trMi[5])

    return trM1_out, trM2_out, trM3_out, trM4_out
end

"""
    calc_Q(
        trMi::Vector{T}
    ) where T -> (real(T), real(T), real(T), real(T))

Compute the first four moments ``Q_n \\; (n=1,2,3,4)`` from ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` vectors.

# Arguments
- `trMi::Vector{T}`: Vector containing traces, where
    - `trMi[2]` is ``\\text{Tr} \\, M^{-1}``
    - `trMi[3]` is ``\\text{Tr} \\, M^{-2}``
    - `trMi[4]` is ``\\text{Tr} \\, M^{-3}``
    - `trMi[5]` is ``\\text{Tr} \\, M^{-4}``

# Formula
```math
\\begin{aligned}
Q_1 &= N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\\\
Q_2 &= - N_{\\text{f}} \\, \\text{Tr} \\, M^{-2} + \\left( \\text{Tr} \\, M^{-1} \\right)^2 \\\\
Q_3 &= 2 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-3}
   - 3 \\, N_{\\text{f}}^{2} \\, \\text{Tr} \\, M^{-2} \\, \\text{Tr}\\, M^{-1}
   + \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\right)^{3} \\\\
Q_4 &= - 6 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-4}
   + 8 \\, N_{\\text{f}}^{2} \\, \\text{Tr} \\, M^{-3} \\, \\text{Tr} \\, M^{-1} 
   + 3 \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-2} \\right)^{2}
   - 6 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-2} \\, \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\right)^{2}
   + \\left( N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\right)^{4}
\\end{aligned}
```

# Notes
Unlike this [`Deborah.Miriam`](@ref), [`Deborah.Esther`](@ref) uses slightly different definitions for ``Q_{n} \\; (n=1,2,3,4)``.
In this implementation, the factor of ``N_{\\text{f}}`` does not explicitly appear,  
because the trace rescaling step already incorporates ``N_{\\text{f}}`` into the normalization.  

See also:  
- [`Deborah.Miriam.MultiEnsembleLoader.generate_trMi_vector`](@ref)  
- [`Deborah.Esther.TraceRescaler.rescale_trace`](@ref) (for comparison of rescaling conventions)

# Returns
- `Q1::T`: ``Q_1`` moment
- `Q2::T`: ``Q_2`` moment
- `Q3::T`: ``Q_3`` moment
- `Q4::T`: ``Q_4`` moment
"""
function calc_Q(
    trMi::Vector{T}
) where T
    trMi1_2 = trMi[2] * trMi[2]

    Q1_out = real(trMi[2])
    Q2_out = real(-trMi[3] + trMi1_2)
    Q3_out = real(2 * trMi[4] - 3 * trMi[2] * trMi[3] + trMi[2] * trMi1_2)
    Q4_out = real(-6 * trMi[5] + 8 * trMi[2] * trMi[4] + 3 * trMi[3] * trMi[3] -
                  6 * trMi1_2 * trMi[3] + trMi1_2 * trMi1_2)

    return Q1_out, Q2_out, Q3_out, Q4_out
end

"""
    @inline @fastmath calc_cumulants(
        V::T, 
        Q1::T, 
        Q2::T, 
        Q3::T, 
        Q4::T, 
        jobid::Union{Nothing, String}=nothing
    ) where T -> (T, T, T, T)

Compute statistical cumulants from trace moments ``Q_1`` through ``Q_4``.

# Arguments
- `V::T`: Volume (typically ``N_S^3 \\times N_T``), used for normalization.
- `Q1::T`: First moment ``Q_1``
- `Q2::T`: Second moment ``Q_2``
- `Q3::T`: Third moment ``Q_3``
- `Q4::T`: Fourth moment ``Q_4``
- `jobid::Union{Nothing, String}`  : Optional job ID for logging.

# Formula
```math
\\begin{align*}
\\chi &= \\frac{\\left\\langle Q_2 \\right\\rangle - \\left\\langle Q_1 \\right\\rangle^2}{V} \\\\
S &= \\frac{  
    \\left\\langle Q_3 \\right\\rangle 
    - 3 \\, \\left\\langle Q_2 \\right\\rangle \\, \\left\\langle Q_1 \\right\\rangle
    + 2 \\, \\left\\langle Q_1 \\right\\rangle^3
}{\\left( \\left\\langle Q_2 \\right\\rangle - \\left\\langle Q_1 \\right\\rangle^2 \\right)^{\\frac{3}{2}}} \\\\
K &= \\frac{  
    \\left\\langle Q_4 \\right\\rangle 
    - 4 \\, \\left\\langle Q_3 \\right\\rangle \\, \\left\\langle Q_1 \\right\\rangle
    - 3 \\, \\left\\langle Q_2 \\right\\rangle^2
    + 12 \\, \\left\\langle Q_2 \\right\\rangle \\, \\left\\langle Q_1 \\right\\rangle^2
    - 6 \\, \\left\\langle Q_1 \\right\\rangle^4    
}{\\left( \\left\\langle Q_2 \\right\\rangle - \\left\\langle Q_1 \\right\\rangle^2 \\right)^{2}} \\\\
B &= 1 - \\frac{\\left\\langle Q_4 \\right\\rangle}{3 \\, \\left\\langle Q_2 \\right\\rangle^2}
\\end{align*}
```

# Returns
- `susceptibility::T`: susceptibility of chiral condensate (``\\chi``)
- `skewness::T`: skewness of chiral condensate (measures asymmetry)
- `kurtosis::T`: kurtosis of chiral condensate (measures tailedness)
- `binder_cumulant::T`: Binder cumulant

# Notes
- Prevents division by near-zero variance using `eps(T)` safeguard.
- Assumes input types are floating point (e.g., `Float64`, `Float32`).
"""
@inline @fastmath function calc_cumulants(
    V::T, 
    Q1::T, 
    Q2::T, 
    Q3::T, 
    Q4::T, 
    jobid::Union{Nothing, String}=nothing
) where T
    # Precompute powers of Q1 and Q2
    Q1_2 = Q1^2
    Q1_3 = Q1 * Q1_2
    Q1_4 = Q1_2^2
    Q2_2 = Q2^2

    # Compute variance-like quantity σ = ⟨x²⟩ − ⟨x⟩²
    sigma = Q2 - Q1_2
    if sigma <= eps(T)
        JobLoggerTools.warn_benji("Variance is too small, using fallback value", jobid)
        sigma = eps(T)  # Avoid division by near-zero
    end

    # Final cumulant calculations
    susceptibility = sigma / V
    skewness = (Q3 - 3.0 * Q1 * Q2 + 2.0 * Q1_3) / (sigma^1.5)
    kurtosis = (Q4 - 4.0 * Q1 * Q3 - 3.0 * Q2_2 + 12.0 * Q1_2 * Q2 - 6.0 * Q1_4) / (sigma^2)
    binder_cumulant = 1.0 - Q4 / (3.0 * Q2_2)

    return susceptibility, skewness, kurtosis, binder_cumulant
end

"""
    compute_traces_raw(
        bin_size::Int,
        obs::Vector{Vector{T}},
        jobid::Union{Nothing, String}=nothing
    ) where T -> Vector{Vector{T}}

Make jackknife resamples for traces ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` in the same
return shape as [`compute_cumulants_raw`](@ref), but without computing cumulants.

The input `obs[i]` is assumed to be a length-5 vector like:
    `[dummy, trM1, trM2, trM3, trM4]`
The first entry is ignored; only indices `2..5` are used.

# Arguments
- `bin_size::Int`: Jackknife bin size.
- `obs::Vector{Vector{T}}`: Per-configuration trace vectors (`length = 5`).
- `jobid`: Optional job id for logging.

# Returns
- `Vector{Vector{T}}` with 4 elements:
  `[trM1_jk, trM2_jk, trM3_jk, trM4_jk]`,
  where each `trM*_jk` is a vector of length `njk` containing the jackknife
  resample values for that trace component.

# Notes
- This is a lightweight companion to [`compute_cumulants_raw`](@ref). No volume ``V``
  or ``\\kappa``-scaling is applied here; it simply jackknifes the provided ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
"""
function compute_traces_raw(
    bin_size::Int,
    obs::Vector{Vector{T}},
    jobid::Union{Nothing, String}=nothing
) where T
    nconf = length(obs)
    nconf == 0 && JobLoggerTools.error_benji("compute_traces_jk: empty obs", jobid)

    # Collect trM1..trM4 across configurations (ignore the first slot)
    trM1 = Vector{T}(undef, nconf)
    trM2 = Vector{T}(undef, nconf)
    trM3 = Vector{T}(undef, nconf)
    trM4 = Vector{T}(undef, nconf)

    @inbounds for i in 1:nconf
        length(obs[i]) >= 5 || JobLoggerTools.error_benji(
            "compute_traces_jk: obs[$i] must have length ≥ 5, got $(length(obs[i]))", jobid)

        trM1[i] = obs[i][2]
        trM2[i] = obs[i][3]
        trM3[i] = obs[i][4]
        trM4[i] = obs[i][5]
    end

    # Jackknife resamples per trace
    trM1_jk = Jackknife.make_jackknife_samples(bin_size, trM1)
    trM2_jk = Jackknife.make_jackknife_samples(bin_size, trM2)
    trM3_jk = Jackknife.make_jackknife_samples(bin_size, trM3)
    trM4_jk = Jackknife.make_jackknife_samples(bin_size, trM4)

    return [trM1_jk, trM2_jk, trM3_jk, trM4_jk]
end

"""
    compute_moments_raw(
        bin_size::Int,
        obs::Vector{Vector{T}},
        jobid::Union{Nothing, String}=nothing
    ) where T -> Vector{Vector{T}}

Compute jackknife resamples of the moments ``Q_{n} \\; (n=1,2,3,4)`` (of single ensemble) from per-configuration
inputs `obs`. For each configuration `i`, [`calc_Q`](@ref) is called to obtain
(`Q1[i]`, `Q2[i]`, `Q3[i]`, `Q4[i]`), and then jackknife resampling is applied
component-wise.

# Arguments
- `bin_size::Int`: Jackknife bin size.
- `obs::Vector{Vector{T}}`: Per-configuration vectors consumed by [`calc_Q`](@ref).
- `jobid`: Optional job id for logging.

# Returns
- `Vector{Vector{T}}` with 4 elements:
  `[Q1_jk, Q2_jk, Q3_jk, Q4_jk]`, where each is a length-`njk` vector of
  jackknife resample values.
"""
function compute_moments_raw(
    bin_size::Int,
    obs::Vector{Vector{T}},
    jobid::Union{Nothing, String}=nothing
) where T
    nconf = length(obs)

    nconf > 0 || JobLoggerTools.error_benji("compute_moments_raw: empty `obs`", jobid)
    bin_size ≥ 1 || JobLoggerTools.error_benji("compute_moments_raw: bin_size must be ≥ 1", jobid)
    bin_size ≤ nconf || JobLoggerTools.error_benji("compute_moments_raw: bin_size > nconf", jobid)

    # Allocate moment vectors for each configuration
    Q1 = Vector{T}(undef, nconf)
    Q2 = Vector{T}(undef, nconf)
    Q3 = Vector{T}(undef, nconf)
    Q4 = Vector{T}(undef, nconf)

    # Compute moments Q1–Q4 for each configuration
    for i in 1:nconf
        Q1[i], Q2[i], Q3[i], Q4[i] = calc_Q(obs[i])
    end

    # Perform jackknife resampling for each moment
    Q1_jk = Jackknife.make_jackknife_samples(bin_size, Q1)
    Q2_jk = Jackknife.make_jackknife_samples(bin_size, Q2)
    Q3_jk = Jackknife.make_jackknife_samples(bin_size, Q3)
    Q4_jk = Jackknife.make_jackknife_samples(bin_size, Q4)

    # === Pack raw jackknife moment vectors (before averaging) ===
    moments_jk = [Q1_jk, Q2_jk, Q3_jk, Q4_jk]

    return moments_jk
end

"""
    compute_cumulants_raw(
        bin_size::Int, 
        V::Int, 
        obs::Vector{Vector{T}}, 
        jobid::Union{Nothing, String}=nothing
    ) where T -> Vector{Vector{T}}

Compute raw jackknife-resampled cumulants from single ensemble.

This function performs jackknife resampling of raw ``Q``-moments (``Q_{n} \\; (n=1,2,3,4)``)  
and computes the corresponding cumulants -- susceptibility, skewness, kurtosis, and Binder cumulant —-  
for each resample. The resulting cumulant vectors are returned directly,  
without performing any averaging or error estimation.

# Arguments
- `bin_size::Int`: Bin size used for jackknife resampling
- `V::Int`: Lattice volume, used for normalization of condensate and susceptibility
- `obs::Vector{Vector{T}}`: List of raw ``Q``-observable vectors per configuration
- `jobid::Union{Nothing, String}`: Optional job ID for logging or debugging

# Returns
- `cumulants_jk::Vector{Vector{T}}`:  
  Jackknife-resampled cumulants packed as a vector of vectors:  
    1. `cond_jk`: Chiral condensate
    2. `susp_jk`: Susceptibility  
    3. `skew_jk`: Skewness  
    4. `kurt_jk`: Kurtosis  
    5. `bind_jk`: Binder cumulant

Each sub-vector has length equal to the number of jackknife bins (`n_jk = nconf / bin_size`),  
and can be passed to [`Deborah.Sarah.Jackknife.jackknife_average_error`](@ref) externally for analysis.
"""
function compute_cumulants_raw(
    bin_size::Int, 
    V::Int, 
    obs::Vector{Vector{T}}, 
    jobid::Union{Nothing, String}=nothing
) where T
    nconf = length(obs)

    # Allocate moment vectors for each configuration
    Q1 = Vector{T}(undef, nconf)
    Q2 = Vector{T}(undef, nconf)
    Q3 = Vector{T}(undef, nconf)
    Q4 = Vector{T}(undef, nconf)

    # Compute moments Q1–Q4 for each configuration
    for i in 1:nconf
        Q1[i], Q2[i], Q3[i], Q4[i] = calc_Q(obs[i])
    end

    # Perform jackknife resampling for each moment
    Q1_jk = Jackknife.make_jackknife_samples(bin_size, Q1)
    Q2_jk = Jackknife.make_jackknife_samples(bin_size, Q2)
    Q3_jk = Jackknife.make_jackknife_samples(bin_size, Q3)
    Q4_jk = Jackknife.make_jackknife_samples(bin_size, Q4)

    njk = length(Q1_jk)
    cond_jk = Vector{T}(undef, njk)
    susp_jk = Vector{T}(undef, njk)
    skew_jk = Vector{T}(undef, njk)
    kurt_jk = Vector{T}(undef, njk)
    bind_jk = Vector{T}(undef, njk)

    # Compute cumulants for each jackknife resample
    for i in 1:njk
        susp_tmp, skew_tmp, kurt_tmp, bind_tmp = calc_cumulants(
            float(V), 
            Q1_jk[i], 
            Q2_jk[i], 
            Q3_jk[i], 
            Q4_jk[i],
            jobid
        )
        cond_jk[i] = Q1_jk[i] / V
        susp_jk[i] = susp_tmp
        skew_jk[i] = skew_tmp
        kurt_jk[i] = kurt_tmp
        bind_jk[i] = bind_tmp
    end

    # === Pack raw jackknife cumulant vectors (before averaging) ===
    cumulants_jk   = [cond_jk, susp_jk, skew_jk, kurt_jk, bind_jk]

    return cumulants_jk
end

"""
    compute_cumulants(
        bin_size::Int, 
        V::Int, 
        obs::Vector{Vector{T}}, 
        w::Vector{T}, 
        jobid::Union{Nothing, String}=nothing
    ) where T -> Vector{Vector{T}}

Compute raw jackknife-resampled cumulants from reweighted trace observables.

This function performs jackknife resampling of reweighted ``Q``-moments (``Q_{n} \\; (n=1,2,3,4)``),  
and computes the corresponding cumulants -- susceptibility, skewness, kurtosis, and Binder cumulant --  
for each resample. The resulting cumulant vectors are returned directly,  
without performing any averaging or error estimation.

# Arguments
- `bin_size::Int`: Bin size used for jackknife resampling
- `V::Int`: Lattice volume, used for normalization of condensate and susceptibility
- `obs::Vector{Vector{T}}`: List of ``Q``-observable vectors per configuration
- `w::Vector{T}`: Reweighting factors per configuration
- `jobid::Union{Nothing, String}`: Optional job ID for logging or debugging

# Returns
- `cumulants_jk::Vector{Vector{T}}`:  
  Jackknife-resampled cumulants packed as a vector of vectors:  
    1. `cond_jk`: Chiral condensate
    2. `susp_jk`: Susceptibility  
    3. `skew_jk`: Skewness  
    4. `kurt_jk`: Kurtosis  
    5. `bind_jk`: Binder cumulant

Each sub-vector has length equal to the number of jackknife bins (`n_jk = nconf / bin_size`),  
and can be passed to [`Deborah.Sarah.Jackknife.jackknife_average_error`](@ref) externally for analysis.
"""
function compute_cumulants(
    bin_size::Int, 
    V::Int, 
    obs::Vector{Vector{T}}, 
    w::Vector{T}, 
    jobid::Union{Nothing, String}=nothing
) where T
    nconf = length(obs)

    # Allocate moment vectors (Q1 to Q4)
    Q1 = Vector{T}(undef, nconf)
    Q2 = Vector{T}(undef, nconf)
    Q3 = Vector{T}(undef, nconf)
    Q4 = Vector{T}(undef, nconf)

    # Compute reweighted Q moments
    for i in 1:nconf
        Q1[i], Q2[i], Q3[i], Q4[i] = calc_Q(obs[i])
        Q1[i] *= w[i]
        Q2[i] *= w[i]
        Q3[i] *= w[i]
        Q4[i] *= w[i]
    end

    # Jackknife resampling for moments and weights
    Q1_jk = Jackknife.make_jackknife_samples(bin_size, Q1)
    Q2_jk = Jackknife.make_jackknife_samples(bin_size, Q2)
    Q3_jk = Jackknife.make_jackknife_samples(bin_size, Q3)
    Q4_jk = Jackknife.make_jackknife_samples(bin_size, Q4)
    w_jk  = Jackknife.make_jackknife_samples(bin_size, w)

    njk = length(Q1_jk)
    cond_jk = Vector{T}(undef, njk)
    susp_jk = Vector{T}(undef, njk)
    skew_jk = Vector{T}(undef, njk)
    kurt_jk = Vector{T}(undef, njk)
    bind_jk = Vector{T}(undef, njk)

    # Normalize Q1 and store cumulants per jackknife bin
    for i in 1:njk
        Q1_jk[i] /= w_jk[i]
        Q2_jk[i] /= w_jk[i]
        Q3_jk[i] /= w_jk[i]
        Q4_jk[i] /= w_jk[i]

        susp_tmp, skew_tmp, kurt_tmp, bind_tmp = calc_cumulants(
            float(V), 
            Q1_jk[i], 
            Q2_jk[i], 
            Q3_jk[i], 
            Q4_jk[i],
            jobid
        )

        cond_jk[i] = Q1_jk[i] / V
        susp_jk[i] = susp_tmp
        skew_jk[i] = skew_tmp
        kurt_jk[i] = kurt_tmp
        bind_jk[i] = bind_tmp
    end

    # === Pack raw jackknife cumulant vectors (before averaging) ===
    cumulants_jk   = [cond_jk, susp_jk, skew_jk, kurt_jk, bind_jk]

    return cumulants_jk
end

end  # module Cumulants