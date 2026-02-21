# ============================================================================
# src/Esther/TraceRescaler.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module TraceRescaler

"""
    rescale_trace(
        X_in::AbstractArray,
        Kappa::Float64,
        LatVol::Int,
        power::Int
    ) -> Vector{Float64}

Rescale the input trace data by a physical volume and hopping parameter factor.  

This rescaling is performed in the context of research by **Benjamin J. Choi**,  
who uses measurement data provided by **Hiroshi Ohno et al. ([PoS(LATTICE2018)174](https://inspirehep.net/literature/1706759))**.  
These data were generated with the **BQCD HMC program ([PoS(LATTICE2010)040](https://inspirehep.net/literature/875095) and [EPJ Web Conf. 175, 14011 (2018)](https://inspirehep.net/literature/1635527))**.  

Here, as noted in **[PoS(LATTICE2010)040, Sec. 3 (p. 3)](https://inspirehep.net/literature/875095)**,  
the original trace data is normalized by a factor of ``12V``,  
where ``V`` is the lattice volume.  
To make the normalization consistent with later analyses  
(e.g. **[Phys. Rev. D94, 114507 (2016), Eq. (4)](https://inspirehep.net/literature/1459066)**),  
we rescale as follows:

```math
\\left[ \\text{Tr} \\, M^{-p} \\right]_{\\text{rescaled}} = 12 \\, V \\, \\left( 2 \\, \\kappa \\right)^{p} \\, \\left[ \\text{Tr} \\, M^{-p} \\right]_{\\text{in}}
```

where  
- ``V`` is the lattice volume,  
- ``\\kappa`` is the hopping parameter,  
- ``p`` is the power.

# Arguments
- `X_in::AbstractArray` : Raw input array (typically bootstrap samples).  
- `Kappa::Float64`      : Hopping parameter.  
- `LatVol::Int`         : Lattice volume (e.g. ``N_S^3 \\times N_T``).  
- `power::Int`          : Power to which the hopping parameter is raised.  

# Returns
- `Vector{Float64}` : Rescaled trace data.
"""
function rescale_trace(
    X_in::AbstractArray,
    Kappa::Float64,
    LatVol::Int,
    power::Int
)::Vector{Float64}
    factor = 12 * Float64(LatVol) * (2 * Kappa)^power
    N_cnf = length(X_in)
    X_out = zeros(Float64, N_cnf)
    for idx in 1:N_cnf
        X_out[idx] = factor * X_in[idx]
    end
    return X_out
end

"""
    rescale_all_traces(
        trace_data::Dict{String, Vector{Vector{Float64}}},
        KappaF64::Float64,
        LatVol::Int
    ) -> Dict{String, Vector{Any}}

Rescale all input trace data by appropriate powers of the hopping parameter.

# Arguments
- `trace_data::Dict{String, Vector{Vector{Float64}}}`: Dictionary containing raw trace data for each label.
- `KappaF64::Float64`: Hopping parameter ``\\kappa`` as a Float64.
- `LatVol::Int`: Lattice volume (e.g., ``N_S^3 \\times N_T``).

# Returns
- `Dict{String, Vector{Any}}`: Dictionary with rescaled trace data.
"""
function rescale_all_traces(
    trace_data::Dict{String, Vector{Vector{Float64}}},
    KappaF64::Float64,
    LatVol::Int
)::Dict{String, Vector{Any}}
    trace_rscl = Dict{String, Vector{Any}}()
    for (label, vecs) in trace_data
        trace_rscl[label] = Vector{Any}(undef, length(vecs))
        for i in eachindex(vecs)
            trace_rscl[label][i] = rescale_trace(vecs[i], KappaF64, LatVol, i)
        end
    end
    return trace_rscl
end

end  # module TraceRescaler