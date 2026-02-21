# ============================================================================
# src/Esther/QMomentCalculator.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module QMomentCalculator

import ..Sarah.JobLoggerTools
import ..SingleQMoment

"""
    compute_Q_moments(
        trace_rscl::Dict{String, Vector{Any}}, 
        Nf::Int, 
        jobid::Union{Nothing, String}=nothing
    ) -> Dict{String, Vector{Float64}}

Compute the ``Q``-moments (``Q_n \\; (n=1,2,3,4)``) for each trace label in the rescaled trace data.

This function iterates over the keys of the input trace dictionary and calculates the 1st through 4th order ``Q``-moments for each observable, storing them in a new dictionary with keys formatted as `"Q1:label"`, `"Q2:label"`, etc.

# Arguments
- `trace_rscl::Dict{String, Vector{Any}}`: Dictionary where each key corresponds to an observable label (e.g., `"Y_tr"`), and each value is a vector of rescaled trace components. Expected to contain at least four elements per label: `[trM1, trM2, trM3, trM4]`.
- `Nf::Int`: Number of fermion flavors (used in ``Q``-moment calculations).
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `Dict{String, Vector{Float64}}`: A dictionary mapping Q-moment labels like `"Q1:Y_tr"` to their computed values as vectors.

# Example
```julia
Q = compute_Q_moments(
    trace_rscl, 
    4
)
q2_tr = Q["Q2:Y_tr"]
```

# Notes
- Each [`Deborah.Esther.SingleQMoment.calc_Q1`](@ref), [`Deborah.Esther.SingleQMoment.calc_Q2`](@ref), [`Deborah.Esther.SingleQMoment.calc_Q3`](@ref), and [`Deborah.Esther.SingleQMoment.calc_Q4`](@ref) must be defined separately and accept the required number of trace inputs and ``N_\\text{f}``.
- Assumes all `trace_rscl[label]`` vectors contain at least 4 elements.

"""
function compute_Q_moments(
    trace_rscl::Dict{String, Vector{Any}}, 
    Nf::Int, 
    jobid::Union{Nothing, String}=nothing
)
    labels = keys(trace_rscl)
    Q_moment = Dict{String, Vector{Float64}}()

    for label in labels
        tr = trace_rscl[label]
        if length(tr) < 4 || any(x -> !(x isa AbstractVector), tr)
            JobLoggerTools.warn_benji("trace_rscl[$label] has insufficient or invalid components → skipping", jobid)
            Q_moment["Q1:$label"] = Float64[]
            Q_moment["Q2:$label"] = Float64[]
            Q_moment["Q3:$label"] = Float64[]
            Q_moment["Q4:$label"] = Float64[]
            continue
        end

        Q_moment["Q1:$label"] = SingleQMoment.calc_Q1(tr[1], Nf)
        Q_moment["Q2:$label"] = SingleQMoment.calc_Q2(tr[1], tr[2], Nf)
        Q_moment["Q3:$label"] = SingleQMoment.calc_Q3(tr[1], tr[2], tr[3], Nf)
        Q_moment["Q4:$label"] = SingleQMoment.calc_Q4(tr[1], tr[2], tr[3], tr[4], Nf)
    end

    return Q_moment
end

end  # module QMomentCalculator