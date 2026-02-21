# =============================================================================
# src/Miriam/ReweightingBundle.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# =============================================================================

module ReweightingBundle

import ..Sarah.JobLoggerTools
import ..Ensemble
import ..Reweighting

"""
    struct ReweightingSolverBundle{T}

Structure to hold multiple [`Deborah.Miriam.Reweighting.ReweightingSolver`](@ref) instances along with metadata
for ensemble tagging and configuration tracking.

# Fields
- [`solvers::Vector{Reweighting.ReweightingSolver{T}}`](@ref Deborah.Miriam.Reweighting.ReweightingSolver): Vector of [`Reweighting.ReweightingSolver{T}`](@ref Deborah.Miriam.Reweighting.ReweightingSolver) for each ensemble array
- `tags::Vector{String}  `: Identifier tags for each ensemble (e.g., `"Y_tr"`, `"Y_bc"`, `"Y_ul"`)
- `conf_nums::Vector{Vector{Int}}  `: Flattened configuration numbers per solver, useful for traceability
"""
struct ReweightingSolverBundle{T}
    solvers::Vector{Reweighting.ReweightingSolver{T}}    # Collection of reweighting solvers
    tags::Vector{String}                                 # Tags identifying each ensemble array
    conf_nums::Vector{Vector{Int}}                       # Flattened config numbers per solver
end

"""
    ReweightingSolverBundle(
        bundle::Ensemble.EnsembleArrayBundle{T}, 
        maxiter::Int, 
        eps::T
    ) where T -> ReweightingSolverBundle{T}

Construct a [`ReweightingSolverBundle`](@ref) by initializing a solver for each ensemble array.

# Arguments
- [`bundle::Ensemble.EnsembleArrayBundle{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Structure containing ensemble arrays and tags
- `maxiter::Int`: Maximum number of iterations for solver convergence
- `eps::T`: Convergence tolerance

# Returns
- A [`ReweightingSolverBundle`](@ref) containing initialized solvers and configuration indices
"""
function ReweightingSolverBundle(
    bundle::Ensemble.EnsembleArrayBundle{T}, 
    maxiter::Int, 
    eps::T
) where T
    # Initialize a solver for each ensemble array
    # solvers = Reweighting.ReweightingSolver.(bundle.arrays, Ref(maxiter), Ref(eps))

    # Flatten configuration numbers for each ensemble array
    conf_lists = [reduce(vcat, [e.conf_nums for e in ea.data]) for ea in bundle.arrays]

    return ReweightingSolverBundle(
        Reweighting.ReweightingSolver.(
            bundle.arrays, 
            Ref(maxiter), 
            Ref(eps)
        ), 
        bundle.tags, 
        conf_lists
    )
end

"""
    calc_f_all!(
        rw_bundle::ReweightingSolverBundle,
        info_file::String,
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Compute free energy differences ``f`` for all solvers in the bundle.

# Arguments
- [`rw_bundle::ReweightingSolverBundle`](@ref ReweightingSolverBundle): A bundle of reweighting solvers (each with a different trace source).
- `info_file::String`: Path to the [`TOML`](https://toml.io/en/) file where solver summary info is saved.
- `jobid::Union{Nothing, String}`: Optional job ID for logging context.

# Behavior
- Iterates over each solver in the bundle and calls [`Deborah.Miriam.Reweighting.calc_f!`](@ref) with its corresponding tag.
- Logs progress and timing info for each tag.

# Returns
- `Nothing`; results are written to [`TOML`](https://toml.io/en/) and stored in-place in each solver.
"""
function calc_f_all!(
    rw_bundle::ReweightingSolverBundle, 
    info_file::String,
    jobid::Union{Nothing, String}=nothing
)
    for (i, rw) in enumerate(rw_bundle.solvers)
        tag = rw_bundle.tags[i]
        JobLoggerTools.log_stage_sub1_benji("tag: $(tag)", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            Reweighting.calc_f!(
                rw, 
                info_file, 
                tag, 
                jobid
            )
        end
    end
end

"""
    calc_w_all!(
        rw_bundle::ReweightingSolverBundle,
        paramT::Ensemble.Params
    ) -> Nothing

Compute reweighting weights ``w_i(\\kappa_T)`` for all solvers in the bundle using the same target parameter set.

# Arguments
- [`rw_bundle::ReweightingSolverBundle`](@ref Deborah.Miriam.ReweightingBundle.ReweightingSolverBundle): A bundle of reweighting solvers containing ensemble data.
- [`paramT::Ensemble.Params`](@ref Deborah.Miriam.Ensemble.Params): Target simulation parameters (including ``\\kappa_T``) to compute weights against.

# Behavior
- Calls [`Deborah.Miriam.Reweighting.calc_w!`](@ref) on each solver in the bundle using the provided `paramT`.

# Returns
- `Nothing`; each solver's internal weight vector is updated in-place.
"""
function calc_w_all!(
    rw_bundle::ReweightingSolverBundle, 
    paramT::Ensemble.Params
)
    for rw in rw_bundle.solvers
        Reweighting.calc_w!(
            rw, 
            paramT
        )
    end
end

end  # module ReweightingBundle