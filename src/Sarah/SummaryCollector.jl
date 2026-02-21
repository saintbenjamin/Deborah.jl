# ============================================================================
# src/Sarah/SummaryCollector.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SummaryCollector

import ..NameParser
import ..Jackknife
import ..Bootstrap

"""
    collect_summaries_Deborah(
        trace_data::Dict{String, Vector{Vector{T}}},
        bootstrap_data::Dict{Symbol, Any},
        bin_size::Int
    ) -> Tuple{Dict{String, Tuple{T, T}}, Dict{String, Tuple{T, T}}} where T<:Real

Collect jackknife and bootstrap summary statistics for [`Deborah.DeborahCore`](@ref) model output.

# Arguments
- `trace_data::Dict{String, Vector{Vector{T}}}`: Raw trace data, where `"Y_info"` key holds a vector of vectors.
- `bootstrap_data::Dict{Symbol, Any}`: Contains precomputed bootstrap means under `:mean` field.
- `bin_size::Int`: Bin size to use for jackknife resampling.

# Returns
- `(summary_jackknife, summary_bootstrap)`:
  - `Dict{String, Tuple{T, T}}`: Maps observable names to `(mean, error)` from jackknife.
  - `Dict{String, Tuple{T, T}}`: Maps observable names to `(mean, error)` from bootstrap.

# Notes
- Jackknife is computed only for `"Y_info"` (first vector in the list).
- Bootstrap summaries are computed for all observables in the suffix list.
- Uses [`Jackknife.jackknife_average_error_from_raw`](@ref) and [`Bootstrap.bootstrap_average_error`](@ref) internally.
"""
function collect_summaries_Deborah(
    trace_data::Dict{String, Vector{Vector{T}}}, 
    bootstrap_data::Dict{Symbol, Any}, 
    bin_size::Int
)::Tuple{
    Dict{String, Tuple{T, T}}, 
    Dict{String, Tuple{T, T}}
} where T<:Real

    summary_jackknife = Dict{String, Tuple{T, T}}()
    summary_bootstrap = Dict{String, Tuple{T, T}}()

    suffixes = ["Y_info", "YmYP", "YP_ul", "Y_P1", "Y_P2", "Y_lb"]

    for suf in suffixes
        key = "$(suf)"

        if haskey(trace_data, "Y_info")
            src = trace_data["Y_info"][1]
            summary_jackknife[key] = Jackknife.jackknife_average_error_from_raw(src, bin_size)
        end

        summary_bootstrap[key] = Bootstrap.bootstrap_average_error(bootstrap_data[:mean][key])
    end

    return summary_jackknife, summary_bootstrap
end

"""
    collect_summaries_Esther(
        trace_data::Dict{String, Vector{Vector{T}}},
        Q_moment::Dict{String, Vector{T}},
        jackknife_data::Dict{Symbol, Any},
        bootstrap_data::Dict{Symbol, Any},
        bin_size::Int
    ) -> Tuple{Dict{String, Tuple{T, T}}, Dict{String, Tuple{T, T}}} where T<:Real

Collect jackknife and bootstrap summary results for the [`Deborah.Esther`](@ref) model into two dictionaries.

# Arguments
- `trace_data::Dict{String, Vector{Vector{T}}}`: Raw trace data grouped by `trM1`-`trM4`.
- `Q_moment::Dict{String, Vector{T}}`: Moment-transformed trace data grouped by `Q1`-`Q4`.
- `jackknife_data::Dict{Symbol, Any}`: Precomputed jackknife means under `:mean`.
- `bootstrap_data::Dict{Symbol, Any}`: Precomputed bootstrap means under `:mean`.
- `bin_size::Int`: Bin size for jackknife error estimation.

# Returns
- `(summary_jackknife, summary_bootstrap)`:
  - `Dict{String, Tuple{T, T}}`: Maps each observable (e.g., `"trM1:Y_info"`) to a tuple of `(mean, error)` for jackknife.
  - `Dict{String, Tuple{T, T}}`: Same structure for bootstrap results.

# Notes
- Uses [`Jackknife.jackknife_average_error_from_raw`](@ref), [`Jackknife.jackknife_average_error`](@ref), and [`Bootstrap.bootstrap_average_error`](@ref) internally to extract `(mean, error)` tuples.
- Jackknife is only computed for `Y_info`; bootstrap is computed for all suffixes.
"""
function collect_summaries_Esther(
    trace_data::Dict{String, Vector{Vector{T}}},
    Q_moment::Dict{String, Vector{T}},
    jackknife_data::Dict{Symbol, Any},
    bootstrap_data::Dict{Symbol, Any},
    bin_size::Int
)::Tuple{
    Dict{String, Tuple{T, T}},
    Dict{String, Tuple{T, T}}
} where T<:Real

    summary_jackknife = Dict{String, Tuple{T, T}}()
    summary_bootstrap = Dict{String, Tuple{T, T}}()

    models_tr = ["trM1", "trM2", "trM3", "trM4"]
    models_q  = ["Q1", "Q2", "Q3", "Q4"]
    obs_names = ["cond", "susp", "skew", "kurt"]
    suffixes  = ["Y_info", "YmYP", "YP_ul", "Y_P1", "Y_P2", "Y_lb"]

    groups = [
        (models_tr, trace_data["Y_info"],  true, Jackknife.jackknife_average_error_from_raw),
        (models_q,  Q_moment,              true, Jackknife.jackknife_average_error_from_raw),
        (obs_names, jackknife_data[:mean], true, Jackknife.jackknife_average_error)
    ]

    for (group, source, jk_src, jk_func) in groups
        for (i, name) in enumerate(group)
            for suf in suffixes
                key = "$(name):$(suf)"

                if jk_src && suf == "Y_info"
                    src = group === obs_names ? source["$(name):Y_info"] :
                          group === models_q  ? source["$(name):Y_info"] :
                                                source[i]

                    if jk_func === Jackknife.jackknife_average_error
                        summary_jackknife[key] = jk_func(src)
                    else
                        summary_jackknife[key] = jk_func(src, bin_size)
                    end
                end

                summary_bootstrap[key] = Bootstrap.bootstrap_average_error(bootstrap_data[:mean][key])
            end
        end
    end

    return summary_jackknife, summary_bootstrap
end

end  # module SummaryCollector