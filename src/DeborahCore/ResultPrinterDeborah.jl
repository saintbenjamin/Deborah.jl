# ============================================================================
# src/DeborahCore/ResultPrinterDeborah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ResultPrinterDeborah

import ..Sarah.JobLoggerTools
import ..Sarah.NameParser
import ..Sarah.SummaryFormatter
import ..TOMLConfigDeborah

"""
    print_summary_results_Deborah(
        data::TOMLConfigDeborah.TraceDataConfig,
        trace_data::Dict{String, Vector{Vector{T}}},
        bootstrap_data::Dict{Symbol, Any},
        bin_size::Int,
        jobid::Union{Nothing, String}=nothing
    ) where T<:Real -> Nothing

Print summary statistics (jackknife and bootstrap) for [`Deborah.DeborahCore`](@ref) results.

# Arguments
- [`data::TOMLConfigDeborah.TraceDataConfig`](@ref Deborah.DeborahCore.TOMLConfigDeborah.TraceDataConfig): Trace data configuration, includes model info.
- `trace_data::Dict{String, Vector{Vector{T}}}`: Raw trace data per observable (`label` → `list of time series vectors`).
- `bootstrap_data::Dict{Symbol, Any}`: Dictionary holding resampled mean values.
- `bin_size::Int`: Binning size for jackknife error.
- `jobid::Union{Nothing, String}`: Optional job identifier for logging.

# Behavior
Prints per-observable average with error for each statistical method.
"""
function print_summary_results_Deborah(
    data::TOMLConfigDeborah.TraceDataConfig,
    trace_data::Dict{String, Vector{Vector{T}}},
    bootstrap_data::Dict{Symbol, Any},
    bin_size::Int, 
    jobid::Union{Nothing, String}=nothing
) where T<:Real

    model_tag = NameParser.model_suffix(data.model, jobid)

    suffixes = [
        ("Y_info",  true,  "Y:JK",    "Y:BS",    "OJK", "OBS"),
        ("YmYP",    false, "BIAS",    "BIAS",    "",    model_tag),
        ("YP_ul",   false, "Y_UL",    "Y_UL",    "",    model_tag),
        ("Y_P1",    false, "Y_P1",    "Y_P1",    "",    model_tag),
        ("Y_P2",    false, "Y_P2",    "Y_P2",    "",    model_tag),
        ("Y_lb",    false, "Y_LB",    "Y_LB",    "",    "OBS")
    ]

    JobLoggerTools.println_benji("=====================", jobid)

    for (sfx, is_jk, jk_tag, bs_tag, jk_grp, bs_grp) in suffixes
        fullkey = "$sfx"

        if is_jk
            _ = SummaryFormatter.print_jackknife_average_error_from_raw(
                trace_data["Y_info"][1], bin_size, "$jk_tag", jk_grp, jobid
            )
        end

        if haskey(bootstrap_data[:mean], fullkey)
            _ = SummaryFormatter.print_bootstrap_average_error(
                bootstrap_data[:mean][fullkey], "$bs_tag", bs_grp, jobid
            )
        end

        if sfx == "Y_info" || sfx == "Y_P2"
            JobLoggerTools.println_benji("---------------------", jobid)
        end
    end

    JobLoggerTools.println_benji("=====================", jobid)
end

end  # module ResultPrinterDeborah