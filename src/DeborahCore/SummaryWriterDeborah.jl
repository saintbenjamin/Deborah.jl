# ============================================================================
# src/DeborahCore/SummaryWriterDeborah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SummaryWriterDeborah

import ..Printf: @printf

import ..Sarah.NameParser
import ..TOMLConfigDeborah

"""
    write_summary_file_Deborah(
        data::TOMLConfigDeborah.TraceDataConfig,
        summary_jackknife::Dict{String, Tuple{T, T}},
        summary_bootstrap::Dict{String, Tuple{T, T}},
        overall_name::String,
        analysis_dir::String,
        jobid::Union{Nothing, String}=nothing
    ) where T<:Real -> Nothing

Write summary statistics (jackknife and bootstrap) to a file for [`Deborah.DeborahCore`](@ref) results.

# Arguments
- [`data::TOMLConfigDeborah.TraceDataConfig`](@ref Deborah.DeborahCore.TOMLConfigDeborah.TraceDataConfig): Trace data configuration containing model name.
- `summary_jackknife::Dict{String, Tuple{T, T}}`: Dictionary of ``(\\mu, \\sigma)`` from jackknife estimates.
- `summary_bootstrap::Dict{String, Tuple{T, T}}`: Dictionary of ``(\\mu, \\sigma)`` from bootstrap estimates.
- `overall_name::String`: Identifier for output file naming.
- `analysis_dir::String`: Target directory to save the summary file.
- `jobid::Union{Nothing, String}` : Optional job ID for logging.

# Output
- Writes formatted summary lines to `summary_Deborah_<overall_name>.dat` under `analysis_dir`.
"""
function write_summary_file_Deborah(
    data::TOMLConfigDeborah.TraceDataConfig,
    summary_jackknife::Dict{String, Tuple{T, T}},
    summary_bootstrap::Dict{String, Tuple{T, T}},
    overall_name::String,
    analysis_dir::String,
    jobid::Union{Nothing, String}=nothing
) where T<:Real

    model_tag = NameParser.model_suffix(data.model, jobid)

    labels_order = [
        ("Y_info", "Y:JK", "OJK",                        summary_jackknife),
        ("Y_info", "Y:BS", "OBS",                        summary_bootstrap),
        ("Y_lb",   "Y_LB", "OBS",                        summary_bootstrap),
        ("YmYP",   "BIAS", model_tag,                    summary_bootstrap),
        ("YP_ul",  "Y_UL", model_tag,                    summary_bootstrap),
        ("Y_P1",   "Y_P1", model_tag,                    summary_bootstrap),
        ("Y_P2",   "Y_P2", model_tag,                    summary_bootstrap),
    ]

    mkpath(analysis_dir)
    fname = analysis_dir*"/"*"summary_Deborah_$overall_name.dat"
    open(fname, "w") do io
        for (suf, label, tag, dict) in labels_order
            key = "$(suf)"
            if haskey(dict, key)
                @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                    "#$(label)",
                    dict[key][1], dict[key][2], tag
                )
            else
                @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                    "#$(label)", 0.0, 0.0, tag
                )
            end
        end
    end
end

end  # module SummaryWriterDeborah