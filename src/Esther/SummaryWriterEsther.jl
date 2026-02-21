# ============================================================================
# src/Esther/SummaryWriterEsther.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SummaryWriterEsther

import Printf: @printf
import ..Sarah.NameParser
import ..TOMLConfigEsther

"""
    write_summary_file_Esther(
        data::TOMLConfigEsther.TraceDataConfig,
        summary_jackknife::Dict{String, Tuple{T, T}},
        summary_bootstrap::Dict{String, Tuple{T, T}},
        overall_name::String,
        analysis_dir::String,
        jobid::Union{Nothing, String}=nothing
    ) where T<:Real -> Nothing

Write summary statistics (jackknife and bootstrap) to a file for Esther results.

# Arguments
- `data`: Trace data configuration including ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` model names.
- `summary_jackknife`: Dictionary of ``(\\mu, \\sigma)`` from jackknife estimates.
- `summary_bootstrap`: Dictionary of ``(\\mu, \\sigma)`` from bootstrap estimates.
- `overall_name`: Identifier for output file naming.
- `analysis_dir`: Target directory to save the summary file.
- `jobid::Union{Nothing, String}` : Optional job ID for logging.

# Output
- Writes formatted summary lines to `summary_Esther_<overall_name>.dat` under `analysis_dir`.
- Includes model-wise results ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``, ``Q_n \\; (n=1,2,3,4)``, and condensed observables (`cond`, `susp`, `skew`, `kurt`).
"""
function write_summary_file_Esther(
    data::TOMLConfigEsther.TraceDataConfig,
    summary_jackknife::Dict{String, Tuple{T, T}},
    summary_bootstrap::Dict{String, Tuple{T, T}},
    overall_name::String,
    analysis_dir::String,
    jobid::Union{Nothing, String}=nothing
) where T<:Real

    TrM1_model_tag = NameParser.model_suffix(data.TrM1_model, jobid)
    TrM2_model_tag = NameParser.model_suffix(data.TrM2_model, jobid)
    TrM3_model_tag = NameParser.model_suffix(data.TrM3_model, jobid)
    TrM4_model_tag = NameParser.model_suffix(data.TrM4_model, jobid)

    if TrM1_model_tag == TrM2_model_tag == TrM3_model_tag == TrM4_model_tag
        model_tag = TrM1_model_tag
    else
        model_tag = "MIX"
    end

    models    = ["trM1", "trM2", "trM3", "trM4"]
    trace_tag = [TrM1_model_tag, TrM2_model_tag, TrM3_model_tag, TrM4_model_tag]
    qmodels   = ["Q1", "Q2", "Q3", "Q4"]
    obs_names = ["cond", "susp", "skew", "kurt"]

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
    fname = analysis_dir*"/"*"summary_Esther_$overall_name.dat"
    open(fname, "w") do io
        for (i, model) in enumerate(models)
            for (suf, label, tag, dict) in labels_order
                key = "$(model):$(suf)"
                tag_to_use = (tag == "OJK" || tag == "OBS") ? tag : trace_tag[i]
                if haskey(dict, key)
                    val1 = isnan(dict[key][1]) ? 0.0 : dict[key][1]
                    val2 = isnan(dict[key][2]) ? 0.0 : dict[key][2]
                    @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                        "#$(model):$(label)",
                        val1, val2, tag_to_use
                    )
                else
                    @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                        "#$(model):$(label)",
                        0.0, 0.0, tag_to_use
                    )
                end
            end
        end
        for q in qmodels
            for (suf, label, tag, dict) in labels_order
                key = "$(q):$(suf)"
                if haskey(dict, key)
                    val1 = isnan(dict[key][1]) ? 0.0 : dict[key][1]
                    val2 = isnan(dict[key][2]) ? 0.0 : dict[key][2]
                    @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                        "#$(q):$(label)",
                        val1, val2, tag
                    )
                else
                    @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                        "#$(q):$(label)",
                        0.0, 0.0, tag
                    )
                end
            end
        end
        for name in obs_names
            for (suf, label, tag, dict) in labels_order
                key = "$(name):$(suf)"
                if haskey(dict, key)
                    val1 = isnan(dict[key][1]) ? 0.0 : dict[key][1]
                    val2 = isnan(dict[key][2]) ? 0.0 : dict[key][2]
                    @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                        "#$(name):$(label)",
                        val1, val2, tag
                    )
                else
                    @printf(io, "%s\t%.14e\t%.14e\t%s\n",
                        "#$(name):$(label)",
                        0.0, 0.0, tag
                    )
                end
            end
        end
    end
end

end  # module SummaryWriterEsther