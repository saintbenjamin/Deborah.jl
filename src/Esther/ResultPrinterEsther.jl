# ============================================================================
# src/Esther/ResultPrinterEsther.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ResultPrinterEsther

import ..Sarah.JobLoggerTools
import ..Sarah.NameParser
import ..Sarah.SummaryFormatter
import ..TOMLConfigEsther

"""
    print_summary_results_Esther(
        data::TOMLConfigEsther.TraceDataConfig,
        trace_data::Dict{String, Vector{Vector{T}}},
        jackknife_data::Dict{Symbol, Any},
        bootstrap_data::Dict{Symbol, Any},
        bin_size::Int,
        jobid::Union{Nothing, String}=nothing
    ) where T<:Real -> nothing

Print summary statistics (jackknife and bootstrap) for Esther results with per-observable detail.

# Arguments
- [`data::TOMLConfigEsther.TraceDataConfig`](@ref Deborah.Esther.TOMLConfigEsther.TraceDataConfig): Trace data config, includes models for ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
- `trace_data::Dict{String, Vector{Vector{T}}}`: Raw trace data (`label` → `list of time series vectors`).
- `jackknife_data::Dict{Symbol, Any}`: Dictionary of jackknife mean estimates.
- `bootstrap_data::Dict{Symbol, Any}`: Dictionary of bootstrap mean estimates.
- `bin_size::Int`: Binning size for jackknife error.
- `jobid::Union{Nothing, String}`: Optional job identifier for logging.

# Behavior
Prints average and error for each observable and trace model,
distinguishing model types and grouped outputs.
"""
function print_summary_results_Esther(
    data::TOMLConfigEsther.TraceDataConfig,
    trace_data::Dict{String, Vector{Vector{T}}},
    jackknife_data::Dict{Symbol, Any},
    bootstrap_data::Dict{Symbol, Any},
    bin_size::Int, 
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

    keys_main = ["trM1", "trM2", "trM3", "trM4", "cond", "susp", "skew", "kurt"]
    trace_tag = [TrM1_model_tag, TrM2_model_tag, TrM3_model_tag, TrM4_model_tag]
    suffixes = [
        ("Y_info",  true,  "Y:JK",    "Y:BS",    "OJK", "OBS"),
        ("YmYP",    false, "BIAS",    "BIAS",    "",    model_tag),
        ("YP_ul",   false, "Y_UL",    "Y_UL",    "",    model_tag),
        ("Y_P1",    false, "Y_P1",    "Y_P1",    "",    model_tag),
        ("Y_P2",    false, "Y_P2",    "Y_P2",    "",    model_tag),
        ("Y_lb",    false, "Y_LB",    "Y_LB",    "",    "OBS")
    ]

    for (i, key) in enumerate(keys_main)
        JobLoggerTools.println_benji("=====================", jobid)

        for (sfx, is_jk, jk_tag, bs_tag, jk_grp, bs_grp) in suffixes
            fullkey = "$key:$sfx"
            use_bs_grp = (startswith(key, "trM") && !(bs_grp == "OJK" || bs_grp == "OBS")) ? trace_tag[i] : bs_grp

            if is_jk
                if startswith(key, "trM")
                    tr_idx = parse(Int, last(key))
                    _ = SummaryFormatter.print_jackknife_average_error_from_raw(
                        trace_data["Y_info"][tr_idx], bin_size, "$key:$jk_tag", jk_grp, jobid
                    )
                else
                    _ = SummaryFormatter.print_jackknife_average_error(
                        jackknife_data[:mean][fullkey], "$key:$jk_tag", jk_grp, jobid
                    )
                end
            end

            if haskey(bootstrap_data[:mean], fullkey)
                _ = SummaryFormatter.print_bootstrap_average_error(
                    bootstrap_data[:mean][fullkey], "$key:$bs_tag", use_bs_grp, jobid
                )
            end

            if sfx == "Y_info" || sfx == "Y_P2"
                JobLoggerTools.println_benji("---------------------", jobid)
            end
        end
    end

    JobLoggerTools.println_benji("=====================", jobid)
end

end  # module ResultPrinter
