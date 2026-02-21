# ============================================================================
# src/MiriamDocument/MiriamDocumentRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MiriamDocumentRunner

import TOML
import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..RebekahMiriam.SummaryLoaderRebekahMiriam
import ..RebekahMiriam.JLD2SaverRebekahMiriam

"""
    run_MiriamDocument(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Main driver function to generate all [`Deborah.Miriam`](@ref) [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) for a multi-ensemble reweighting calculation.

This function parses a [`TOML`](https://toml.io/en/) configuration file, loads summary statistics, writes a [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshot.

# Arguments
- `toml_path::String` : Path to the [`TOML`](https://toml.io/en/) configuration file, which must include:
    - `data.labels` and `data.trains` for `LBP` and `TRP` values
    - `data.location` and `data.multi_ensemble` for path resolution
    - `data.TrM1_X`, `TrM1_Y`, `TrM1_model` (and similarly for `TrM2`--`TrM4`)
    - `data.analysis_header` to form the ensemble name
    - `data.use_abbreviation` to toggle abbreviation usage
    - `abbreviation` dictionary mapping names to codes
- `jobid::Union{Nothing, String}`: Optional job ID for contextual logging.

# Workflow Overview
1. Load raw summary files and merge into `new_dict`
2. Write [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshot.
"""
function run_MiriamDocument(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)

    cfg = TOML.parsefile(toml_path)

    labels           = cfg["data"]["labels"]
    trains           = cfg["data"]["trains"]

    location         = cfg["data"]["location"]
    multi_ensemble   = cfg["data"]["multi_ensemble"]

    ensembles        = cfg["data"]["ensembles"]
    
    TrM1_X           = cfg["data"]["TrM1_X"]
    TrM1_Y           = cfg["data"]["TrM1_Y"]
    TrM1_model       = cfg["data"]["TrM1_model"]

    TrM2_X           = cfg["data"]["TrM2_X"]
    TrM2_Y           = cfg["data"]["TrM2_Y"]
    TrM2_model       = cfg["data"]["TrM2_model"]

    TrM3_X           = cfg["data"]["TrM3_X"]
    TrM3_Y           = cfg["data"]["TrM3_Y"]
    TrM3_model       = cfg["data"]["TrM3_model"]

    TrM4_X           = cfg["data"]["TrM4_X"]
    TrM4_Y           = cfg["data"]["TrM4_Y"]
    TrM4_model       = cfg["data"]["TrM4_model"]

    analysis_header  = cfg["data"]["analysis_header"]
    use_abbreviation = cfg["data"]["use_abbreviation"]
    raw_abbrev      =  cfg["abbreviation"]

    abbreviation = StringTranscoder.parse_string_dict(raw_abbrev)
    
    TrM1_code = StringTranscoder.input_encoder_abbrev_dict(TrM1_X, TrM1_Y, abbreviation)
    TrM2_code = StringTranscoder.input_encoder_abbrev_dict(TrM2_X, TrM2_Y, abbreviation)
    TrM3_code = StringTranscoder.input_encoder_abbrev_dict(TrM3_X, TrM3_Y, abbreviation)
    TrM4_code = StringTranscoder.input_encoder_abbrev_dict(TrM4_X, TrM4_Y, abbreviation)

    TrM1_X_Y  = NameParser.make_X_Y(TrM1_X, TrM1_Y)
    TrM2_X_Y  = NameParser.make_X_Y(TrM2_X, TrM2_Y)
    TrM3_X_Y  = NameParser.make_X_Y(TrM3_X, TrM3_Y)
    TrM4_X_Y  = NameParser.make_X_Y(TrM4_X, TrM4_Y)

    TrM1_suffix = NameParser.model_suffix(TrM1_model, jobid)
    TrM2_suffix = NameParser.model_suffix(TrM2_model, jobid)
    TrM3_suffix = NameParser.model_suffix(TrM3_model, jobid)
    TrM4_suffix = NameParser.model_suffix(TrM4_model, jobid)

    if use_abbreviation
        learning = "$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)"
    else
        learning = "$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)"
    end

    analysis_ensemble="$(analysis_header)_$(multi_ensemble)"
    overall_name = "$(multi_ensemble)_$(learning)"
    cumulant_name = "$(analysis_header)_$(overall_name)"

    first_block_keys  = [:kappa,   :cond, :susp, :skew, :kurt, :bind]
    second_block_keys = [:kappa_t, :cond, :susp, :skew, :kurt, :bind]  

    tags =        [:Y_BS, :Y_JK, :Y_P1, :Y_P2, 
                   :RWBS, :RWJK, :RWP1, :RWP2]   

    RWtags = [:RWBS, :RWJK, :RWP1, :RWP2]
    Y_tags = [:Y_BS, :Y_JK, :Y_P1, :Y_P2]

    interpolation_criterion = ["susp", "skew", "kurt"]

    # ================================================================================

    new_dict = 
    SummaryLoaderRebekahMiriam.load_miriam_summary(
        location, analysis_ensemble, 
        cumulant_name, overall_name, 
        labels, trains, 
        interpolation_criterion, RWtags, 
        second_block_keys
    )

    T_tags = [:T_BS, :T_JK, :T_P1, :T_P2]
    T_keys = [:kappa, :trM1, :trM2, :trM3, :trM4]  

    new_dict_trace_meas, kappa_list = 
    SummaryLoaderRebekahMiriam.load_miriam_summary_for_measurement(
        location, analysis_ensemble, 
        cumulant_name, overall_name, 
        labels, trains, 
        ensembles, multi_ensemble,
        T_tags, T_keys
    )

    Q_tags = [:Q_BS, :Q_JK, :Q_P1, :Q_P2]
    Q_keys = [:kappa, :Q1, :Q2, :Q3, :Q4]

    new_dict_moment_meas, _ = 
    SummaryLoaderRebekahMiriam.load_miriam_summary_for_measurement(
        location, analysis_ensemble, 
        cumulant_name, overall_name, 
        labels, trains, 
        ensembles, multi_ensemble,
        Q_tags, Q_keys
    )

    new_dict_cumulant_meas, _ = 
    SummaryLoaderRebekahMiriam.load_miriam_summary_for_measurement(
        location, analysis_ensemble, 
        cumulant_name, overall_name, 
        labels, trains, 
        ensembles, multi_ensemble,
        Y_tags, first_block_keys
    )

    # ================================================================================

    path_template = (label, train, tag) -> "$(location)/$(analysis_ensemble)/$(cumulant_name)/$(overall_name)_LBP_$(label)_TRP_$(train)/$(String(tag))_$(overall_name)_LBP_$(label)_TRP_$(train).dat"

    rw_data = SummaryLoaderRebekahMiriam.load_all_rw_data(
        labels, trains, 
        tags, path_template
    )

    # ================================================================================

    infos_path_template = (label::String, train::String) ->
        "$(location)/$(analysis_ensemble)/$(cumulant_name)/" *
        "$(overall_name)_LBP_$(label)_TRP_$(train)/" *
        "infos_Miriam_$(overall_name)_LBP_$(label)_TRP_$(train).toml"

    nlsolve_status =
        SummaryLoaderRebekahMiriam.load_all_nlsolve_status(
            labels, trains, infos_path_template;
            solver_prefix = "nlsolve_f_solver_",
            on_missing    = :warn
        )

    # ================================================================================

    jld2_name = joinpath(location, analysis_ensemble, cumulant_name, "results_$(overall_name).jld2")

    JLD2SaverRebekahMiriam.save_miriam_results(
        jld2_name, 
        new_dict, 
        new_dict_trace_meas, 
        new_dict_moment_meas, 
        new_dict_cumulant_meas, 
        kappa_list, 
        rw_data, 
        nlsolve_status, 
        labels, trains
    )

    HERE = pwd()
    jld2_HERE = joinpath(HERE, "results_$(overall_name).jld2")
    cp(jld2_name, jld2_HERE; force=true)

end

end  # module MiriamDocumentRunner