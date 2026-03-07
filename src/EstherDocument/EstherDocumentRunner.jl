# ============================================================================
# src/EstherDocumentRunner/EstherDocumentRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module EstherDocumentRunner

import ..TOML

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Rebekah.SummaryLoader
import ..Rebekah.JLD2Saver

"""
    run_EstherDocument(
      toml_path::String, 
      jobid::Union{Nothing,String}=nothing
    ) -> Nothing

Run the full Esther [`JLD2`](https://juliaio.github.io/JLD2.jl/stable)-generation pipeline for a single ensemble.

# Overview
- Produces a multi-target comparison report across **four prediction models** (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``).
- Reads a [`TOML`](https://toml.io/en/) config, loads/extends summary data, and writes a [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshot.

# Config ([`TOML`](https://toml.io/en/)) expectations
- The file at `toml_path` must define, at minimum, the following keys:
  `[data]`
    - `location::String`            : project root (e.g., `"./nf4_clover_wilson_finiteT"`)
    - `ensemble::String`            : ensemble name (e.g., `"L8T4b1.60k13570"`)
    - `analysis_header::String`     : analysis prefix (e.g., `"analysis"`)
    - `labels::Vector{String}`      : label indices as strings (parsed to `Int`)
    - `trains::Vector{String}`      : train indices as strings (parsed to `Int`)
    - `use_abbreviation::Bool`      : whether to abbreviate observable names in outputs
    - `TrM1_X::Vector{String}`, `TrM1_Y::String`, `TrM1_model::String`
    - `TrM2_X::Vector{String}`, `TrM2_Y::String`, `TrM2_model::String`
    - `TrM3_X::Vector{String}`, `TrM3_Y::String`, `TrM3_model::String`
    - `TrM4_X::Vector{String}`, `TrM4_Y::String`, `TrM4_model::String`
  `[abbreviation]`
    - `Dict{String,String} `mapping raw filenames to short tags (used when `use_abbreviation=true`).
  `[bootstrap]`, `[jackknife]` sections exist and are consumed upstream.

# Arguments
- `toml_path::String`            : Path to the configuration [`TOML`](https://toml.io/en/) file.
- `jobid::Union{Nothing,String}` : Optional job ID appended by [`Deborah.Sarah.NameParser.model_suffix`](@ref)
                                   to disambiguate/log runs.

# Outputs
- Console:
  - Formatted `AVG(ERR)` tables for each prediction tag (as produced by downstream steps).
- Files:
  - Results snapshot ([`JLD2`](https://juliaio.github.io/JLD2.jl/stable)):
      `<location>/<analysis_header>_<ensemble>/<analysis_header>_<overall_name>/results_<overall_name>.jld2`
      and a copy: `./results_<overall_name>.jld2` in the current working directory.

# Workflow
1. Parse configuration from `toml_path`.
2. Build code strings:
   - Abbreviation map via [`Deborah.Sarah.StringTranscoder.parse_string_dict`](@ref).
   - Encoded inputs per model via [`Deborah.Sarah.StringTranscoder.input_encoder_abbrev_dict`](@ref).
   - Full `"X_Y"` names via [`Deborah.Sarah.NameParser.make_X_Y`](@ref).
   - Model suffixes via [`Deborah.Sarah.NameParser.model_suffix`](@ref).
   - `learning` assembled from either abbreviated or full names for all four models.
3. Construct naming:
   - `analysis_ensemble = "<analysis_header>_<ensemble>"`
   - `overall_name      = "<ensemble>_<learning>"`
   - `cumulant_name     = "<analysis_header>_<overall_name>"`
4. Load and extend summary dictionary via:
   [`Deborah.Rebekah.SummaryLoader.load_summary`](@ref).
5. Save results to [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) and copy to CWD via [`Deborah.Rebekah.JLD2Saver.save_results`](@ref) and `cp(...; force=true)`.

# Notes
- This function focuses on configuration parsing, naming, summary loading, and [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) persistence.
"""
function run_EstherDocument(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)

    cfg = TOML.parsefile(toml_path)

    labels           = cfg["data"]["labels"]
    trains           = cfg["data"]["trains"]

    location         = cfg["data"]["location"]
    ensemble         = cfg["data"]["ensemble"]

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
    raw_abbrev       = cfg["abbreviation"]

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

    analysis_ensemble="$(analysis_header)_$(ensemble)"
    overall_name = "$(ensemble)_$(learning)"
    cumulant_name = "$(analysis_header)_$(overall_name)"

    keys = ["trM1", "trM2", "trM3", "trM4", 
            "Q1",   "Q2",   "Q3",   "Q4", 
            "cond", "susp", "skew", "kurt"]

    new_dict = SummaryLoader.load_summary(
        location, analysis_ensemble,
        cumulant_name, overall_name, keys, 
        labels, trains
    )

    jld2_name = joinpath(location, analysis_ensemble, cumulant_name, "results_$(overall_name).jld2")

    JLD2Saver.save_results(jld2_name, new_dict, labels, trains)

    HERE = pwd()
    jld2_HERE = joinpath(HERE, "results_$(overall_name).jld2")
    cp(jld2_name, jld2_HERE; force=true)

end

end  # module EstherDocumentRunner