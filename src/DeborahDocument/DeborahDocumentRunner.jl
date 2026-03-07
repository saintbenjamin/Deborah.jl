# ============================================================================
# src/DeborahDocument/DeborahDocumentRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DeborahDocumentRunner

import ..TOML

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Rebekah.SummaryLoader
import ..Rebekah.JLD2Saver

"""
    run_DeborahDocument(
      toml_path::String, 
      jobid::Union{Nothing,String}=nothing
    ) -> Nothing

Run the full [`Deborah.DeborahCore`](@ref) document-generation pipeline for a single ensemble.

# Overview
- Reads a [`TOML`](https://toml.io/en/) config, loads/extends summary data, saves them into [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) format.

# Config ([`TOML`](https://toml.io/en/)) expectations
- The file at `toml_path` must contain, at minimum, the following sections/keys:

  `[data]`
    - `location::String`            : project root (e.g., `"./nf4_clover_wilson_finiteT"`)
    - `ensemble::String`            : ensemble name (e.g., `"L8T4b1.60k13570"`)
    - `analysis_header::String`     : analysis prefix (e.g., `"analysis"`)
    - `X::Vector{String}`           : input observable file names (e.g., `["plaq.dat","rect.dat"]`)
    - `Y::String`                   : target observable file name (e.g., `"pbp.dat"`)
    - `model::String`               : learner/model tag (e.g., `"LightGBM"`)
    - `labels::Vector{String}`      : label indices as strings (later parsed to `Int`)
    - `trains::Vector{String}`      : train indices as strings (later parsed to `Int`)
    - `use_abbreviation::Bool`      : whether to abbreviate observable names in outputs

  `[bootstrap]`
    - `ranseed::Int`, `N_bs::Int`, `blk_size::Int`

  `[jackknife]`
    - `bin_size::Int`

  `[abbreviation]`
    - `Dict{String,String}` mapping raw filenames to short tags
      (e.g., `"pbp.dat" => "TrM1"`, `"plaq.dat" => "Plaq"`, ...)

# Arguments
- `toml_path::String`            : Path to the [`TOML`](https://toml.io/en/) configuration file.
- `jobid::Union{Nothing,String}` : Optional job identifier appended to `model` suffixes for logging/auditing.

# Outputs
- Results snapshot files:
    `results_<overall_name>.jld2` saved under the analysis path and copied to `HERE`.

# Workflow
1. Parse configuration from `toml_path`.
2. Build name strings:
   - `analysis_ensemble = "<analysis_header>_<ensemble>"`
   - `overall_name = "<ensemble>_<learning>"`
   - `cumulant_name = "<analysis_header>_<overall_name>"`
   - where `learning` is either abbreviated (`XY_code`) or full (`X_Y`) with `model`(+`jobid`) suffix.
3. Load and extend summary data via [`Deborah.Rebekah.SummaryLoader.load_summary`](@ref).
4. Save results to [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) (`results_<overall_name>.jld2`) and copy to `HERE`.

# Notes
- Abbreviation map in `[abbreviation]` is parsed through [`Deborah.Sarah.StringTranscoder.parse_string_dict`](@ref).
- Input code strings for `X`/`Y` are built with [`Deborah.Sarah.StringTranscoder.input_encoder_abbrev_dict`](@ref) (when `use_abbreviation=true`)
  and name tuples with [`Deborah.Sarah.NameParser.make_X_Y`](@ref) + [`Deborah.Sarah.NameParser.model_suffix`](@ref).
"""
function run_DeborahDocument(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)

    cfg = TOML.parsefile(toml_path)

    labels           = cfg["data"]["labels"]
    trains           = cfg["data"]["trains"]

    location         = cfg["data"]["location"]
    ensemble         = cfg["data"]["ensemble"]

    X                = cfg["data"]["X"]
    Y                = cfg["data"]["Y"]
    model            = cfg["data"]["model"]
    
    analysis_header  = cfg["data"]["analysis_header"]
    use_abbreviation = cfg["data"]["use_abbreviation"]
    raw_abbrev       = cfg["abbreviation"]

    abbreviation = StringTranscoder.parse_string_dict(raw_abbrev)

    XY_code      = StringTranscoder.input_encoder_abbrev_dict(X, Y, abbreviation)

    X_Y          = NameParser.make_X_Y(X, Y)

    model_suffix = NameParser.model_suffix(model, jobid)
    
    if use_abbreviation
        learning = "$(XY_code)_$(model_suffix)"
    else
        learning = "$(X_Y)_$(model_suffix)"
    end

    analysis_ensemble="$(analysis_header)_$(ensemble)"
    overall_name = "$(ensemble)_$(learning)"
    cumulant_name = "$(analysis_header)_$(overall_name)"

    keys = ["Deborah"]

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

end  # module DeborahDocumentRunner