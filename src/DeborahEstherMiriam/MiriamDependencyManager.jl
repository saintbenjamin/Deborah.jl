# ============================================================================
# src/DeborahEstherMiriam/MiriamDependencyManager.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MiriamDependencyManager

import ..TOML
import ..OrderedCollections

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.ControllerCommon
import ..Esther.EstherRunner
import ..DeborahEsther.EstherDependencyManager

"""
    ensure_ensemble_exists(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Ensures that all necessary outputs for each ensemble exist by invoking the [`Deborah.Esther`](@ref) pipeline.
For each ensemble listed in the configuration, this function checks or generates data by 
calling [`MiriamDependencyManager.run_Esther_from_Miriam`](@ref).

# Arguments
- `toml_path::String`                 : Path to the [`TOML`](https://toml.io/en/) configuration file.
- `jobid::Union{Nothing, String}`     : Optional job ID for logging purposes.

# Behavior
- Parses the given [`TOML`](https://toml.io/en/) file.
- Iterates through all listed ensembles and associated ``\\kappa`` values.
- For each ensemble, invokes [`MiriamDependencyManager.run_Esther_from_Miriam`](@ref) to ensure outputs are created.

# Returns
- `Nothing` : All effects are side-effectual (file generation/logging).
"""
function ensure_ensemble_exists(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)

    JobLoggerTools.println_benji("Reading config file: $toml_path", jobid)

    cfg = TOML.parsefile(toml_path)

    location           = cfg["data"]["location"]
    ensembles          = cfg["data"]["ensembles"]
    analysis_header    = cfg["data"]["analysis_header"]
    TrM1_X             = cfg["data"]["TrM1_X"]
    TrM1_Y             = cfg["data"]["TrM1_Y"]
    TrM1_model         = cfg["data"]["TrM1_model"]
    TrM1_read_column_X = cfg["data"]["TrM1_read_column_X"]
    TrM1_read_column_Y = cfg["data"]["TrM1_read_column_Y"]
    TrM1_index_column  = cfg["data"]["TrM1_index_column"]
    TrM2_X             = cfg["data"]["TrM2_X"]
    TrM2_Y             = cfg["data"]["TrM2_Y"]
    TrM2_model         = cfg["data"]["TrM2_model"]
    TrM2_read_column_X = cfg["data"]["TrM2_read_column_X"]
    TrM2_read_column_Y = cfg["data"]["TrM2_read_column_Y"]
    TrM2_index_column  = cfg["data"]["TrM2_index_column"]
    TrM3_X             = cfg["data"]["TrM3_X"]
    TrM3_Y             = cfg["data"]["TrM3_Y"]
    TrM3_model         = cfg["data"]["TrM3_model"]
    TrM3_read_column_X = cfg["data"]["TrM3_read_column_X"]
    TrM3_read_column_Y = cfg["data"]["TrM3_read_column_Y"]
    TrM3_index_column  = cfg["data"]["TrM3_index_column"]
    TrM4_X             = cfg["data"]["TrM4_X"]
    TrM4_Y             = cfg["data"]["TrM4_Y"]
    TrM4_model         = cfg["data"]["TrM4_model"]
    TrM4_read_column_X = cfg["data"]["TrM4_read_column_X"]
    TrM4_read_column_Y = cfg["data"]["TrM4_read_column_Y"]
    TrM4_index_column  = cfg["data"]["TrM4_index_column"]
    use_abbreviation   = cfg["data"]["use_abbreviation"]
    LBP                = cfg["data"]["LBP"]
    TRP                = cfg["data"]["TRP"]
    ns                 = cfg["input_meta"]["ns"]
    nt                 = cfg["input_meta"]["nt"]
    nf                 = cfg["input_meta"]["nf"]
    beta               = cfg["input_meta"]["beta"]
    kappa_list         = cfg["input_meta"]["kappa_list"]
    bin_size           = cfg["jackknife"]["bin_size"]
    ranseed            = cfg["bootstrap"]["ranseed"]
    N_bs               = cfg["bootstrap"]["N_bs"]
    blk_size           = cfg["bootstrap"]["blk_size"]
    method             = cfg["bootstrap"]["method"]
    IDX_shift          = cfg["deborah"]["IDX_shift"]
    dump_X             = cfg["deborah"]["dump_X"]
    raw_abbrev         = cfg["abbreviation"]

    abbreviation = StringTranscoder.parse_string_dict(raw_abbrev)
    TrM1_code    = StringTranscoder.input_encoder_abbrev_dict(TrM1_X, TrM1_Y, abbreviation)
    TrM2_code    = StringTranscoder.input_encoder_abbrev_dict(TrM2_X, TrM2_Y, abbreviation)
    TrM3_code    = StringTranscoder.input_encoder_abbrev_dict(TrM3_X, TrM3_Y, abbreviation)
    TrM4_code    = StringTranscoder.input_encoder_abbrev_dict(TrM4_X, TrM4_Y, abbreviation)
    TrM1_X_Y     = NameParser.make_X_Y(TrM1_X, TrM1_Y)
    TrM2_X_Y     = NameParser.make_X_Y(TrM2_X, TrM2_Y)
    TrM3_X_Y     = NameParser.make_X_Y(TrM3_X, TrM3_Y)
    TrM4_X_Y     = NameParser.make_X_Y(TrM4_X, TrM4_Y)
    TrM1_suffix  = NameParser.model_suffix(TrM1_model, jobid)
    TrM2_suffix  = NameParser.model_suffix(TrM2_model, jobid)
    TrM3_suffix  = NameParser.model_suffix(TrM3_model, jobid)
    TrM4_suffix  = NameParser.model_suffix(TrM4_model, jobid)

    for (i, (ensemble, kappa)) in enumerate(zip(ensembles, kappa_list))
        JobLoggerTools.println_benji("[$i] Ensemble: $ensemble | Kappa: $kappa", jobid)
        collection = analysis_header * "_" * ensemble
        if use_abbreviation
            overall_name = "$(ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)_LBP_$(string(LBP))_TRP_$(string(TRP))"
            anly_prefixes = "$(analysis_header)_$(ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)"
        else
            overall_name = "$(ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)_LBP_$(string(LBP))_TRP_$(string(TRP))"
            anly_prefixes = "$(analysis_header)_$(ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)"
        end
        analysis_dir = "$(location)/$(collection)/$(anly_prefixes)/$(overall_name)"

        good_flag = true
        check_file = joinpath(analysis_dir, "summary_Esther_$(overall_name).dat")
        if !isfile(check_file)
            JobLoggerTools.warn_benji("[Esther.jl] $(check_file) is not found.", jobid)
            good_flag = false
        end
        JobLoggerTools.@logtime_benji jobid begin
            run_Esther_from_Miriam(
                location,
                ensemble,
                TrM1_X,
                TrM1_Y,
                TrM1_model,
                TrM1_read_column_X,
                TrM1_read_column_Y,
                TrM1_index_column,
                TrM2_X,
                TrM2_Y,
                TrM2_model,
                TrM2_read_column_X,
                TrM2_read_column_Y,
                TrM2_index_column,
                TrM3_X,
                TrM3_Y,
                TrM3_model,
                TrM3_read_column_X,
                TrM3_read_column_Y,
                TrM3_index_column,
                TrM4_X,
                TrM4_Y,
                TrM4_model,
                TrM4_read_column_X,
                TrM4_read_column_Y,
                TrM4_index_column,
                LBP,
                TRP,
                ns,
                nt,
                nf,
                beta,
                kappa,
                ranseed,
                N_bs,
                blk_size,
                method,
                bin_size,
                analysis_header,
                IDX_shift,
                dump_X,
                overall_name,
                abbreviation,
                use_abbreviation,
                good_flag,
                jobid
            )
        end
    end
end

"""
    run_Esther_from_Miriam(
        location::String,
        ensemble::String,
        TrM1_X::Vector{String},
        TrM1_Y::String,
        TrM1_model::String,
        TrM1_read_column_X::Vector{Int},
        TrM1_read_column_Y::Int,
        TrM1_index_column::Int,
        TrM2_X::Vector{String},
        TrM2_Y::String,
        TrM2_model::String,
        TrM2_read_column_X::Vector{Int},
        TrM2_read_column_Y::Int,
        TrM2_index_column::Int,
        TrM3_X::Vector{String},
        TrM3_Y::String,
        TrM3_model::String,
        TrM3_read_column_X::Vector{Int},
        TrM3_read_column_Y::Int,
        TrM3_index_column::Int,
        TrM4_X::Vector{String},
        TrM4_Y::String,
        TrM4_model::String,
        TrM4_read_column_X::Vector{Int},
        TrM4_read_column_Y::Int,
        TrM4_index_column::Int,
        LBP::Int,
        TRP::Int,
        ns::Int,
        nt::Int,
        nf::Int,
        beta::Float64,
        kappa::Float64,
        ranseed::Int,
        N_bs::Int,
        blk_size::Int,
        method::String,
        bin_size::Int,
        analysis_header::String,
        IDX_shift::Int,
        dump_X::Bool,
        overall_name::String,
        abbreviation::Dict{String,String},
        use_abbreviation::Bool,
        good_flag::Bool,
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Run the full [`Deborah.Esther`](@ref) pipeline for a single ensemble, using four ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` inputs
(as configured by [`Deborah.Miriam`](@ref)). Creates/reads required ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``, performs modeling and
error estimation, and writes results to disk.

# Arguments
- `location::String` : Root directory containing ensemble data.
- `ensemble::String` : Ensemble identifier (e.g., `"L8T4b1.60k13580"`).
- `TrM*_X::Vector{String}` : `X`-observable names for each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` input.
- `TrM*_Y::String` : `Y` observable name for each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` input.
- `TrM*_model::String` : Model name used for each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` input.
- `TrM*_read_column_X::Vector{Int}` : ``1``-based value-column indices for each `X` file.
- `TrM*_read_column_Y::Int` : ``1``-based value-column index for each `Y` trace file.
- `TrM*_index_column::Int` : ``1``-based column index of configuration IDs in files.
- `LBP::Int` : Labeled set partition/group identifier.
- `TRP::Int` : Training set partition/group identifier.
- `ns`, `nt`, `nf` (`::Int`) : Lattice meta (spatial size, temporal size, and an integer meta parameter used by downstream transforms).
- `beta::Float64`, `kappa::Float64` : Simulation parameters.
- `ranseed::Int` : Random seed for bootstrap.
- `N_bs::Int` : Number of bootstrap replicates.
- `blk_size::Int` : Block length for block bootstrap (``\\ge 1``).
- `method::String` : Block-bootstrap scheme:
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB)
    - `"moving"` — Moving Block Bootstrap (MBB)
    - `"circular"` — Circular Block Bootstrap (CBB; wrap-around windows)
- `bin_size::Int` : Jackknife bin size.
- `analysis_header::String` : Analysis prefix (e.g., `"analysis"`).
- `IDX_shift::Int` : Optional index shift applied to real datasets.
- `dump_X::Bool` : Whether to save/dump input `X` matrices.
- `overall_name::String` : Directory-safe tag for output labeling.
- `abbreviation::Dict{String,String}` : Mapping of variable names to encoded tokens.
- `use_abbreviation::Bool` : If `true`, use abbreviation-based paths/filenames.
- `good_flag::Bool` : Caller-provided gate to proceed or skip (e.g., sanity check outcome).
- `jobid::Union{Nothing,String}` : Optional job identifier for structured logging.

# Behavior
- Resolves paths (respecting `use_abbreviation`) and prepares per-``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` inputs.
- If required ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` files are missing, may trigger [`Deborah.DeborahCore`](@ref) generation
  with a [`TOML`](https://toml.io/en/) that includes `ranseed`, `N_bs`, `blk_size`, `method`, and `bin_size`.
- Runs modeling per ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``, assembles outputs, and performs uncertainty estimates via
  block bootstrap (`N_bs`, `blk_size`, `method`) and jackknife (`bin_size`).

# Returns
- `Nothing` — side-effecting orchestration routine.

# Notes
- `method` must be one of `"nonoverlapping"`, `"moving"`, `"circular"`; invalid values should be rejected early.
- All column indices are **``1``-based**.
- Downstream components assume consistency between lattice meta (`ns`, `nt`, `nf`) and the loaded traces.
"""
function run_Esther_from_Miriam(
    location::String,
    ensemble::String,
    TrM1_X::Vector{String},
    TrM1_Y::String,
    TrM1_model::String,
    TrM1_read_column_X::Vector{Int},
    TrM1_read_column_Y::Int,
    TrM1_index_column::Int,
    TrM2_X::Vector{String},
    TrM2_Y::String,
    TrM2_model::String,
    TrM2_read_column_X::Vector{Int},
    TrM2_read_column_Y::Int,
    TrM2_index_column::Int,
    TrM3_X::Vector{String},
    TrM3_Y::String,
    TrM3_model::String,
    TrM3_read_column_X::Vector{Int},
    TrM3_read_column_Y::Int,
    TrM3_index_column::Int,
    TrM4_X::Vector{String},
    TrM4_Y::String,
    TrM4_model::String,
    TrM4_read_column_X::Vector{Int},
    TrM4_read_column_Y::Int,
    TrM4_index_column::Int,
    LBP::Int,
    TRP::Int,
    ns::Int,
    nt::Int,
    nf::Int,
    beta::Float64,
    kappa::Float64,
    ranseed::Int,
    N_bs::Int,
    blk_size::Int,
    method::String,
    bin_size::Int,
    analysis_header::String,
    IDX_shift::Int,
    dump_X::Bool,
    overall_name::String,
    abbreviation::Dict{String,String},
    use_abbreviation::Bool,
    good_flag::Bool,
    jobid::Union{Nothing, String}=nothing
)
    toml_dict = generate_toml_dict(
        location, ensemble,
        TrM1_X, TrM1_Y, TrM1_model, TrM1_read_column_X, TrM1_read_column_Y, TrM1_index_column,
        TrM2_X, TrM2_Y, TrM2_model, TrM2_read_column_X, TrM2_read_column_Y, TrM2_index_column,
        TrM3_X, TrM3_Y, TrM3_model, TrM3_read_column_X, TrM3_read_column_Y, TrM3_index_column,
        TrM4_X, TrM4_Y, TrM4_model, TrM4_read_column_X, TrM4_read_column_Y, TrM4_index_column,
        LBP, TRP, ns, nt, nf, beta, kappa,
        ranseed, N_bs, blk_size, method,
        bin_size,
        analysis_header, IDX_shift, 
        dump_X,
        abbreviation, use_abbreviation
    )

    TrM1_code   = StringTranscoder.input_encoder_abbrev_dict(TrM1_X, TrM1_Y, abbreviation)
    TrM2_code   = StringTranscoder.input_encoder_abbrev_dict(TrM2_X, TrM2_Y, abbreviation)
    TrM3_code   = StringTranscoder.input_encoder_abbrev_dict(TrM3_X, TrM3_Y, abbreviation)
    TrM4_code   = StringTranscoder.input_encoder_abbrev_dict(TrM4_X, TrM4_Y, abbreviation)
    TrM1_X_Y    = NameParser.make_X_Y(TrM1_X, TrM1_Y)
    TrM2_X_Y    = NameParser.make_X_Y(TrM2_X, TrM2_Y)
    TrM3_X_Y    = NameParser.make_X_Y(TrM3_X, TrM3_Y)
    TrM4_X_Y    = NameParser.make_X_Y(TrM4_X, TrM4_Y)
    TrM1_suffix = NameParser.model_suffix(TrM1_model, jobid)
    TrM2_suffix = NameParser.model_suffix(TrM2_model, jobid)
    TrM3_suffix = NameParser.model_suffix(TrM3_model, jobid)
    TrM4_suffix = NameParser.model_suffix(TrM4_model, jobid)

    if use_abbreviation
        output_dir = joinpath(
            location, 
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)", 
            "$(overall_name)"
        )
    else
        output_dir = joinpath(
            location, 
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)", 
            "$(overall_name)"
        )
    end
    mkpath(output_dir)
    toml_path = joinpath(
        output_dir, 
        "config_Esther_$(overall_name).toml"
    )
    ControllerCommon.save_toml_file(toml_dict, toml_path)

    JobLoggerTools.println_benji("[Miriam.jl→EstherChecker.jl] Running EstherChecker.jl on $overall_name", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        jobid_esther = "Esther_LBP" * string(LBP) * "_TRP" * string(TRP)
        EstherDependencyManager.ensure_TrM_exists(toml_path, jobid_esther)
    end

    if !good_flag
        JobLoggerTools.println_benji("[Miriam.jl→Esther.jl] Running Esther.jl on $overall_name", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            jobid_esther = "Esther_LBP" * string(LBP) * "_TRP" * string(TRP)
            EstherRunner.run_Esther(toml_path, jobid_esther)
        end
    end

end

"""
    generate_toml_dict(
        location::String,
        ensemble::String,
        TrM1_X::Vector{String},
        TrM1_Y::String,
        TrM1_model::String,
        TrM1_read_column_X::Vector{Int},
        TrM1_read_column_Y::Int,
        TrM1_index_column::Int,
        TrM2_X::Vector{String},
        TrM2_Y::String,
        TrM2_model::String,
        TrM2_read_column_X::Vector{Int},
        TrM2_read_column_Y::Int,
        TrM2_index_column::Int,
        TrM3_X::Vector{String},
        TrM3_Y::String,
        TrM3_model::String,
        TrM3_read_column_X::Vector{Int},
        TrM3_read_column_Y::Int,
        TrM3_index_column::Int,
        TrM4_X::Vector{String},
        TrM4_Y::String,
        TrM4_model::String,
        TrM4_read_column_X::Vector{Int},
        TrM4_read_column_Y::Int,
        TrM4_index_column::Int,
        LBP::Int,
        TRP::Int,
        ns::Int,
        nt::Int,
        nf::Int,
        beta::Float64,
        kappa::Float64,
        ranseed::Int,
        N_bs::Int,
        blk_size::Int,
        method::String,
        bin_size::Int,
        analysis_header::String,
        IDX_shift::Int,
        dump_X::Bool,
        abbreviation::Dict{String,String},
        use_abbreviation::Bool
    ) -> Dict

Generate a [`TOML`](https://toml.io/en/)-ready nested dictionary capturing all parameters needed for the
[`Deborah.DeborahCore`](@ref)-[`Deborah.Esther`](@ref)-[`Deborah.Miriam`](@ref) pipeline.

# Arguments
- `location::String`  : Root path for outputs.
- `ensemble::String`  : Ensemble identifier (e.g., `"L8T4b1.60k13570"`).
- `TrMi_X`, `TrMi_Y`, `TrMi_model` : Input–output specs and model for ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
- `TrM*_read_column_X::Vector{Int}` : ``1``-based value-column indices for each `X` file.
- `TrM*_read_column_Y::Int`         : ``1``-based value-column index for each `Y` file.
- `TrM*_index_column::Int`          : ``1``-based column index of configuration IDs.
- `LBP::Int`, `TRP::Int`            : Label/training partition identifiers used across the pipeline.
- `ns::Int`, `nt::Int`, `nf::Int`   : Lattice/meta parameters consumed downstream.
- `beta::Float64`, `kappa::Float64` : Simulation parameters.
- `ranseed::Int`, `N_bs::Int`, `blk_size::Int` : Bootstrap configuration.
- `method::String`                  : Block-bootstrap scheme to encode in [`TOML`](https://toml.io/en/):
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB)
    - `"moving"`         — Moving Block Bootstrap (MBB)
    - `"circular"`       — Circular Block Bootstrap (CBB; wrap-around windows)
- `bin_size::Int`                   : Jackknife bin size.
- `analysis_header::String`         : Analysis group/prefix (e.g., `"analysis"`).
- `IDX_shift::Int`                  : Index offset/shift propagated to [`Deborah.DeborahCore`](@ref).
- `dump_X::Bool`                    : Whether to dump/save input `X` matrices.
- `abbreviation::Dict{String,String}` : Abbreviation map for feature/path encoding.
- `use_abbreviation::Bool`          : If `true`, use abbreviation-based names/paths.

# Returns
- `Dict` : A nested [`TOML`](https://toml.io/en/)-compatible dictionary including sections/keys for:
  - data/paths (`location`, `ensemble`, `analysis_header`, `abbreviation`, `use_abbreviation`),
  - IO columns (`read_column_X`, `read_column_Y`, `index_column`, `IDX_shift`, `dump_X`),
  - bootstrap (`ranseed`, `N_bs`, `blk_size`, `method`),
  - jackknife (`bin_size`),
  - partitions (`LBP`, `TRP`),
  - model & targets (per-``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` `X`, `Y`, `model`),
  - lattice/meta (`ns`, `nt`, `nf`, `beta`, `kappa`).

# Notes
- All column indices are ``1``-based.
- `method` must be one of the three literal strings above; callers should validate before writing [`TOML`](https://toml.io/en/).
"""
function generate_toml_dict(
    location::String,
    ensemble::String,
    TrM1_X::Vector{String},
    TrM1_Y::String,
    TrM1_model::String,
    TrM1_read_column_X::Vector{Int},
    TrM1_read_column_Y::Int,
    TrM1_index_column::Int,
    TrM2_X::Vector{String},
    TrM2_Y::String,
    TrM2_model::String,
    TrM2_read_column_X::Vector{Int},
    TrM2_read_column_Y::Int,
    TrM2_index_column::Int,
    TrM3_X::Vector{String},
    TrM3_Y::String,
    TrM3_model::String,
    TrM3_read_column_X::Vector{Int},
    TrM3_read_column_Y::Int,
    TrM3_index_column::Int,
    TrM4_X::Vector{String},
    TrM4_Y::String,
    TrM4_model::String,
    TrM4_read_column_X::Vector{Int},
    TrM4_read_column_Y::Int,
    TrM4_index_column::Int,
    LBP::Int,
    TRP::Int,
    ns::Int,
    nt::Int,
    nf::Int,
    beta::Float64,
    kappa::Float64,
    ranseed::Int,
    N_bs::Int,
    blk_size::Int,
    method::String,
    bin_size::Int,
    analysis_header::String,
    IDX_shift::Int,
    dump_X::Bool,
    abbreviation::Dict{String,String},
    use_abbreviation::Bool
)::Dict
    return OrderedCollections.OrderedDict(
        "data" => OrderedCollections.OrderedDict(
            "location" => location,
            "ensemble" => ensemble,
            "analysis_header" => analysis_header,
            "TrM1_X" => TrM1_X,
            "TrM1_Y" => TrM1_Y,
            "TrM1_model" => TrM1_model,
            "TrM1_read_column_X" => TrM1_read_column_X,
            "TrM1_read_column_Y" => TrM1_read_column_Y,
            "TrM1_index_column" => TrM1_index_column,
            "TrM2_X" => TrM2_X,
            "TrM2_Y" => TrM2_Y,
            "TrM2_model" => TrM2_model,
            "TrM2_read_column_X" => TrM2_read_column_X,
            "TrM2_read_column_Y" => TrM2_read_column_Y,
            "TrM2_index_column" => TrM2_index_column,
            "TrM3_X" => TrM3_X,
            "TrM3_Y" => TrM3_Y,
            "TrM3_model" => TrM3_model,
            "TrM3_read_column_X" => TrM3_read_column_X,
            "TrM3_read_column_Y" => TrM3_read_column_Y,
            "TrM3_index_column" => TrM3_index_column,
            "TrM4_X" => TrM4_X,
            "TrM4_Y" => TrM4_Y,
            "TrM4_model" => TrM4_model,
            "TrM4_read_column_X" => TrM4_read_column_X,
            "TrM4_read_column_Y" => TrM4_read_column_Y,
            "TrM4_index_column" => TrM4_index_column,
            "LBP" => LBP,
            "TRP" => TRP,
            "use_abbreviation" => use_abbreviation
        ),
        "input_meta" => OrderedCollections.OrderedDict(
            "ns" => ns,
            "nt" => nt,
            "nf" => nf,
            "beta" => beta,
            "kappa" => kappa
        ),
        "bootstrap" => OrderedCollections.OrderedDict(
            "ranseed" => ranseed,
            "N_bs" => N_bs,
            "blk_size" => blk_size,
            "method" => method
        ),
        "jackknife" => OrderedCollections.OrderedDict(
            "bin_size" => bin_size
        ),
        "deborah" => OrderedCollections.OrderedDict(
            "IDX_shift" => IDX_shift,
            "dump_X" => dump_X
        ),
        "abbreviation" => OrderedCollections.OrderedDict(abbreviation)
    )
end

end  # module MiriamDependencyManager