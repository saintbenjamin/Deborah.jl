# ============================================================================
# src/DeborahEsther/EstherDependencyManager.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module EstherDependencyManager

import ..TOML
import ..OrderedCollections

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.ControllerCommon
import ..DeborahCore.DeborahRunner

"""
    ensure_TrM_exists(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Ensures that all required ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` data files exist for each input group.
If any expected file is missing, the function automatically invokes [`EstherDependencyManager.run_Deborah_from_Esther`](@ref)
to regenerate the missing outputs.

# Arguments
- `toml_path::String` : Path to the [`TOML`](https://toml.io/en/) configuration file specifying input features, models, and output options.
- `jobid::Union{Nothing, String}` : Optional job ID string used for logging.

# Behavior
- Parses the configuration and checks for the presence of output files associated with each `TrMi` group.
- Each `TrMi` group (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) consists of (`X`, `Y`, `model`) triplets.
- For each group, it verifies the existence of files such as `Y_info`, `Y_bc`, `YP_bc`, etc.
- If any required file is missing, [`EstherDependencyManager.run_Deborah_from_Esther`](@ref) is called to regenerate the outputs.

# Returns
- `Nothing` : This is a side-effect function that ensures required files exist or are created.
"""
function ensure_TrM_exists(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)

    JobLoggerTools.println_benji("Reading config file: $toml_path", jobid)
    
    cfg = TOML.parsefile(toml_path)

    location           = cfg["data"]["location"]
    ensemble           = cfg["data"]["ensemble"]
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
    LBP                = cfg["data"]["LBP"]
    TRP                = cfg["data"]["TRP"]
    use_abbreviation   = cfg["data"]["use_abbreviation"]
    IDX_shift          = cfg["deborah"]["IDX_shift"]
    dump_X             = cfg["deborah"]["dump_X"]
    ranseed            = cfg["bootstrap"]["ranseed"]
    N_bs               = cfg["bootstrap"]["N_bs"]
    blk_size           = cfg["bootstrap"]["blk_size"]
    method             = cfg["bootstrap"]["method"]
    bin_size           = cfg["jackknife"]["bin_size"]
    raw_abbrev         = cfg["abbreviation"]

    abbreviation = StringTranscoder.parse_string_dict(raw_abbrev)

    TrMi_X     = [TrM1_X, TrM2_X, TrM3_X, TrM4_X]
    TrMi_Y     = [TrM1_Y, TrM2_Y, TrM3_Y, TrM4_Y]
    TrMi_model = [TrM1_model, TrM2_model, TrM3_model, TrM4_model]
    TrMi_read_column_X = [TrM1_read_column_X, TrM2_read_column_X, TrM3_read_column_X, TrM4_read_column_X]
    TrMi_read_column_Y = [TrM1_read_column_Y, TrM2_read_column_Y, TrM3_read_column_Y, TrM4_read_column_Y]
    TrMi_index_column  = [TrM1_index_column,  TrM2_index_column,  TrM3_index_column,  TrM4_index_column ]
    labels = ["Y_info", "Y_tr", "Y_bc", "Y_ul", "Y_lb", "YP_tr", "YP_bc", "YP_ul"]

    for i in 1:4
        model_tag = NameParser.model_suffix(TrMi_model[i], jobid)
        suffix = model_tag * "_LBP_" * string(LBP) * "_TRP_" * string(TRP)
        collection = analysis_header * "_" * ensemble
        if use_abbreviation
            XY_code = StringTranscoder.input_encoder_abbrev_dict(TrMi_X[i], TrMi_Y[i], abbreviation)
            overall_name = "$(ensemble)_$(XY_code)_$(suffix)"
            anly_prefixes = analysis_header * "_" * ensemble * "_" * XY_code * "_" * model_tag
            traceM_name = NameParser.build_trace_name(ensemble, XY_code, TrMi_X[i], TrMi_Y[i], string(LBP), string(TRP), model_tag)
        else
            X_Y = NameParser.make_X_Y(TrMi_X[i], TrMi_Y[i])
            overall_name = "$(ensemble)_$(X_Y)_$(suffix)"
            anly_prefixes = analysis_header * "_" * ensemble * "_" * X_Y * "_" * model_tag
            traceM_name = NameParser.build_trace_name(ensemble, X_Y, TrMi_X[i], TrMi_Y[i], string(LBP), string(TRP), model_tag)
        end
        analysis_dir = "$location/$collection/$anly_prefixes/$traceM_name"

        good_flag = true
        for label in labels
            check_file = "$(analysis_dir)/$(label)_$(overall_name).dat"
            if !isfile(check_file)
                JobLoggerTools.warn_benji("[Esther.jl] $(check_file) is not found.", jobid)
                good_flag = false
            end
        end

        if !good_flag
            JobLoggerTools.println_benji("[Esther.jl] Launching Deborah.jl to generate $(overall_name) ...", jobid)
            JobLoggerTools.@logtime_benji jobid begin
                run_Deborah_from_Esther(
                    location,
                    ensemble,
                    analysis_header,
                    TrMi_X[i],
                    TrMi_Y[i],
                    TrMi_model[i],
                    TrMi_read_column_X[i],
                    TrMi_read_column_Y[i],
                    TrMi_index_column[i],
                    LBP,
                    TRP,
                    IDX_shift,
                    dump_X,
                    ranseed,
                    N_bs,
                    blk_size,
                    method,
                    bin_size,
                    overall_name,
                    abbreviation,
                    use_abbreviation,
                    jobid
                )
            end
        end
    end
end

"""
    run_Deborah_from_Esther(
        location::String,
        ensemble::String,
        analysis_header::String,
        X::Vector{String},
        Y::String,
        model::String,
        read_column_X::Vector{Int},
        read_column_Y::Int,
        index_column::Int,
        LBP::Int,
        TRP::Int,
        IDX_shift::Int,
        dump_X::Bool,
        ranseed::Int,
        N_bs::Int,
        blk_size::Int,
        method::String,
        bin_size::Int,
        overall_name::String,
        abbreviation::Dict{String,String},
        use_abbreviation::Bool,
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Invoke [`Deborah.DeborahCore`](@ref) from within [`Deborah.Esther`](@ref) if required ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` files are missing.
This function generates a temporary [`TOML`](https://toml.io/en/) configuration from the provided arguments,
writes it to disk, and launches the [`Deborah.DeborahCore`](@ref) workflow.

# Arguments
- `location::String`               : Base path for output directory.
- `ensemble::String`               : Ensemble identifier (e.g., `"L8T4b1.60k13570"`).
- `analysis_header::String`        : Analysis name prefix (e.g., `"analysis"`).
- `X::Vector{String}`              : Input feature list.
- `Y::String`                      : Target observable key.
- `model::String`                  : Model name (e.g., `"LightGBM"`).
- `read_column_X::Vector{Int}`     : ``1``-based column indices for values in each `X` file.
- `read_column_Y::Int`             : ``1``-based column index for values in the `Y` file.
- `index_column::Int`              : ``1``-based column index for configuration index.
- `LBP::Int`                       : Label group ID.
- `TRP::Int`                       : Training group ID.
- `IDX_shift::Int`                 : Index offset shift used by [`Deborah.DeborahCore`](@ref).
- `dump_X::Bool`                   : Whether to dump input matrices.
- `ranseed::Int`                   : Random seed for bootstrap.
- `N_bs::Int`                      : Number of bootstrap replicates.
- `blk_size::Int`                  : Block length used for block bootstrap (``\\ge 1``).
- `method::String`                 : Block-bootstrap scheme to use:
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB).
    - `"moving"`         — Moving Block Bootstrap (MBB).
    - `"circular"`       — Circular Block Bootstrap (CBB; wrap-around windows).
- `bin_size::Int`                  : Jackknife bin size.
- `overall_name::String`           : Unified name tag for output files.
- `abbreviation::Dict{String,String}` : Abbreviation map for input encoding.
- `use_abbreviation::Bool`         : Whether to use abbreviations in paths/filenames.
- `jobid::Union{Nothing,String}`   : Optional job ID for logging.

# Behavior
- Resolves output directory according to `use_abbreviation`.
- Saves the [`TOML`](https://toml.io/en/) as `config_Deborah_*.toml` under the `output_dir`.
- Launches [`Deborah.DeborahCore`](@ref) with the generated configuration.

# Returns
- `Nothing` — side-effecting helper.

# Notes
- `method` must be one of `"nonoverlapping"`, `"moving"`, `"circular"`; invalid values should raise an error before launching.
- If existing Deborah outputs are present and valid, this function should be a no-op (caller-dependent).
"""
function run_Deborah_from_Esther(
    location::String,
    ensemble::String,
    analysis_header::String,
    X::Vector{String},
    Y::String,
    model::String,
    read_column_X::Vector{Int},
    read_column_Y::Int,
    index_column::Int,
    LBP::Int,
    TRP::Int,
    IDX_shift::Int,
    dump_X::Bool,
    ranseed::Int,
    N_bs::Int,
    blk_size::Int,
    method::String,
    bin_size::Int,
    overall_name::String,
    abbreviation::Dict{String,String},
    use_abbreviation::Bool,
    jobid::Union{Nothing, String}=nothing
)
    toml_dict = generate_toml_dict(
        location, ensemble, analysis_header,
        X, Y, model, 
        read_column_X, read_column_Y, index_column, 
        LBP, TRP, IDX_shift,
        dump_X, ranseed, 
        N_bs, blk_size, method, bin_size,
        abbreviation, use_abbreviation
    )
    model_tag = NameParser.model_suffix(model, jobid)
    if use_abbreviation
        XY_code = StringTranscoder.input_encoder_abbrev_dict(X, Y, abbreviation)
        output_dir = joinpath(
            location, 
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(XY_code)_$(model_tag)", 
            "$(overall_name)"
        )
    else
        X_Y = NameParser.make_X_Y(X, Y)
        output_dir = joinpath(
            location, 
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(X_Y)_$(model_tag)", 
            "$(overall_name)"
        )
    end
    mkpath(output_dir)
    toml_path = joinpath(output_dir, "config_Deborah_$(overall_name).toml")
    ControllerCommon.save_toml_file(toml_dict, toml_path)

    JobLoggerTools.println_benji("[Esther.jl→Deborah.jl] Running Deborah on $overall_name", jobid)
    JobLoggerTools.@logtime_benji jobid begin
        jobid_deborah = "Deborah_LBP" * string(LBP) * "_TRP" * string(TRP)
        DeborahRunner.run_Deborah(toml_path, jobid_deborah)
    end
end

"""
    generate_toml_dict(
        location::String,
        ensemble::String,
        analysis_header::String,
        X::Vector{String},
        Y::String,
        model::String,
        read_column_X::Vector{Int},
        read_column_Y::Int,
        index_column::Int,
        LBP::Int,
        TRP::Int,
        IDX_shift::Int,
        dump_X::Bool,
        ranseed::Int,
        N_bs::Int,
        blk_size::Int,
        method::String,
        bin_size::Int,
        abbreviation::Dict{String,String},
        use_abbreviation::Bool
    ) -> Dict

Construct a [`TOML`](https://toml.io/en/)-compatible configuration dictionary for the [`Deborah.DeborahCore`](@ref) workflow.

# Arguments
- `location::String`            : Base directory for outputs.
- `ensemble::String`            : Ensemble identifier (e.g., `"L8T4b1.60k13570"`).
- `analysis_header::String`     : Analysis folder prefix (e.g., `"analysis"`).
- `X::Vector{String}`           : Input feature keys.
- `Y::String`                   : Target observable key.
- `model::String`               : Model type (e.g., `"LightGBM"`, `"Lasso"`).
- `read_column_X::Vector{Int}`  : ``1``-based value-column indices for each `X` file.
- `read_column_Y::Int`          : ``1``-based value-column index for the `Y` file.
- `index_column::Int`           : ``1``-based column index of configuration IDs in files.
- `LBP::Int`                    : Label group ID (label partition parameter).
- `TRP::Int`                    : Training group ID (training partition parameter).
- `IDX_shift::Int`              : Index offset/shift used by [`Deborah.DeborahCore`](@ref).
- `dump_X::Bool`                : Whether to dump input matrices.
- `ranseed::Int`                : Random seed for bootstrap.
- `N_bs::Int`                   : Number of bootstrap replicates.
- `blk_size::Int`               : Block length for block bootstrap (``\\ge 1``).
- `method::String`              : Block-bootstrap scheme to encode in [`TOML`](https://toml.io/en/):
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB)
    - `"moving"`         — Moving Block Bootstrap (MBB)
    - `"circular"`       — Circular Block Bootstrap (CBB)
- `bin_size::Int`               : Jackknife bin size.
- `abbreviation::Dict{String,String}` : Abbreviation map for input encoding.
- `use_abbreviation::Bool`      : Use abbreviations in paths/filenames if `true`.

# Returns
- `Dict` : A nested, [`TOML`](https://toml.io/en/)-ready dictionary including (at minimum) sections/keys for:
  - data/paths (`location`, `ensemble`, `analysis_header`, `abbreviations`),
  - IO columns (`read_column_X`, `read_column_Y`, `index_column`, `IDX_shift`, `dump_X`),
  - bootstrap (`ranseed`, `N_bs`, `blk_size`, `method`),
  - jackknife (`bin_size`),
  - partitions (`LBP`, `TRP`),
  - model (`model`, `X`, `Y`).

# Notes
- All column indices are ``1``-based.
- `method` must be one of the three literals above; invalid values should be rejected by the caller.
"""
function generate_toml_dict(
    location::String,
    ensemble::String,
    analysis_header::String,
    X::Vector{String},
    Y::String,
    model::String,
    read_column_X::Vector{Int},
    read_column_Y::Int,
    index_column::Int,
    LBP::Int,
    TRP::Int,
    IDX_shift::Int,
    dump_X::Bool,
    ranseed::Int,
    N_bs::Int,
    blk_size::Int,
    method::String,
    bin_size::Int,
    abbreviation::Dict{String,String},
    use_abbreviation::Bool
)::Dict
    return OrderedCollections.OrderedDict(
        "data" => OrderedCollections.OrderedDict(
            "location" => location,
            "ensemble" => ensemble,
            "analysis_header" => analysis_header,
            "X" => X,
            "Y" => Y,
            "model" => model,
            "read_column_X" => read_column_X,
            "read_column_Y" => read_column_Y,
            "index_column" => index_column,
            "LBP" => LBP,
            "TRP" => TRP,
            "IDX_shift" => IDX_shift,
            "dump_X" => dump_X,
            "use_abbreviation" => use_abbreviation
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
        "abbreviation" => OrderedCollections.OrderedDict(abbreviation)
    )
end

end  # module EstherDependencyManager