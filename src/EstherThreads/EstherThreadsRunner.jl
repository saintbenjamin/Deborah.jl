# ============================================================================
# src/EstherThreads/EstherThreadsRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module EstherThreadsRunner

import TOML
import OrderedCollections
import Dates
import Base.Threads: @spawn, nthreads
import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.ControllerCommon
import ..Esther.EstherRunner
import ..DeborahEsther.EstherDependencyManager

"""
    struct JobArgsEsther

Container for all job parameters required to execute a single [`Deborah.Esther`](@ref) computation.

# Fields
- `location::String`               : Root directory for data and outputs.
- `ensemble::String`               : Ensemble identifier.
- `analysis_header::String`        : Prefix for analysis/output directories.

- `TrM1_X::Vector{String}`         : Input feature names for ``\\text{Tr} \\, M^{-1}``.
- `TrM1_Y::String`                 : Target observable for ``\\text{Tr} \\, M^{-1}``.
- `TrM1_model::String`             : Model name for ``\\text{Tr} \\, M^{-1}`` (e.g., `"LightGBM"`).
- `TrM1_read_column_X::Vector{Int}`: ``1``-based value-column indices for `X` (features) of ``\\text{Tr} \\, M^{-1}``.
- `TrM1_read_column_Y::Int`        : ``1``-based value-column index for ``\\text{Tr} \\, M^{-1}``.
- `TrM1_index_column::Int`         : ``1``-based column index of configuration IDs (``\\text{Tr} \\, M^{-1}``).

- `TrM2_X::Vector{String}`         : Input feature names for ``\\text{Tr} \\, M^{-2}``.
- `TrM2_Y::String`                 : Target observable for ``\\text{Tr} \\, M^{-2}``.
- `TrM2_model::String`             : Model name for ``\\text{Tr} \\, M^{-2}``.
- `TrM2_read_column_X::Vector{Int}`: ``1``-based value-column indices for `X` (features) of ``\\text{Tr} \\, M^{-2}``.
- `TrM2_read_column_Y::Int`        : ``1``-based value-column index for ``\\text{Tr} \\, M^{-2}``
- `TrM2_index_column::Int`         : ``1``-based column index of configuration IDs (``\\text{Tr} \\, M^{-2}``).

- `TrM3_X::Vector{String}`         : Input feature names for ``\\text{Tr} \\, M^{-3}``.
- `TrM3_Y::String`                 : Target observable for ``\\text{Tr} \\, M^{-3}``.
- `TrM3_model::String`             : Model name for ``\\text{Tr} \\, M^{-3}``.
- `TrM3_read_column_X::Vector{Int}`: ``1``-based value-column indices for `X` (features) of ``\\text{Tr} \\, M^{-3}``.
- `TrM3_read_column_Y::Int`        : ``1``-based value-column index for ``\\text{Tr} \\, M^{-3}``.
- `TrM3_index_column::Int`         : ``1``-based column index of configuration IDs (``\\text{Tr} \\, M^{-3}``).

- `TrM4_X::Vector{String}`         : Input feature names for ``\\text{Tr} \\, M^{-4}``.
- `TrM4_Y::String`                 : Target observable for ``\\text{Tr} \\, M^{-4}``.
- `TrM4_model::String`             : Model name for ``\\text{Tr} \\, M^{-4}``.
- `TrM4_read_column_X::Vector{Int}`: ``1``-based value-column indices for `X` (features) of ``\\text{Tr} \\, M^{-4}``.
- `TrM4_read_column_Y::Int`        : ``1``-based value-column index for ``\\text{Tr} \\, M^{-4}``.
- `TrM4_index_column::Int`         : ``1``-based column index of configuration IDs (``\\text{Tr} \\, M^{-4}``).

- `label::String`                  : LBP value (as string) propagated to configs/paths.
- `train::String`                  : TRP value (as string) propagated to configs/paths.

- `ns::Int`, `nt::Int`, `nf::Int`  : Lattice/meta parameters used downstream.
- `beta::Float64`, `kappa::Float64`: Simulation parameters.

- `ranseed::Int`                   : Random seed for bootstrap.
- `N_bs::Int`                      : Number of bootstrap replicates.
- `blk_size::Int`                  : Block length for block bootstrap (``\\ge 1``).
- `method::String`                 : Block-bootstrap scheme:
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB)
    - `"moving"`         — Moving Block Bootstrap (MBB)
    - `"circular"`       — Circular Block Bootstrap (CBB; wrap-around windows)
- `bin_size::Int`                  : Jackknife bin size.
- `IDX_shift::Int`                 : Index shift forwarded to [`Deborah.DeborahCore`](@ref) processing.
- `dump_X::Bool`                   : Whether to dump/save `X` feature matrices.

- `overall_name::String`           : Directory-safe tag for output labeling.
- `abbreviation::Dict{String,String}` : Abbreviation dictionary for feature/path encoding.
- `use_abbreviation::Bool`         : If `true`, use abbreviation-based paths/filenames.
- `output_base::String`            : Base output directory (resolved root for artifacts).

# Notes
- All column indices are **``1``-based**.
- `method` must be one of the three literal strings above; validate before use.
"""
struct JobArgsEsther
    location::String
    ensemble::String
    analysis_header::String
    TrM1_X::Vector{String}
    TrM1_Y::String
    TrM1_model::String
    TrM1_read_column_X::Vector{Int}
    TrM1_read_column_Y::Int
    TrM1_index_column::Int
    TrM2_X::Vector{String}
    TrM2_Y::String
    TrM2_model::String
    TrM2_read_column_X::Vector{Int}
    TrM2_read_column_Y::Int
    TrM2_index_column::Int
    TrM3_X::Vector{String}
    TrM3_Y::String
    TrM3_model::String
    TrM3_read_column_X::Vector{Int}
    TrM3_read_column_Y::Int
    TrM3_index_column::Int
    TrM4_X::Vector{String}
    TrM4_Y::String
    TrM4_model::String
    TrM4_read_column_X::Vector{Int}
    TrM4_read_column_Y::Int
    TrM4_index_column::Int
    label::String
    train::String
    ns::Int
    nt::Int
    beta::Float64
    kappa::Float64
    nf::Int
    ranseed::Int
    N_bs::Int
    blk_size::Int
    method::String
    bin_size::Int
    IDX_shift::Int
    dump_X::Bool
    overall_name::String
    abbreviation::Dict{String, String}
    use_abbreviation::Bool
    output_base::String
end

"""
    parse_config_EstherThreads(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Vector{JobArgsEsther}

Parse a [`TOML`](https://toml.io/en/) configuration file and return a list of [`JobArgsEsther`](@ref) structs  
for all `(LBP, TRP)` combinations defined in the config.

# Arguments
- `toml_path::String`: Path to the [`TOML`](https://toml.io/en/) config file.
- `jobid::Union{Nothing, String}` : Optional job ID string used for logging.

# Returns
- `Vector{JobArgsEsther}`: All [`Deborah.Esther`](@ref) jobs to be run.
"""
function parse_config_EstherThreads(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)::Vector{JobArgsEsther}
    
    JobLoggerTools.println_benji("Loading config from: $toml_path")
    
    cfg = TOML.parsefile(toml_path)

    # Sweep ranges
    labels             = cfg["data"]["labels"]
    trains             = cfg["data"]["trains"]

    # Common metadata
    location           = cfg["data"]["location"]
    ensemble           = cfg["data"]["ensemble"]
    analysis_header    = cfg["data"]["analysis_header"]

    # TrM1–4 specification
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

    # Input meta
    ns                 = cfg["input_meta"]["ns"]
    nt                 = cfg["input_meta"]["nt"]
    nf                 = cfg["input_meta"]["nf"]
    beta               = cfg["input_meta"]["beta"]
    kappa              = cfg["input_meta"]["kappa"]

    # Bootstrap & jackknife settings
    ranseed            = cfg["bootstrap"]["ranseed"]
    N_bs               = cfg["bootstrap"]["N_bs"]
    blk_size           = cfg["bootstrap"]["blk_size"]
    method             = cfg["bootstrap"]["method"]
    bin_size           = cfg["jackknife"]["bin_size"]

    # Deborah-style options (used internally by Esther)
    IDX_shift          = cfg["deborah"]["IDX_shift"]
    dump_X             = cfg["deborah"]["dump_X"]
    raw_abbrev         = cfg["abbreviation"]

    abbreviation = StringTranscoder.StringTranscoder.parse_string_dict(raw_abbrev)

    TrM1_code    = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM1_X, TrM1_Y, abbreviation)
    TrM2_code    = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM2_X, TrM2_Y, abbreviation)
    TrM3_code    = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM3_X, TrM3_Y, abbreviation)
    TrM4_code    = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM4_X, TrM4_Y, abbreviation)

    TrM1_X_Y     = NameParser.NameParser.make_X_Y(TrM1_X, TrM1_Y)
    TrM2_X_Y     = NameParser.NameParser.make_X_Y(TrM2_X, TrM2_Y)
    TrM3_X_Y     = NameParser.NameParser.make_X_Y(TrM3_X, TrM3_Y)
    TrM4_X_Y     = NameParser.NameParser.make_X_Y(TrM4_X, TrM4_Y)

    TrM1_suffix  = NameParser.NameParser.model_suffix(TrM1_model, jobid)
    TrM2_suffix  = NameParser.NameParser.model_suffix(TrM2_model, jobid)
    TrM3_suffix  = NameParser.NameParser.model_suffix(TrM3_model, jobid)
    TrM4_suffix  = NameParser.NameParser.model_suffix(TrM4_model, jobid)

    if use_abbreviation
        output_base = joinpath(
            location, 
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)"
        )
    else
        output_base = joinpath(
            location, 
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)"
        )
    end

    args_list = JobArgsEsther[]

    for label in labels
        for train in trains
            if use_abbreviation
                overall_name = "$(ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)_LBP_$(label)_TRP_$(train)"
            else
                overall_name = "$(ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)_LBP_$(label)_TRP_$(train)"
            end
            push!(
                args_list, 
                JobArgsEsther(
                    location, 
                    ensemble,
                    analysis_header,
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
                    label, train,
                    ns, nt, beta, kappa, nf,
                    ranseed, 
                    N_bs, 
                    blk_size, 
                    method,
                    bin_size,
                    IDX_shift, 
                    dump_X,
                    overall_name, 
                    abbreviation, 
                    use_abbreviation, 
                    output_base
                )
            )
        end
    end

    return args_list
end

"""
    generate_toml_dict(
        args::JobArgsEsther
    ) -> Dict

Generate a [`TOML`](https://toml.io/en/)-compatible dictionary from a [`JobArgsEsther`](@ref) instance.

# Arguments
- [`args::JobArgsEsther`](@ref EstherThreadsRunner.JobArgsEsther): Struct containing all job configuration parameters.

# Returns
- `Dict`: Nested dictionary structured for [`TOML`](https://toml.io/en/) serialization.
"""
function generate_toml_dict(
    args::JobArgsEsther
)::Dict
    return OrderedCollections.OrderedDict(
        "data" => OrderedCollections.OrderedDict(
            "location" => args.location,
            "ensemble" => args.ensemble,
            "analysis_header" => args.analysis_header,
            "TrM1_X" => args.TrM1_X,
            "TrM1_Y" => args.TrM1_Y,
            "TrM1_model" => args.TrM1_model,
            "TrM1_read_column_X" => args.TrM1_read_column_X,
            "TrM1_read_column_Y" => args.TrM1_read_column_Y,
            "TrM1_index_column" => args.TrM1_index_column,
            "TrM2_X" => args.TrM2_X,
            "TrM2_Y" => args.TrM2_Y,
            "TrM2_model" => args.TrM2_model,
            "TrM2_read_column_X" => args.TrM2_read_column_X,
            "TrM2_read_column_Y" => args.TrM2_read_column_Y,
            "TrM2_index_column" => args.TrM2_index_column,
            "TrM3_X" => args.TrM3_X,
            "TrM3_Y" => args.TrM3_Y,
            "TrM3_model" => args.TrM3_model,
            "TrM3_read_column_X" => args.TrM3_read_column_X,
            "TrM3_read_column_Y" => args.TrM3_read_column_Y,
            "TrM3_index_column" => args.TrM3_index_column,
            "TrM4_X" => args.TrM4_X,
            "TrM4_Y" => args.TrM4_Y,
            "TrM4_model" => args.TrM4_model,
            "TrM4_read_column_X" => args.TrM4_read_column_X,
            "TrM4_read_column_Y" => args.TrM4_read_column_Y,
            "TrM4_index_column" => args.TrM4_index_column,
            "LBP" => parse(Int, args.label),
            "TRP" => parse(Int, args.train),
            "use_abbreviation" => args.use_abbreviation
        ),
        "input_meta" => OrderedCollections.OrderedDict(
            "ns" => args.ns,
            "nt" => args.nt,
            "nf" => args.nf,
            "beta" => args.beta,
            "kappa" => args.kappa
        ),
        "bootstrap" => OrderedCollections.OrderedDict(
            "ranseed" => args.ranseed,
            "N_bs" => args.N_bs,
            "blk_size" => args.blk_size,
            "method" => args.method
        ),
        "jackknife" => OrderedCollections.OrderedDict(
            "bin_size" => args.bin_size
        ),
        "deborah" => OrderedCollections.OrderedDict(
            "IDX_shift" => args.IDX_shift,
            "dump_X" => args.dump_X
        ),
        "abbreviation" => OrderedCollections.OrderedDict(args.abbreviation)
    )
end

# ----------------------------------------------------------------------------
# Job Execution
# ----------------------------------------------------------------------------

"""
    run_one_job(
        args::JobArgsEsther, 
        output_dir::String, 
        log_path::String
    ) -> Nothing

Run a single [`Deborah.Esther`](@ref) job: generate [`TOML`](https://toml.io/en/) file, ensure ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` data exists, then run [`Deborah.Esther`](@ref) and log output.

# Arguments
- [`args::JobArgsEsther`](@ref EstherThreadsRunner.JobArgsEsther): Struct with full job configuration.
- `output_dir::String`: Directory path to write [`TOML`](https://toml.io/en/) file and results.
- `log_path::String`: File path for logging execution output and errors.

# Side Effects
- Saves [`TOML`](https://toml.io/en/) configuration to disk.
- Calls [`Deborah.DeborahEsther.EstherDependencyManager.ensure_TrM_exists`](@ref) and [`Deborah.Esther.EstherRunner.run_Esther`](@ref).
- Logs outputs and errors to the specified file.
"""
function run_one_job(
    args::JobArgsEsther, 
    output_dir::String, 
    log_path::String
)
    toml_dict = generate_toml_dict(args)
    mkpath(output_dir)
    toml_file = joinpath(output_dir, "config_Esther_$(args.overall_name).toml")
    ControllerCommon.save_toml_file(toml_dict, toml_file)

    open(log_path, "w") do io
        println(io, "Start: ", Dates.now())
        try
            jobid = "EstherChecker_LBP" * args.label * "_TRP" * args.train
            EstherDependencyManager.ensure_TrM_exists(toml_file, jobid)
            jobid = "Esther_LBP" * args.label * "_TRP" * args.train
            EstherRunner.run_Esther(toml_file, jobid)
        catch e
            println(io, "[ERROR] ", e)
            println(io, sprint(showerror, e))
        end
        println(io, "End: ", Dates.now())
    end
end

# ----------------------------------------------------------------------------
# Parallel Dispatcher
# ----------------------------------------------------------------------------

"""
    run_EstherThreads(
        toml_path::String
    ) -> Nothing

Parse [`Deborah.Esther`](@ref) configuration [`TOML`](https://toml.io/en/) and run all jobs in parallel, batched by number of threads.

# Arguments
- `toml_path::String`: Path to the master configuration [`TOML`](https://toml.io/en/) file.

# Side Effects
- Spawns tasks using [`Base.Threads.@spawn`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@spawn) to run multiple jobs in parallel.
- Logs each job's output in its own result directory.
"""
function run_EstherThreads(
    toml_path::String
)
    args_list = parse_config_EstherThreads(toml_path)
    batch_size = Threads.nthreads()
    N = length(args_list)
    for batch_start in 1:batch_size:N
        batch_end = min(batch_start + batch_size - 1, N)
        tasks = Task[]
        for i in batch_start:batch_end
            args = args_list[i]
            output_dir = joinpath(args.output_base, args.overall_name)
            log_path = joinpath(output_dir, "run_Esther_$(args.overall_name).log")
            push!(tasks, Threads.@spawn run_one_job(args, output_dir, log_path))
        end
        wait.(tasks)
    end
end

end  # module EstherThreadsRunner