# ============================================================================
# src/MiriamThreads/MiriamThreadsRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MiriamThreadsRunner

import ..TOML
import ..OrderedCollections
import ..Dates
import ..Base.Threads: @spawn, nthreads

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.ControllerCommon
import ..Miriam.MiriamRunner
import ..DeborahEsther.EstherDependencyManager
import ..DeborahEstherMiriam.MiriamDependencyManager
import ..DeborahEstherMiriam.MiriamExistenceManager

"""
    struct JobArgsMiriam

Container struct holding all job parameters required for [`Deborah.Miriam`](@ref) parallel execution.

# Fields
- `location::String`             : Base directory containing data
- `multi_ensemble::String`       : Group name for multiple ensembles
- `ensembles::Vector{String}`    : List of ensemble subdirectories

## Training Model Inputs (TrM1–TrM4)
- `TrM1_X`, `TrM1_Y`, `TrM1_model` : Inputs, target, model tag for ``\\text{Tr} \\, M^{-1}``
- `TrM2_X`, `TrM2_Y`, `TrM2_model` : Inputs, target, model tag for ``\\text{Tr} \\, M^{-2}``
- `TrM3_X`, `TrM3_Y`, `TrM3_model` : Inputs, target, model tag for ``\\text{Tr} \\, M^{-3}``
- `TrM4_X`, `TrM4_Y`, `TrM4_model` : Inputs, target, model tag for ``\\text{Tr} \\, M^{-4}``
- `TrM{1-4}_read_column_X::Vector{Int}` : ``1``-based column index to read the numerical value from each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` file.
- `TrM{1-4}_read_column_Y::Int` : ``1``-based column index to read the numerical value from each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` file.
- `TrM{1-4}_index_column::Int` : ``1``-based column index to read the configuration index from each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` file.

## Job Sweep Parameters
- `label::String`                : `LBP` identifier
- `train::String`                : `TRP` identifier
- `dump_original::Bool`         : Whether to dump original raw ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` files for reference

## Input Meta
- `ns::Int`                      : Spatial lattice size
- `nt::Int`                      : Temporal lattice size
- `nf::Int`                      : Number of quark flavors ``N_{\\text{f}}``
- `beta::Float64`               : Gauge coupling ``\\beta``
- `csw::Float64`                : Clover coefficient ``c_{\\text{SW}}``
- `kappa_list::Vector{Float64}` : ``\\kappa`` values used for the multi-ensemble reweighting

## Solver Parameters
- `maxiter::Int`                : Maximum solver iterations
- `eps::Float64`                : Solver precision

## Resampling Parameters
- `bin_size::Int`               : Jackknife bin size
- `ranseed::Int`                : Random seed for bootstrap
- `N_bs::Int`                   : Number of bootstrap resamples
- `blk_size::Int`               : Bootstrap block size
- `method::String`   : Block-bootstrap scheme (case-sensitive):
    - `"nonoverlapping"` — Nonoverlapping Block Bootstrap (NBB; resample disjoint blocks).
    - `"moving"`        — Moving Block Bootstrap (MBB; resample sliding windows).
    - `"circular"`      — Circular Block Bootstrap (CBB; sliding windows with wrap-around).

## Trajectory Meta
- `nkappaT::Int`                : Number of total ``\\kappa`` trajectories

## [`Deborah.DeborahCore`](@ref) Metadata
- `analysis_header::String`     : Prefix for analysis directories
- `IDX_shift::Int`              : Index shift for additional analysis
- `dump_X::Bool`                : Dump feature matrices?

## Naming
- `overall_name::String`        : Unique name for each job instance
- `abbreviation::Dict{String,String}` : Abbreviation map for input features
- `use_abbreviation::Bool`      : Use abbreviations when naming outputs?
- `output_base::String`         : Root directory for output data
"""
struct JobArgsMiriam
    location::String
    multi_ensemble::String
    ensembles::Vector{String}
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
    dump_original::Bool
    ns::Int
    nt::Int
    nf::Int
    beta::Float64
    csw::Float64
    kappa_list::Vector{Float64}
    maxiter::Int
    eps::Float64
    bin_size::Int
    ranseed::Int
    N_bs::Int
    blk_size::Int
    method::String
    nkappaT::Int
    analysis_header::String
    IDX_shift::Int
    dump_X::Bool
    overall_name::String
    abbreviation::Dict{String,String}
    use_abbreviation::Bool
    output_base::String
end

"""
    parse_config_MiriamThreads(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Vector{JobArgsMiriam}

Parse a configuration [`TOML`](https://toml.io/en/) file and construct a list of [`JobArgsMiriam`](@ref) structs
to represent all job combinations of `LBP`/`TRP` for the [`Deborah.Miriam`](@ref) analysis.

# Arguments
- `toml_path::String`: Path to the configuration [`TOML`](https://toml.io/en/) file.
- `jobid::Union{Nothing, String}` : Optional job ID string used for logging.

# Returns
- `Vector{JobArgsMiriam}`: A list of job argument structs, each corresponding to a distinct label/train combination.

# Notes
- This function extracts and resolves ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` inputs, solver parameters, bootstrap/jackknife configs, and meta info.
- It builds the `overall_name` for each job using either full names or abbreviations.
- The output directory base path is constructed using `analysis_header`, `multi_ensemble`, and ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``  details.
"""
function parse_config_MiriamThreads(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)::Vector{JobArgsMiriam}

    JobLoggerTools.println_benji("Loading config from: $toml_path")
    
    cfg = TOML.parsefile(toml_path)

    labels             = cfg["data"]["labels"]
    trains             = cfg["data"]["trains"]
    location           = cfg["data"]["location"]
    multi_ensemble     = cfg["data"]["multi_ensemble"]
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
    dump_original      = cfg["data"]["dump_original"]
    use_abbreviation   = cfg["data"]["use_abbreviation"]

    ns                 = cfg["input_meta"]["ns"]
    nt                 = cfg["input_meta"]["nt"]
    nf                 = cfg["input_meta"]["nf"]
    beta               = cfg["input_meta"]["beta"]
    csw                = cfg["input_meta"]["csw"]
    kappa_list         = cfg["input_meta"]["kappa_list"]

    maxiter            = cfg["solver"]["maxiter"]
    eps                = cfg["solver"]["eps"]

    bin_size           = cfg["jackknife"]["bin_size"]
    ranseed            = cfg["bootstrap"]["ranseed"]
    N_bs               = cfg["bootstrap"]["N_bs"]
    blk_size           = cfg["bootstrap"]["blk_size"]
    method            = cfg["bootstrap"]["method"]

    nkappaT            = cfg["trajectory"]["nkappaT"]

    IDX_shift          = cfg["deborah"]["IDX_shift"]
    dump_X             = cfg["deborah"]["dump_X"]

    raw_abbrev         = cfg["abbreviation"]

    abbreviation = StringTranscoder.StringTranscoder.parse_string_dict(raw_abbrev)

    TrM1_code = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM1_X, TrM1_Y, abbreviation)
    TrM2_code = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM2_X, TrM2_Y, abbreviation)
    TrM3_code = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM3_X, TrM3_Y, abbreviation)
    TrM4_code = StringTranscoder.StringTranscoder.input_encoder_abbrev_dict(TrM4_X, TrM4_Y, abbreviation)

    TrM1_X_Y = NameParser.NameParser.make_X_Y(TrM1_X, TrM1_Y)
    TrM2_X_Y = NameParser.NameParser.make_X_Y(TrM2_X, TrM2_Y)
    TrM3_X_Y = NameParser.NameParser.make_X_Y(TrM3_X, TrM3_Y)
    TrM4_X_Y = NameParser.NameParser.make_X_Y(TrM4_X, TrM4_Y)

    TrM1_suffix = NameParser.NameParser.model_suffix(TrM1_model, jobid)
    TrM2_suffix = NameParser.NameParser.model_suffix(TrM2_model, jobid)
    TrM3_suffix = NameParser.NameParser.model_suffix(TrM3_model, jobid)
    TrM4_suffix = NameParser.NameParser.model_suffix(TrM4_model, jobid)

    if use_abbreviation
        output_base = joinpath(
            location,
            "$(analysis_header)_$(multi_ensemble)",
            "$(analysis_header)_$(multi_ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)"
        )
    else
        output_base = joinpath(
            location,
            "$(analysis_header)_$(multi_ensemble)",
            "$(analysis_header)_$(multi_ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)"
        )
    end

    args_list = JobArgsMiriam[]

    for label in labels
        for train in trains
            if use_abbreviation
                overall_name = "$(multi_ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)_LBP_$(label)_TRP_$(train)"
            else
                overall_name = "$(multi_ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)_LBP_$(label)_TRP_$(train)"
            end

            push!(
                args_list,
                JobArgsMiriam(
                    location, 
                    multi_ensemble, 
                    ensembles,
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
                    dump_original,
                    ns, nt, nf, beta, csw,
                    kappa_list,
                    maxiter, eps,
                    bin_size, 
                    ranseed, 
                    N_bs, 
                    blk_size,
                    method,
                    nkappaT,
                    analysis_header, 
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
        args::JobArgsMiriam
    ) -> Dict

Convert a [`JobArgsMiriam`](@ref) struct into a [`TOML`](https://toml.io/en/)-compatible nested `OrderedCollections.OrderedDict`.

The returned dictionary mirrors the structure of the config [`TOML`](https://toml.io/en/) file,
grouped into sections such as `data`, `input_meta`, `solver`, etc.

# Arguments
- [`args::JobArgsMiriam`](@ref JobArgsMiriam): The full job parameter struct to be serialized.

# Returns
- `Dict`: A [`TOML`](https://toml.io/en/)-ready nested dictionary ([`OrderedCollections.OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/stable/#OrderedDicts)) representing the job configuration.

# [`TOML`](https://toml.io/en/) Sections
- `data`         : Ensemble paths, model inputs/targets, flags
- `input_meta`   : Lattice geometry and coupling constants
- `solver`       : Maximum iteration and precision
- `jackknife`    : Bin size for jackknife resampling
- `bootstrap`    : Random seed, number of resamples, block size
- `trajectory`   : Number of ``\\kappa``-trajectories
- `deborah`      : [`Deborah.DeborahCore`](@ref)-specific metadata
- `abbreviation` : Feature name abbreviation map
"""
function generate_toml_dict(
    args::JobArgsMiriam
)::Dict
    return OrderedCollections.OrderedDict(
        "data" => OrderedCollections.OrderedDict(
            "location" => args.location,
            "multi_ensemble" => args.multi_ensemble,
            "ensembles" => args.ensembles,
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
            "dump_original" => args.dump_original,
            "use_abbreviation" => args.use_abbreviation
        ),
        "input_meta" => OrderedCollections.OrderedDict(
            "ns" => args.ns,
            "nt" => args.nt,
            "nf" => args.nf,
            "beta" => args.beta,
            "csw" => args.csw,
            "kappa_list" => args.kappa_list
        ),
        "solver" => OrderedCollections.OrderedDict(
            "maxiter" => args.maxiter,
            "eps" => args.eps
        ),
        "jackknife" => OrderedCollections.OrderedDict(
            "bin_size" => args.bin_size
        ),
        "bootstrap" => OrderedCollections.OrderedDict(
            "ranseed" => args.ranseed,
            "N_bs" => args.N_bs,
            "blk_size" => args.blk_size,
            "method" => args.method
        ),
        "trajectory" => OrderedCollections.OrderedDict(
            "nkappaT" => args.nkappaT
        ),
        "deborah" => OrderedCollections.OrderedDict(
            "IDX_shift" => args.IDX_shift,
            "dump_X" => args.dump_X
        ),
        "abbreviation" => OrderedCollections.OrderedDict(args.abbreviation)
    )
end

"""
    run_one_job(
        args::JobArgsMiriam, 
        output_dir::String, 
        log_path::String
    ) -> Nothing

Run a single full [`Deborah.Miriam`](@ref) job: generate [`TOML`](https://toml.io/en/) config, run ensemble checker, and launch main computation.

# Arguments
- [`args::JobArgsMiriam`](@ref JobArgsMiriam): Struct containing all parameters for a [`Deborah.Miriam`](@ref) job.
- `output_dir::String`: Target output directory where logs and config will be stored.
- `log_path::String`: Path to write job-specific logs.

# Behavior
- Generates a [`TOML`](https://toml.io/en/) config file.
- Checks ensemble availability via [`Deborah.DeborahEstherMiriam.MiriamDependencyManager.ensure_ensemble_exists`](@ref).
- Runs the main [`Deborah.Miriam.MiriamRunner.run_Miriam`](@ref) pipeline using the given configuration.

# Returns
- `Nothing` (side effects: file I/O and computation).
"""
function run_one_job(
    args::JobArgsMiriam, 
    output_dir::String, 
    log_path::String
)
    toml_dict = generate_toml_dict(args)
    mkpath(output_dir)
    toml_file = joinpath(output_dir, "config_Miriam_$(args.overall_name).toml")
    ControllerCommon.save_toml_file(toml_dict, toml_file)

    open(log_path, "w") do io
        println(io, "Start: ", Dates.now())
        try
            jobid = "MiriamChecker_LBP" * args.label * "_TRP" * args.train
            MiriamDependencyManager.ensure_ensemble_exists(toml_file, jobid)
            jobid = "Miriam_LBP" * args.label * "_TRP" * args.train
            MiriamRunner.run_Miriam(toml_file, jobid)
        catch e
            println(io, "[ERROR] ", e)
            println(io, sprint(showerror, e))
        end
        println(io, "End: ", Dates.now())
    end
end

"""
    run_MiriamThreads(
        toml_path::String
    ) -> Nothing

Run all [`Deborah.Miriam`](@ref) jobs defined in the given [`TOML`](https://toml.io/en/) config file using multithreading.

# Arguments
- `toml_path::String`: Path to the batch config file containing multiple jobs.

# Behavior
- Parses all jobs via [`parse_config_MiriamThreads`](@ref).
- Spawns tasks in batches of [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads) to run [`run_one_job`](@ref) in parallel.

# Returns
- `Nothing` (side effects: runs all jobs and generates logs/configs).
"""
function run_MiriamThreads(
    toml_path::String
)
    args_list = parse_config_MiriamThreads(toml_path)
    batch_size = Threads.nthreads()
    N = length(args_list)
    for batch_start in 1:batch_size:N
        batch_end = min(batch_start + batch_size - 1, N)
        tasks = Task[]
        for i in batch_start:batch_end
            args = args_list[i]
            output_dir = joinpath(args.output_base, args.overall_name)
            log_path = joinpath(output_dir, "run_Miriam_$(args.overall_name).log")
            push!(tasks, Threads.@spawn run_one_job(args, output_dir, log_path))
        end
        wait.(tasks)
    end
end

"""
    run_one_check_job(
        args::JobArgsMiriam, 
        output_dir::String, 
        log_path::String
    ) -> Nothing

Run a single check-only job to verify multi-ensemble reweighting result existence.

# Arguments
- [`args::JobArgsMiriam`](@ref JobArgsMiriam): Job parameter set.
- `output_dir::String`: Directory to save the generated [`TOML`](https://toml.io/en/) config.
- `log_path::String`: Log file for result check.

# Behavior
- Writes a config file for the job.
- Calls [`Deborah.DeborahEstherMiriam.MiriamExistenceManager.ensure_multi_ensemble_exists`](@ref) to verify expected results.

# Returns
- `Nothing` (side effect: logs and possible error messages).
"""
function run_one_check_job(
    args::JobArgsMiriam, 
    output_dir::String, 
    log_path::String
)
    toml_dict = generate_toml_dict(args)
    mkpath(output_dir)
    toml_file = joinpath(output_dir, "config_Miriam_$(args.overall_name).toml")
    ControllerCommon.save_toml_file(toml_dict, toml_file)

    open(log_path, "w") do io
        println(io, "Start: ", Dates.now())
        try
            jobid = "MiriamChecker_LBP" * args.label * "_TRP" * args.train
            MiriamExistenceManager.ensure_multi_ensemble_exists
(toml_file, jobid)
        catch e
            println(io, "[ERROR] ", e)
            println(io, sprint(showerror, e))
        end
        println(io, "End: ", Dates.now())
    end
end

"""
    run_MiriamThreadsCheck(
        toml_path::String
    ) -> Nothing

Check-only mode for [`Deborah.Miriam`](@ref): validate that all expected multi-ensemble reweighting results exist.

# Arguments
- `toml_path::String`: Path to the batch config file.

# Behavior
- Parses all jobs using [`parse_config_MiriamThreads`](@ref).
- Verifies result presence using [`Deborah.DeborahEstherMiriam.MiriamExistenceManager.ensure_multi_ensemble_exists`](@ref).
- Uses threading to batch check jobs in parallel.

# Returns
- `Nothing` (side effects: logs for each check job).
"""
function run_MiriamThreadsCheck(
    toml_path::String
)
    args_list = parse_config_MiriamThreads(toml_path)
    batch_size = Threads.nthreads()
    N = length(args_list)
    for batch_start in 1:batch_size:N
        batch_end = min(batch_start + batch_size - 1, N)
        tasks = Task[]
        for i in batch_start:batch_end
            args = args_list[i]
            output_dir = joinpath(args.output_base, args.overall_name)
            log_path = joinpath(output_dir, "run_Miriam_$(args.overall_name).log")
            push!(tasks, Threads.@spawn run_one_check_job(args, output_dir, log_path))
        end
        wait.(tasks)
    end
end

end  # module MiriamThreadsRunner