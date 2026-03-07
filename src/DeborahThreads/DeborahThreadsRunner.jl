# ============================================================================
# src/DeborahThreads/DeborahThreadsRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DeborahThreadsRunner

import ..TOML
import ..OrderedCollections
import ..Dates
import ..Base.Threads: @spawn, nthreads

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.ControllerCommon
import ..DeborahCore.DeborahRunner

"""
    struct JobArgsDeborah

Container struct for a single [`Deborah.DeborahCore`](@ref) job configuration.

# Fields
- `location::String`: Base path for input/output.
- `ensemble::String`: Ensemble name (e.g., `"L8T4b1.60k13570"`).
- `analysis_header::String`: Header string for organizing analysis subdirectories.
- `X::Vector{String}`: Input feature names.
- `Y::String`: Target observable name.
- `model::String`: ML model name (e.g., `"LightGBM"`).
- `read_column_X::Vector{Int}`: Vector of ``1``-based column indices specifying which column to read from each input feature file in `X`.
- `read_column_Y::Int`: ``1``-based column index specifying which column to read from the target observable file `Y`.
- `index_column::Int` : ``1``-based column index to read the configuration index from files  
                        (typically `1` if configuration index is in the first column).
- `label::String`: LBP label identifier (e.g., `"75"`).
- `train::String`: TRP train identifier (e.g., `"40"`).
- `ranseed::Int`: Random seed for bootstrap.
- `N_bs::Int`: Number of bootstrap samples.
- `blk_size::Int`: Bootstrap block size.
- `bin_size::Int`: Jackknife bin size.
- `IDX_shift::Int`: Index shift.
- `dump_X::Bool`: Whether to dump the `X` matrix.
- `overall_name::String`: Filename prefix used throughout output files.
- `abbreviation::Dict{String, String}`: Dictionary mapping observables to abbreviations.
- `use_abbreviation::Bool`: Whether to encode names using abbreviations.
- `output_base::String`: Directory path for outputs (not including filename prefix).
"""
struct JobArgsDeborah
    location::String
    ensemble::String
    analysis_header::String
    X::Vector{String}
    Y::String
    model::String
    read_column_X::Vector{Int}
    read_column_Y::Int
    index_column::Int
    label::String
    train::String
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
    parse_config_DeborahThreads(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Vector{JobArgsDeborah}

Parse a [`TOML`](https://toml.io/en/) config file and generate one [`DeborahThreadsRunner.JobArgsDeborah`](@ref) struct per label/train pair.
This function is used in [`Deborah.DeborahCore`](@ref) parallel execution to schedule multiple jobs.

# Arguments
- `toml_path::String`: Path to a [`TOML`](https://toml.io/en/) configuration file defining multiple `LBP`/`TRP` jobs.
- `jobid::Union{Nothing, String}` : Optional job ID string used for logging.

# Returns
- `Vector{JobArgsDeborah}`: A list of job argument structs, each ready to be passed into a [`Deborah.DeborahCore`](@ref) run.
"""
function parse_config_DeborahThreads(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)::Vector{JobArgsDeborah}
    
    JobLoggerTools.println_benji("Loading config from: $toml_path")

    cfg = TOML.parsefile(toml_path)
    labels           = cfg["data"]["labels"]
    trains           = cfg["data"]["trains"]
    location         = cfg["data"]["location"]
    ensemble         = cfg["data"]["ensemble"]
    analysis_header  = cfg["data"]["analysis_header"]
    X                = cfg["data"]["X"]
    Y                = cfg["data"]["Y"]
    model            = cfg["data"]["model"]
    read_column_X    = cfg["data"]["read_column_X"]
    read_column_Y    = cfg["data"]["read_column_Y"]
    index_column     = cfg["data"]["index_column"]
    ranseed          = cfg["bootstrap"]["ranseed"]
    N_bs             = cfg["bootstrap"]["N_bs"]
    blk_size         = cfg["bootstrap"]["blk_size"]
    method           = cfg["bootstrap"]["method"]
    bin_size         = cfg["jackknife"]["bin_size"]
    IDX_shift        = cfg["data"]["IDX_shift"]
    dump_X           = cfg["data"]["dump_X"]
    use_abbreviation = cfg["data"]["use_abbreviation"]
    raw_abbrev       = cfg["abbreviation"]

    abbreviation = StringTranscoder.parse_string_dict(raw_abbrev)
    XY_code      = StringTranscoder.input_encoder_abbrev_dict(X, Y, abbreviation)
    X_Y          = NameParser.make_X_Y(X, Y)
    suffix       = NameParser.model_suffix(model, jobid)

    output_base = if use_abbreviation
        joinpath(
            location,
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(XY_code)_$(suffix)"
        )
    else
        joinpath(
            location, 
            "$(analysis_header)_$(ensemble)", 
            "$(analysis_header)_$(ensemble)_$(X_Y)_$(suffix)"
        )
    end

    args_list = JobArgsDeborah[]
    
    for label in labels
        for train in trains
            overall_name = if use_abbreviation
                "$(ensemble)_$(XY_code)_$(suffix)_LBP_$(label)_TRP_$(train)"
            else
                "$(ensemble)_$(X_Y)_$(suffix)_LBP_$(label)_TRP_$(train)"
            end
            
            push!(
                args_list, 
                JobArgsDeborah(
                    location,
                    ensemble,
                    analysis_header,
                    X, Y, model,
                    read_column_X,
                    read_column_Y,
                    index_column, 
                    label, train,
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
        args::JobArgsDeborah
    ) -> Dict

Generate a [`TOML`](https://toml.io/en/)-compatible dictionary based on a [`DeborahThreadsRunner.JobArgsDeborah`](@ref) struct.  
This dictionary can be serialized to disk and used as a configuration file for [`Deborah.DeborahCore`](@ref) or [`Deborah.Esther`](@ref).

# Arguments
- [`args::JobArgsDeborah`](@ref Deborah.DeborahThreads.DeborahThreadsRunner.JobArgsDeborah): Struct containing all parameters for a single `LBP`/`TRP` job.

# Returns
- `Dict`: A [`TOML`](https://toml.io/en/)-structured dictionary, with properly nested `data`, `bootstrap`, `jackknife`, and `abbreviation` sections.
"""
function generate_toml_dict(
    args::JobArgsDeborah
)::Dict
    return OrderedCollections.OrderedDict(
        "data" => OrderedCollections.OrderedDict(
            "location" => args.location,
            "ensemble" => args.ensemble,
            "analysis_header" => args.analysis_header,
            "X" => args.X,
            "Y" => args.Y,
            "model" => args.model,
            "read_column_X" => args.read_column_X,
            "read_column_Y" => args.read_column_Y,
            "index_column" => args.index_column,
            "LBP" => parse(Int, args.label),
            "TRP" => parse(Int, args.train),
            "IDX_shift" => args.IDX_shift,
            "dump_X" => args.dump_X,
            "use_abbreviation" => args.use_abbreviation
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
        "abbreviation" => OrderedCollections.OrderedDict(args.abbreviation)
    )
end

# ----------------------------------------------------------------------------
# Job Execution
# ----------------------------------------------------------------------------

"""
    run_one_job(
        args::JobArgsDeborah, 
        output_dir::String, 
        log_path::String
    ) -> Nothing

Run a single [`Deborah.DeborahCore`](@ref) job based on the given arguments.  
Generates a [`TOML`](https://toml.io/en/) config, saves it, and runs [`Deborah.DeborahCore.DeborahRunner.run_Deborah()`](@ref) while logging start/end timestamps and errors.

# Arguments
- [`args::JobArgsDeborah`](@ref Deborah.DeborahThreads.DeborahThreadsRunner.JobArgsDeborah): Struct with job parameters.
- `output_dir::String`: Directory where outputs and config file will be saved.
- `log_path::String`: File path to save the run log.
"""
function run_one_job(
    args::JobArgsDeborah, 
    output_dir::String, 
    log_path::String
)
    toml_dict = generate_toml_dict(args)
    mkpath(output_dir)
    toml_file = joinpath(output_dir, "config_Deborah_$(args.overall_name).toml")
    ControllerCommon.save_toml_file(toml_dict, toml_file)

    open(log_path, "w") do io
        println(io, "Start: ", Dates.now())
        try
            jobid = "Deborah_LBP" * args.label * "_TRP" * args.train
            DeborahRunner.run_Deborah(toml_file, jobid)
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
    run_DeborahThreads(
        toml_path::String
    ) -> Nothing

Parse [`TOML`](https://toml.io/en/) config and run all [`Deborah.DeborahCore`](@ref) jobs in parallel,  
dispatching in batches according to the [number of threads](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS).

# Arguments
- `toml_path::String`: Path to the [`TOML`](https://toml.io/en/) configuration file containing job lists.

# Side Effects
- Spawns tasks using [`Base.Threads.@spawn`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.@spawn) to run multiple jobs in parallel.
- Logs each job's output in its own result directory.
"""
function run_DeborahThreads(
    toml_path::String
)
    args_list = parse_config_DeborahThreads(toml_path)
    batch_size = Threads.nthreads()
    N = length(args_list)
    for batch_start in 1:batch_size:N
        batch_end = min(batch_start + batch_size - 1, N)
        tasks = Task[]
        for i in batch_start:batch_end
            args = args_list[i]
            output_dir = joinpath(args.output_base, args.overall_name)
            log_path = joinpath(output_dir, "run_Deborah_$(args.overall_name).log")
            push!(tasks, Threads.@spawn run_one_job(args, output_dir, log_path))
        end
        wait.(tasks)
    end
end

end  # module DeborahThreadsRunner