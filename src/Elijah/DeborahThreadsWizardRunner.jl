# ============================================================================
# src/Elijah/DeborahThreadsWizardRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DeborahThreadsWizardRunner

import TOML
import REPL.TerminalMenus

"""
    run_DeborahThreadsWizard() -> Dict{String, Any}

Interactive terminal wizard for generating a multi-ratio threaded [`TOML`](https://toml.io/en/) configuration for [`Deborah.DeborahThreads`](@ref).

This function is part of the [`Deborah.Elijah.DeborahThreadsWizardRunner`](@ref) module and is designed to produce configuration files for  
multi-ratio, multi-threaded estimation workflows. It differs from [`Deborah.Elijah.DeborahWizardRunner.run_DeborahWizard`](@ref) by allowing  
the user to specify full label and train percentage lists `(labels, trains)` for batch execution.

The resulting configuration includes `[data]`, `[bootstrap]`, `[jackknife]`, and optional `[abbreviation]` sections,  
tailored for use in threaded runners like [`Deborah.DeborahThreads.DeborahThreadsRunner.run_DeborahThreads`](@ref).

# Workflow Overview
1. Prompt for data source, ensemble name, and observable files.
2. Collect model and column indexing info.
3. Accept comma-separated lists of label and train ratios.
4. Collect bootstrap/jackknife settings.
5. Optionally load or define an abbreviation map.
6. Save everything to a `config_DeborahThreads.toml` file.

# Returns
- A `Dict{String, Any}` representing the full [`TOML`](https://toml.io/en/) configuration.

# Side Effects
- Saves the configuration to a [`TOML`](https://toml.io/en/) file specified by the user.
- Displays progress and warnings in the [`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/) interface.
"""
function run_DeborahThreadsWizard()
    println("Starting DeborahThread config wizard...\n")

    # -- [data] section --
    location = Base.prompt("Data location directory", default = "./nf4_clover_wilson_finiteT")
    ensemble = Base.prompt("Ensemble name", default = "L8T4b1.60k13570")
    analysis_header = Base.prompt("Analysis header", default = "analysis")

    X_str = Base.prompt("Input feature files (comma-separated)", default = "plaq.dat,rect.dat")
    X = strip.(split(X_str, ','))
    Y = Base.prompt("Target file", default = "pbp.dat")
    model = Base.prompt("Model name", default = "LightGBM")

    read_column_X_str = Base.prompt("Read columns for X (comma-separated, ``1``-based)", default = "1,1")
    read_column_X = parse.(Int, split(read_column_X_str, ','))
    read_column_Y = parse(Int, Base.prompt("Read column for Y", default = "1"))
    index_column = parse(Int, Base.prompt("Index column", default = "3"))

    labels_str = Base.prompt("Label percentage list (comma-separated)", default = "5,10,15,20,25,30,35,40,45,50")
    labels = strip.(split(labels_str, ','))

    trains_str = Base.prompt("Train percentage list (comma-separated)", default = "10,20,30,40,50,60,70,80,90")
    trains = strip.(split(trains_str, ','))

    IDX_shift = parse(Int, Base.prompt("Index shift", default = "0"))
    dump_X = Base.prompt("Dump X to file? (true/false)", default = "false") == "false"
    use_abbreviation = Base.prompt("Use abbreviation map? (true/false)", default = "true") == "true"

    data_section = Dict(
        "labels" => labels,
        "trains" => trains,
        "location" => location,
        "ensemble" => ensemble,
        "analysis_header" => analysis_header,
        "X" => X,
        "Y" => Y,
        "model" => model,
        "read_column_X" => read_column_X,
        "read_column_Y" => read_column_Y,
        "index_column" => index_column,
        "IDX_shift" => IDX_shift,
        "dump_X" => dump_X,
        "use_abbreviation" => use_abbreviation,
    )

    # -- [bootstrap] section --
    ranseed = parse(Int, Base.prompt("Bootstrap random seed", default = "850528"))
    N_bs = parse(Int, Base.prompt("Number of bootstrap samples", default = "1000"))
    blk_size = parse(Int, Base.prompt("Bootstrap block size", default = "1"))
    method = Base.prompt("Bootstrap method", default = "nonoverlapping") == "nonoverlapping"
    bootstrap_section = Dict(
        "ranseed" => ranseed,
        "N_bs" => N_bs,
        "blk_size" => blk_size,
        "method" => method,
    )

    # -- [jackknife] section --
    bin_size = parse(Int, Base.prompt("Jackknife bin size", default = "1"))
    jackknife_section = Dict("bin_size" => bin_size)

    # -- [abbreviation] section (optional) --
    abbreviation_section = Dict{String, String}()
    if use_abbreviation
        use_file = Base.prompt("Load abbreviation from file? (true/false)", default = "true") == "true"
        
        if use_file
            abbrev_file = Base.prompt("Path to abbreviation TOML file", default = "sample/abbreviation.toml")
            if isfile(abbrev_file)
                parsed = TOML.parsefile(abbrev_file)
                if haskey(parsed, "abbreviation") && isa(parsed["abbreviation"], Dict)
                    parsed = parsed["abbreviation"]
                end
                if all(k -> isa(k, String) && isa(parsed[k], String), keys(parsed))
                    abbreviation_section = parsed
                    println("Loaded abbreviation from $abbrev_file")
                else
                    println("Invalid format in $abbrev_file — skipping abbreviation section.")
                end
            else
                println("File not found: $abbrev_file — skipping abbreviation section.")
            end
        else
            println("Enter abbreviation pairs (type DONE when finished):")
            while true
                raw = Base.prompt("e.g. pbp.dat = TrM1", default = "")
                if strip(raw) == "DONE"
                    break
                elseif occursin('=', raw)
                    k, v = split(raw, '=')
                    abbreviation_section[strip(k)] = strip(v)
                end
            end
        end
    end

    # -- Build final config --
    config = Dict(
        "data" => data_section,
        "bootstrap" => bootstrap_section,
        "jackknife" => jackknife_section,
    )
    if use_abbreviation
        config["abbreviation"] = abbreviation_section
    end

    # -- Save to file --
    filename = Base.prompt("Save filename", default = "config_DeborahThreads.toml")
    open(filename, "w") do io
        TOML.print(io, config)
    end

    println("\nConfiguration saved to $filename.\n")
    return config
end

end  # module DeborahThreadsWizardRunner