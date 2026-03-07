# ============================================================================
# src/Elijah/DeborahWizardRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DeborahWizardRunner

import ..TOML
import ..REPL.TerminalMenus

"""
    run_DeborahWizard() -> Dict{String, Any}

Interactive terminal wizard to generate a [`TOML`](https://toml.io/en/) configuration file for [`Deborah.DeborahCore`](@ref).

This function guides the user through a series of prompts to collect all necessary fields  
for running a [`Deborah.DeborahCore`](@ref)-based machine learning estimation task. The generated configuration  
includes `[data]`, `[bootstrap]`, `[jackknife]`, and optionally `[abbreviation]` sections.

It supports multiple input features, adjustable column indexing, label/train ratio selection,  
bootstrap/jackknife settings, and an optional abbreviation map (either inline or via external [`TOML`](https://toml.io/en/)).

# Workflow Overview
1. Prompt user for data paths, input files, and read columns.
2. Define model metadata and ratio parameters (`LBP`/`TRP`).
3. Collect bootstrap and jackknife configuration.
4. Optionally load or manually define an abbreviation map.
5. Write all values to a `config_Deborah.toml` file.

# Returns
- A `Dict{String, Any}` representing the full [`TOML`](https://toml.io/en/) configuration.

# Side Effects
- Writes a [`TOML`](https://toml.io/en/) file to the specified location via user input.
- Displays progress and warnings in the [`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/).
"""
function run_DeborahWizard()
    println("Starting Deborah config wizard...\n")

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

    LBP = parse(Int, Base.prompt("Labeled set Percentage (LBP)", default = "30"))
    TRP = parse(Int, Base.prompt("Training set Percentage (TRP)", default = "30"))
    IDX_shift = parse(Int, Base.prompt("Index shift", default = "0"))

    dump_X = Base.prompt("Dump X to file? (true/false)", default = "false") == "false"
    use_abbreviation = Base.prompt("Use abbreviation map? (true/false)", default = "true") == "true"

    data_section = Dict(
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

                # If top-level has only one key and it's "abbreviation", extract its content
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
    filename = Base.prompt("Save filename", default = "config_Deborah.toml")
    open(filename, "w") do io
        TOML.print(io, config)
    end

    println("\nConfiguration saved to $filename.\n")
    return config
end

end  # module DeborahWizardRunner