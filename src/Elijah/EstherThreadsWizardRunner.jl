# ============================================================================
# src/Elijah/EstherThreadsWizardRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module EstherThreadsWizardRunner

import ..TOML
import ..REPL.TerminalMenus

"""
    run_EstherThreadsWizard() -> Dict{String, Any}

Interactive terminal wizard for generating a threaded [`TOML`](https://toml.io/en/) configuration for [`Deborah.EstherThreads`](@ref) across multiple label/train ratios.

This function is part of the [`Deborah.Elijah.EstherThreadsWizardRunner`](@ref) module and prepares a full configuration for running  
multi-observable estimation (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) across a grid of labeled/training set ratios,  
suitable for parallel execution via [`Deborah.EstherThreads.EstherThreadsRunner.run_EstherThreads`](@ref) or similar tools.

The resulting config includes `[input_meta]`, `[data]`, `[bootstrap]`, `[jackknife]`, `[deborah]`,  
and optionally `[abbreviation]`, with the ensemble name derived from user-specified `ns`, `nt`, `beta`, and `kappa`.

# Workflow Overview
1. Prompt for lattice metadata and observable parameters.
2. Accept full lists of label and train percentages.
3. Apply shared model and input settings to all four ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` targets.
4. Collect `[bootstrap]`/`[jackknife]`/`[deborah]` options.
5. Load or define an abbreviation dictionary if desired.
6. Save to `config_EstherThreads.toml`.

# Returns
- A `Dict{String, Any}` representing the complete configuration for threaded [`Deborah.EstherThreads`](@ref) runs.

# Side Effects
- Writes a [`TOML`](https://toml.io/en/) file to the specified location.
- Interactively gathers all required input via terminal prompts.
"""
function run_EstherThreadsWizard()
    println("Starting EstherThreads config wizard...\n")

    # -- [input_meta] section --
    ns = parse(Int, Base.prompt("ns (spatial size)", default = "8"))
    nt = parse(Int, Base.prompt("nt (temporal size)", default = "4"))
    nf = parse(Int, Base.prompt("nf (flavor count)", default = "4"))

    beta_str = Base.prompt("beta", default = "1.60")
    beta_val = parse(Float64, beta_str)

    kappa_str = Base.prompt("kappa", default = "0.13570")
    kappa_val = parse(Float64, kappa_str)

    input_meta_section = Dict(
        "ns" => ns,
        "nt" => nt,
        "nf" => nf,
        "beta" => beta_val,
        "kappa" => kappa_val,
    )

    # -- derived: ensemble name --
    ensemble_kappa = replace(kappa_str, "0." => "")  # e.g., "0.13570" → "13570"
    ensemble = "L$(ns)T$(nt)b$(beta_str)k$(ensemble_kappa)"

    # -- [data] section --
    location = Base.prompt("Data location directory", default = "./nf4_clover_wilson_finiteT")
    analysis_header = Base.prompt("Analysis header", default = "analysis")

    labels_str = Base.prompt("Label percentage list (comma-separated)", default = "5,10,15,20,25,30,35,40,45,50")
    labels = strip.(split(labels_str, ','))

    trains_str = Base.prompt("Train percentage list (comma-separated)", default = "10,20,30,40,50,60,70,80,90")
    trains = strip.(split(trains_str, ','))

    model = Base.prompt("Model name", default = "LightGBM")
    X_str = Base.prompt("Input feature files for all targets (comma-separated)", default = "plaq.dat,rect.dat")
    X = strip.(split(X_str, ','))

    read_column_X_str = Base.prompt("Read columns for X (comma-separated)", default = "1,1")
    read_column_X = parse.(Int, split(read_column_X_str, ','))
    read_column_Y = parse(Int, Base.prompt("Read column for Y", default = "1"))
    index_column = parse(Int, Base.prompt("Index column", default = "3"))

    use_abbreviation = Base.prompt("Use abbreviation map? (true/false)", default = "true") == "true"

    targets = [
        ("TrM1", "pbp.dat"),
        ("TrM2", "trdinv2.dat"),
        ("TrM3", "trdinv3.dat"),
        ("TrM4", "trdinv4.dat"),
    ]

    data_section = Dict(
        "labels" => labels,
        "trains" => trains,
        "location" => location,
        "ensemble" => ensemble,
        "analysis_header" => analysis_header,
        "use_abbreviation" => use_abbreviation,
    )

    for (name, yfile) in targets
        data_section["$(name)_X"] = X
        data_section["$(name)_Y"] = yfile
        data_section["$(name)_model"] = model
        data_section["$(name)_read_column_X"] = read_column_X
        data_section["$(name)_read_column_Y"] = read_column_Y
        data_section["$(name)_index_column"] = index_column
    end

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

    # -- [deborah] section --
    IDX_shift = parse(Int, Base.prompt("Index shift", default = "0"))
    dump_X = Base.prompt("Dump X to file? (true/false)", default = "false") == "false"
    deborah_section = Dict(
        "IDX_shift" => IDX_shift,
        "dump_X" => dump_X,
    )

    # -- [abbreviation] section --
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

    # -- Build full config --
    config = Dict(
        "input_meta" => input_meta_section,
        "data" => data_section,
        "bootstrap" => bootstrap_section,
        "jackknife" => jackknife_section,
        "deborah" => deborah_section,
    )
    if use_abbreviation
        config["abbreviation"] = abbreviation_section
    end

    # -- Save file --
    filename = Base.prompt("Save filename", default = "config_EstherThreads.toml")
    open(filename, "w") do io
        TOML.print(io, config)
    end

    println("\nConfiguration saved to $filename.\n")
    return config
end

end  # module EstherThreadsWizardRunner