# ============================================================================
# src/Elijah/MiriamThreadsWizardRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MiriamThreadsWizardRunner

import TOML
import REPL.TerminalMenus

"""
    run_MiriamThreadsWizard() -> Dict{String, Any}

Interactive terminal wizard for generating a multi-threaded [`TOML`](https://toml.io/en/) configuration file for [`Deborah.MiriamThreads`](@ref) across ensembles and label/train ratios.

This function prepares a full configuration suitable for multi-kappa, multi-ratio reweighting analysis  
with batch parallelization. It extends [`Deborah.Elijah.MiriamWizardRunner.run_MiriamWizard`](@ref) by allowing the user to specify a grid  
of label/train percentages and multiple `kappa` values for simultaneous ensemble processing.

The resulting configuration includes `[input_meta]`, `[data]`, `[solver]`, `[jackknife]`, `[bootstrap]`,  
`[trajectory]`, `[deborah]`, and optionally `[abbreviation]`. The derived ensemble names are constructed  
based on user-specified `ns`, `nt`, `beta`, and each `kappa` value.

# Workflow Overview
1. Prompt for lattice and solver parameters, including `kappa_list`.
2. Generate full ensemble names and collect I/O and model parameters.
3. Accept comma-separated `labels` and `trains` for threaded analysis.
4. Define solver tolerance, bootstrap, and interpolation settings.
5. Load or define an abbreviation dictionary if desired.
6. Save the configuration to `config_MiriamThreads.toml`.

# Returns
- A `Dict{String, Any}` containing the full configuration for [`Deborah.MiriamThreads`](@ref) runs.

# Side Effects
- Writes the configuration to a [`TOML`](https://toml.io/en/) file.
- Provides guided prompts to assist interactive configuration.
"""
function run_MiriamThreadsWizard()
    println("Starting MiriamThreads config wizard...\n")

    # -- [input_meta] section --
    ns = parse(Int, Base.prompt("ns (spatial size)", default = "8"))
    nt = parse(Int, Base.prompt("nt (temporal size)", default = "4"))
    nf = parse(Int, Base.prompt("nf (flavor count)", default = "4"))
    beta_str = Base.prompt("beta", default = "1.60")
    beta = parse(Float64, beta_str)
    csw = parse(Float64, Base.prompt("csw", default = "2.065"))

    kappa_list_str = Base.prompt("List of kappa values (comma-separated)", default = "0.13570,0.13575,0.13580,0.13585,0.13590")
    kappa_list = strip.(split(kappa_list_str, ","))

    input_meta_section = Dict(
        "ns" => ns,
        "nt" => nt,
        "nf" => nf,
        "beta" => beta,
        "csw" => csw,
        "kappa_list" => parse.(Float64, kappa_list),
    )

    multi_ensemble = "L$(ns)T$(nt)b$(beta_str)"
    ensembles = ["$(multi_ensemble)k$(replace(k, "0." => ""))" for k in kappa_list]

    # -- [data] section --
    location = Base.prompt("Data location directory", default = "./nf4_clover_wilson_finiteT")
    analysis_header = Base.prompt("Analysis header", default = "analysis")

    labels_str = Base.prompt("Labels (comma-separated)", default = "5,10,15,20,25,30,35,40,45,50")
    trains_str = Base.prompt("Trains (comma-separated)", default = "0,10,20,30,40,50,60,70,80,90,100")

    X_str = Base.prompt("Input feature files (comma-separated)", default = "pbp.dat")
    X = strip.(split(X_str, ","))
    read_column_X = parse.(Int, split(Base.prompt("Read columns for X (comma-separated)", default = "1"), ","))
    read_column_Y = parse(Int, Base.prompt("Read column for Y", default = "1"))
    index_column = parse(Int, Base.prompt("Index column", default = "3"))
    dump_original = Base.prompt("Dump original data? (true/false)", default = "false") == "true"
    use_abbreviation = Base.prompt("Use abbreviation map? (true/false)", default = "true") == "true"

    data_section = Dict(
        "labels" => strip.(split(labels_str, ",")),
        "trains" => strip.(split(trains_str, ",")),
        "location" => location,
        "multi_ensemble" => multi_ensemble,
        "ensembles" => ensembles,
        "analysis_header" => analysis_header,
        "dump_original" => dump_original,
        "use_abbreviation" => use_abbreviation,
    )

    targets = [
        ("TrM1", "pbp.dat", "Baseline"),
        ("TrM2", "trdinv2.dat", "LightGBM"),
        ("TrM3", "trdinv3.dat", "LightGBM"),
        ("TrM4", "trdinv4.dat", "LightGBM"),
    ]

    for (name, yfile, model) in targets
        data_section["$(name)_X"] = X
        data_section["$(name)_Y"] = yfile
        data_section["$(name)_model"] = model
        data_section["$(name)_read_column_X"] = read_column_X
        data_section["$(name)_read_column_Y"] = read_column_Y
        data_section["$(name)_index_column"] = index_column
    end

    # -- [solver] section --
    solver_section = Dict(
        "maxiter" => parse(Int, Base.prompt("Solver maxiter", default = "1000")),
        "eps" => parse(Float64, Base.prompt("Solver eps", default = "1e-13")),
    )

    # -- [jackknife] section --
    jackknife_section = Dict("bin_size" => parse(Int, Base.prompt("Jackknife bin size", default = "1")))

    # -- [bootstrap] section --
    bootstrap_section = Dict(
        "ranseed" => parse(Int, Base.prompt("Bootstrap random seed", default = "850528")),
        "N_bs" => parse(Int, Base.prompt("Number of bootstrap samples", default = "1000")),
        "blk_size" => parse(Int, Base.prompt("Bootstrap block size", default = "1")),
        "method" => Base.prompt("Bootstrap method", default = "nonoverlapping") == "nonoverlapping",
    )

    # -- [trajectory] section --
    trajectory_section = Dict("nkappaT" => parse(Int, Base.prompt("Number of kappa steps (nkappaT)", default = "100")))

    # -- [deborah] section --
    deborah_section = Dict(
        "IDX_shift" => parse(Int, Base.prompt("Index shift", default = "0")),
        "dump_X" => Base.prompt("Dump X? (true/false)", default = "false") == "true",
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
        "solver" => solver_section,
        "jackknife" => jackknife_section,
        "bootstrap" => bootstrap_section,
        "trajectory" => trajectory_section,
        "deborah" => deborah_section,
    )
    if use_abbreviation
        config["abbreviation"] = abbreviation_section
    end

    # -- Save file --
    filename = Base.prompt("Save filename", default = "config_MiriamThreads.toml")
    open(filename, "w") do io
        TOML.print(io, config)
    end

    println("\nConfiguration saved to $filename.\n")
    return config
end

end  # module MiriamThreadsWizardRunner