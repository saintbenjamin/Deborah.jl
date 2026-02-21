# ============================================================================
# src/Elijah/MiriamWizardRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MiriamWizardRunner

import TOML
import REPL.TerminalMenus

"""
    run_MiriamWizard() -> Dict{String, Any}

Interactive terminal wizard for generating a [`TOML`](https://toml.io/en/) configuration file for [`Deborah.Miriam`](@ref) (multi-ensemble reweighting analysis).

This function prepares a complete configuration for [`Deborah.Miriam`](@ref), supporting multiple `kappa` values,  
ensemble tracking, interpolation-based reweighting, and bootstrap/jackknife analysis.
It sets up all relevant sections required by the Miriam pipeline, including `[input_meta]`, `[data]`,  
`[solver]`, `[bootstrap]`, `[jackknife]`, `[trajectory]`, `[deborah]`, and optionally `[abbreviation]`.

It supports up to four targets (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) with shared model features and allows  
the user to specify interpolation parameters such as `nkappaT`.

# Workflow Overview
1. Prompt for ensemble metadata (`ns`, `nt`, `nf`, `beta`, `kappa_list`, `csw`).
2. Derive ensemble names and prompt for I/O settings and model structure.
3. Configure solver tolerance and iteration controls.
4. Specify bootstrap, jackknife, and interpolation parameters.
5. Load or define an abbreviation dictionary if desired.
6. Save to `config_Miriam.toml`.

# Returns
- A `Dict{String, Any}` containing the full [`TOML`](https://toml.io/en/) configuration for [`Deborah.Miriam`](@ref).

# Side Effects
- Writes the configuration file to disk.
- Provides structured guidance for all required settings via [`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/).
"""
function run_MiriamWizard()
    println("Starting Miriam config wizard...\n")

    # -- [input_meta] section --
    ns     = parse(Int, Base.prompt("ns (spatial size)", default = "8"))
    nt     = parse(Int, Base.prompt("nt (temporal size)", default = "4"))
    nf     = parse(Int, Base.prompt("nf (flavor count)", default = "4"))
    beta_str = Base.prompt("beta", default = "1.60")
    beta = parse(Float64, beta_str)
    csw    = parse(Float64, Base.prompt("clover coefficient (csw)", default = "2.065"))

    println("Enter kappa values (comma-separated, e.g. 0.13570,0.13575,0.13580):")
    kappa_strs = strip.(split(Base.prompt("kappa list", default = "0.13570,0.13575,0.13580"), ','))
    kappa_vals = parse.(Float64, kappa_strs)

    input_meta_section = Dict(
        "ns" => ns,
        "nt" => nt,
        "nf" => nf,
        "beta" => beta,
        "csw" => csw,
        "kappa_list" => kappa_vals,
    )

    # -- derived: ensemble list --
    multi_ensemble = "L$(ns)T$(nt)b$(beta_str)"
    ensembles = [multi_ensemble * "k" * replace(k, "0." => "") for k in kappa_strs]

    # -- [data] section --
    location = Base.prompt("Data location directory", default = "./nf4_clover_wilson_finiteT")
    analysis_header = Base.prompt("Analysis header", default = "analysis")

    model = Base.prompt("Model name", default = "LightGBM")
    X_str = Base.prompt("Input feature files for all targets (comma-separated)", default = "plaq.dat,rect.dat")
    X = strip.(split(X_str, ','))

    read_column_X_str = Base.prompt("Read columns for X (comma-separated)", default = "1,1")
    read_column_X = parse.(Int, split(read_column_X_str, ','))
    read_column_Y = parse(Int, Base.prompt("Read column for Y", default = "1"))
    index_column = parse(Int, Base.prompt("Index column", default = "3"))

    LBP = parse(Int, Base.prompt("Label Bootstrap Percentage (LBP)", default = "30"))
    TRP = parse(Int, Base.prompt("Train Bootstrap Percentage (TRP)", default = "30"))
    dump_original = Base.prompt("Dump original data? (true/false)", default = "false") == "true"
    use_abbreviation = Base.prompt("Use abbreviation map? (true/false)", default = "true") == "true"

    # Y targets are fixed
    targets = [
        ("TrM1", "pbp.dat"),
        ("TrM2", "trdinv2.dat"),
        ("TrM3", "trdinv3.dat"),
        ("TrM4", "trdinv4.dat"),
    ]

    data_section = Dict(
        "location" => location,
        "multi_ensemble" => multi_ensemble,
        "ensembles" => ensembles,
        "analysis_header" => analysis_header,
        "LBP" => LBP,
        "TRP" => TRP,
        "dump_original" => dump_original,
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

    # -- [solver] section --
    maxiter = parse(Int, Base.prompt("Solver maxiter", default = "1000"))
    eps = parse(Float64, Base.prompt("Solver epsilon (eps)", default = "1e-13"))
    solver_section = Dict("maxiter" => maxiter, "eps" => eps)

    # -- [bootstrap] section --
    ranseed = parse(Int, Base.prompt("Bootstrap random seed", default = "850528"))
    N_bs = parse(Int, Base.prompt("Number of bootstrap samples", default = "1000"))
    blk_size = parse(Int, Base.prompt("Bootstrap block size", default = "1"))
    method = Base.prompt("Bootstrap method", default = "nonoverlapping") == "nonoverlapping"
    bootstrap_section = Dict("ranseed" => ranseed, "N_bs" => N_bs, "blk_size" => blk_size, "method" => method)

    # -- [jackknife] section --
    bin_size = parse(Int, Base.prompt("Jackknife bin size", default = "1"))
    jackknife_section = Dict("bin_size" => bin_size)

    # -- [trajectory] section --
    nkappaT = parse(Int, Base.prompt("Trajectory nkappaT", default = "10"))
    trajectory_section = Dict("nkappaT" => nkappaT)

    # -- [deborah] section --
    IDX_shift = parse(Int, Base.prompt("Index shift", default = "0"))
    dump_X = Base.prompt("Dump X to file? (true/false)", default = "false") == "true"
    deborah_section = Dict("IDX_shift" => IDX_shift, "dump_X" => dump_X)

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

    # -- build config --
    config = Dict(
        "input_meta" => input_meta_section,
        "data" => data_section,
        "solver" => solver_section,
        "bootstrap" => bootstrap_section,
        "jackknife" => jackknife_section,
        "trajectory" => trajectory_section,
        "deborah" => deborah_section,
    )
    if use_abbreviation
        config["abbreviation"] = abbreviation_section
    end

    filename = Base.prompt("Save filename", default = "config_Miriam.toml")
    open(filename, "w") do io
        TOML.print(io, config)
    end

    println("\nConfiguration saved to $filename.\n")
    return config
end

end  # module MiriamWizardRunner