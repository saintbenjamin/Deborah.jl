# ============================================================================
# src/Elijah/EstherWizardRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module EstherWizardRunner

import TOML
import REPL.TerminalMenus

"""
    run_EstherWizard() -> Dict{String, Any}

Interactive terminal wizard for generating a [`TOML`](https://toml.io/en/) configuration file for [`Deborah.Esther`](@ref) (multi-observable bootstrap runner).

This function builds a full configuration file tailored for [`Deborah.Esther`](@ref)'s use case, which involves  
estimating four observables (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) using fixed input/output structures, shared models,  
and consistent indexing and bootstrap settings.

It automatically derives the `ensemble` name from the physical parameters provided in `[input_meta]`,  
and sets up all necessary sections: `[input_meta]`, `[data]`, `[bootstrap]`, `[jackknife]`, `[deborah]`,  
and optionally `[abbreviation]`.

# Workflow Overview
1. Prompt for lattice metadata: `ns`, `nt`, `nf`, `beta`, `kappa`.
2. Derive ensemble name and prompt for file paths and model settings.
3. Automatically configure all four ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` observables using shared parameters.
4. Add bootstrap/jackknife settings and [`Deborah.DeborahCore`](@ref)-specific controls.
5. Optionally load or define an abbreviation map.
6. Save to `config_Esther.toml`.

# Returns
- A `Dict{String, Any}` representing the complete [`TOML`](https://toml.io/en/) configuration for [`Deborah.Esther`](@ref).

# Side Effects
- Writes the configuration to a [`TOML`](https://toml.io/en/) file as chosen by the user.
- Displays status updates and warnings during input.
"""
function run_EstherWizard()
    println("🔮 Starting Esther config wizard...\n")

    # -- [input_meta] section --
    ns     = parse(Int, Base.prompt("ns (spatial size)", default = "8"))
    nt     = parse(Int, Base.prompt("nt (temporal size)", default = "4"))
    nf     = parse(Int, Base.prompt("nf (flavor count)", default = "4"))
    beta_str  = Base.prompt("beta", default = "1.60")
    beta_val  = parse(Float64, beta_str)
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
    ensemble_kappa = replace(kappa_str, "0." => "")  # "0.13570" → "13570"
    ensemble = "L$(ns)T$(nt)b$(beta_str)k$(ensemble_kappa)"

    # -- [data] section (fixed TrM1–TrM4) --
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
        "ensemble" => ensemble,
        "analysis_header" => analysis_header,
        "LBP" => LBP,
        "TRP" => TRP,
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
    filename = Base.prompt("Save filename", default = "config_Esther.toml")
    open(filename, "w") do io
        TOML.print(io, config)
    end

    println("\nConfiguration saved to $filename.\n")
    return config
end

end  # module EstherWizardRunner