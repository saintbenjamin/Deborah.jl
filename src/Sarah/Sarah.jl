# ============================================================================
# src/Sarah/Sarah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Sarah

# `Deborah.Sarah` — Common utilities for configuration, logging, resampling, I/O, and naming.

`Deborah.Sarah` is the utility toolkit shared across the `Deborah.jl` ecosystem.
It offers job-aware logging, [`TOML`](https://toml.io/en/) helpers, string/name encoding, deterministic
seeding, dataset partitioning and meta extraction, data loading, bootstrap/jackknife resampling, and summary formatting/collection. These utilities keep
the higher-level modules small, consistent, and reproducible.

# Scope & Responsibilities
- **Logging & diagnostics**: structured println/warn, timing, and job-scoped messages ([`Deborah.Sarah.JobLoggerTools`](@ref)).
- **[`TOML`](https://toml.io/en/) helpers**: append well-formed sections to existing [`TOML`](https://toml.io/en/) files ([`TOMLLogger.append_section_to_toml`](@ref)). 
- **Naming & encoding**: derive stable names/paths ([`Deborah.Sarah.NameParser`](@ref)) and map long tokens ``\\Leftrightarrow`` abbreviations/IDs ([`Deborah.Sarah.StringTranscoder`](@ref)).
- **Seeding**: centralized RNG seeding mainly used by bootstrap resampling for reproducible runs ([`Deborah.Sarah.SeedManager`](@ref)).
- **Dataset partitioning**: compute `LB`/`TR`/`BC`/`UL` splits and index ranges ([`Deborah.Sarah.DatasetPartitioner`](@ref)).
- **`XY` metadata & extraction**: read columns, reshape per-(conf,src), and derive configuration indices ([`XYInfoGenerator.gen_conf_from_Y`](@ref), [`XYInfoGenerator.gen_X_info`](@ref)).
- **Data loading**: robust delimited-file loaders with fallbacks ([`Deborah.Sarah.DataLoader`](@ref)).
- **Resampling**: bootstrap and jackknife engines plus runners/utilities ([`Deborah.Sarah.Bootstrap`](@ref), [`Deborah.Sarah.Jackknife`](@ref), [`Deborah.Sarah.BootstrapRunner`](@ref)).
- **Summaries**: pretty `(avg, err)` formatting and printing; accumulate per-observable summaries ([`Deborah.Sarah.SummaryFormatter`](@ref), [`Deborah.Sarah.SummaryCollector`](@ref)).

# Key Components (selected)
- [`Deborah.Sarah.JobLoggerTools`](@ref) — job-scoped logging and convenience printers.
- [`TOMLLogger.append_section_to_toml`](@ref) — append a new `[section]` to a [`TOML`](https://toml.io/en/) file.
- [`Deborah.Sarah.StringTranscoder`](@ref) — encode/decode feature/target names, abbreviation dicts.
- [`Deborah.Sarah.NameParser`](@ref) — build canonical `overall_name`, model suffixes, and path pieces.
- [`Deborah.Sarah.SeedManager`](@ref) — set/get seeds for RNG workflows.
- [`Deborah.Sarah.DatasetPartitioner`](@ref) — compute label/train/bias-correction/unlabeled indices.
- [`XYInfoGenerator.gen_conf_from_Y`](@ref) — pick per-config IDs from a `Y` matrix (or `1:N_cnf`).
- [`XYInfoGenerator.gen_X_info`](@ref) — extract a single component column into `(channels, N_cnf, N_src)`.
- [`Deborah.Sarah.Bootstrap`](@ref), [`Deborah.Sarah.Jackknife`](@ref) — resampling cores; [`Deborah.Sarah.BootstrapRunner`](@ref) orchestrates block bootstrap schemes.
- [`SummaryFormatter.print_bootstrap_average_error`](@ref) — compute & print `AVG(ERR)`.
- [`SummaryFormatter.print_jackknife_average_error`](@ref) — from JK samples.
- [`SummaryFormatter.print_jackknife_average_error_from_raw`](@ref) — from raw data + bin size.
- [`Deborah.Sarah.SummaryCollector`](@ref) — gather per-module summaries into dicts for downstream printers.

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah

# Append metadata to an existing TOML
julia> using Deborah.Sarah.TOMLLogger, OrderedCollections
julia> TOMLLogger.append_section_to_toml("config.toml", "metadata", OrderedCollections.OrderedDict("creator"=>"alice","notes"=>"test"))  # appends a [metadata] section

# Extract configuration IDs and X info from raw matrices
julia> using Deborah.Sarah.XYInfoGenerator
julia> conf = XYInfoGenerator.gen_conf_from_Y(Y, N_cnf, N_src, 3)  # read conf IDs from 3rd column
julia> X_info = XYInfoGenerator.gen_X_info(X, N_cnf, N_src, 1)      # take column 1 as the component

# Print bootstrap and jackknife summaries
julia> using Deborah.Sarah.SummaryFormatter
julia> SummaryFormatter.print_bootstrap_average_error(bs_arr, "Y_P2", "OBS")
julia> SummaryFormatter.print_jackknife_average_error_from_raw(raw_arr, 20, "cond", "OJK")
```

# Notes

* Column indices are **``1``-based** throughout (files and arrays).
* Bootstrap/Jackknife routines return `(mean, std)`; [`Deborah.Sarah.SummaryFormatter`](@ref) also prints a compact `AVG(ERR)` string and warns when `std == 0`. 
* [`TOMLLogger.append_section_to_toml`](@ref) writes keys using `repr`, preserving Julia-literal fidelity in the [`TOML`](https://toml.io/en/). 

# See Also

* [`Deborah.DeborahCore`](@ref), 
* [`Deborah.Esther`](@ref), 
* [`Deborah.Miriam`](@ref), 
* [`Deborah.Rebekah`](@ref), 
* [`Deborah.RebekahMiriam`](@ref).
"""
module Sarah

include("JobLoggerTools.jl")
include("TOMLLogger.jl")
include("StringTranscoder.jl")
include("NameParser.jl")
include("SeedManager.jl")
include("DatasetPartitioner.jl")
include("XYInfoGenerator.jl")
include("DataLoader.jl")
include("Bootstrap.jl")
include("Jackknife.jl")
include("BlockSizeSuggester.jl")
include("AvgErrFormatter.jl")
include("BootstrapDataInit.jl")
include("BootstrapRunner.jl")
include("SummaryFormatter.jl")
include("SummaryCollector.jl")
include("ControllerCommon.jl")

using .JobLoggerTools
using .TOMLLogger
using .StringTranscoder
using .NameParser
using .SeedManager
using .DatasetPartitioner
using .XYInfoGenerator
using .DataLoader
using .Bootstrap
using .Jackknife
using .BlockSizeSuggester
using .AvgErrFormatter
using .BootstrapDataInit
using .BootstrapRunner
using .SummaryFormatter
using .SummaryCollector
using .ControllerCommon

end  # module Sarah