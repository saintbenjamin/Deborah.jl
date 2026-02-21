# ============================================================================
# src/MiriamDocument/MiriamDocument.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module MiriamDocument

# `Deborah.MiriamDocument` — Document & figure generation for multi-ensemble [`Deborah.Miriam`](@ref) results.

[`Deborah.MiriamDocument`](@ref) parses a Miriam [`TOML`](https://toml.io/en/) config, loads reweighting/interpolation summaries
across ensembles and label/train grids, computes overlap and error diagnostics, generates
figures (heatmaps, reweighting curves, etc.), and saves a consolidated [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshot.

# Scope & Responsibilities
- **Config parsing**: read `[data]`, `[abbreviation]` and related sections to resolve paths,
  ensemble names, model suffixes, and encoded feature/target names.
- **Summary loading**: aggregate `RW` (`RWBS`/`RWJK`/`RWP1`/`RWP2`), trace/moment measurement blocks,
  and cumulant blocks for all `(LBP, TRP)` combinations and ensembles.
- **Snapshotting**: write `results_<overall_name>.jld2` under the analysis tree and copy a
  mirror to the current working directory.

# Public API
- [`MiriamDocumentRunner.run_MiriamDocument`](@ref)  
  End-to-end runner that performs loading and [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) emission.

# Config Expectations ([`TOML`](https://toml.io/en/))
- `[data]`: `labels`, `trains`, `location`, `multi_ensemble`, `ensembles`,
  per-target specs for ``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)`` (`*_X`, `*_Y`, `*_model`, `*_read_column_*`, `*_index_column`),
  `analysis_header`, `use_abbreviation`.
- `[abbreviation]`: optional dictionary for filename ``\\leftrightarrow`` token encoding.

# Outputs
- **[`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshot**  
  `<location>/<analysis_header>_<multi_ensemble>/<analysis_header>_<overall_name>/results_<overall_name>.jld2`  
  and a copy `./results_<overall_name>.jld2`.

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah
julia> run_MiriamDocument("config_MiriamDocument.toml")
```

# Notes
* Abbreviation maps, when enabled, drive compact path/name construction for learning tags.
* Column indices are **``1``-based** and propagated as provided in the [`TOML`](https://toml.io/en/).
"""
module MiriamDocument

using ..Sarah
using ..RebekahMiriam

include("MiriamDocumentRunner.jl")

using .MiriamDocumentRunner

end