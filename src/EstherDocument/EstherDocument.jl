# ============================================================================
# src/EstherDocument/EstherDocument.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module EstherDocument

# `Deborah.EstherDocument` — Document & figure generation pipeline for [`Deborah.Esther`](@ref) results.

`Deborah.EstherDocument` loads precomputed [`Deborah.Esther`](@ref) summaries, assembles comparison keys
(``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)`` and ``Q_n \\; (n=1,2,3,4)`` with derived observables), builds stable names/paths
from your configuration (with optional abbreviation encoding), saves a consolidated
[`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshot.

# Scope & Responsibilities
- **Config parsing**: read a single [`TOML`](https://toml.io/en/) and extract `[data]`, `[bootstrap]`, `[jackknife]`,
  and `[abbreviation]` needed for downstream loads.
- **Naming & encoding**: construct `analysis_ensemble`, `overall_name`, `cumulant_name`
  using full `X_Y` names or abbreviation codes plus model suffixes.  
- **Summary loading**: load multi-key results (``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)``, ``Q_n \\; (n=1,2,3,4)``, `cond`/`susp`/`skew`/`kurt`)
  across `label`/`train` grids via the reporting backend.  
- **Snapshot & layout**: save `results_<overall_name>.jld2` under the analysis tree
  and copy a mirror to the current working directory.

# Key Component
- [`EstherDocumentRunner.run_EstherDocument`](@ref)
  End-to-end document-prep runner for a single ensemble (no return; side-effecting).

# Config Expectations ([`TOML`](https://toml.io/en/))
- `[data]`: `location`, `ensemble`, `analysis_header`, `labels`, `trains`,
  `use_abbreviation`, and per-target fields for ``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)``
  (`*_X`, `*_Y`, `*_model`, `*_read_column_X`, `*_read_column_Y`, `*_index_column`).  
- `[abbreviation]`: optional `Dict{String,String}` for filename↔token mapping.  
- `[bootstrap]`, `[jackknife]`: present (consumed upstream in your workflow).

# Outputs
- **[`JLD2`](https://juliaio.github.io/JLD2.jl/stable)**:  
  `<location>/<analysis_header>_<ensemble>/<analysis_header>_<overall_name>/results_<overall_name>.jld2`  
  and a copy `./results_<overall_name>.jld2`.  

# Minimal Usage
```julia
julia> using Deborah
julia> run_EstherDocument("config_EstherDocument.toml")
```

# Notes

* Abbreviation maps are optional; when enabled, encoded names are used to build
  path suffixes and `learning` tags.
* Column indices are **``1``-based** and propagated as-is from the [`TOML`](https://toml.io/en/).
"""
module EstherDocument

import ..TOML

using ..Sarah
using ..Rebekah

include("EstherDocumentRunner.jl")

using .EstherDocumentRunner

end