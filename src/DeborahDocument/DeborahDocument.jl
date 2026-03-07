# ============================================================================
# src/DeborahDocument/DeborahDocument.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module DeborahDocument

# `Deborah.DeborahDocument` — Postprocessing and document generation for [`Deborah.DeborahCore`](@ref) results.

The `Deborah.DeborahDocument` module provides a reporting pipeline on top of [`Deborah.DeborahCore`](@ref).
It loads summary results, archives the results into [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) format. This module is designed as the "document
layer" in the `Deborah.jl` ecosystem, complementing [`Deborah.DeborahCore`](@ref) (ML + bias correction)
with a reporting backend.

# Scope & Responsibilities
- **Configuration**: Parse a [`TOML`](https://toml.io/en/) config for data, bootstrap/jackknife, and abbreviation maps.  
- **Summary loading**: Collect pre-computed results via [`Deborah.Rebekah.SummaryLoader`](@ref).  
- **Archiving**: Save results into [`.jld2`](https://github.com/JuliaIO/JLD2.jl) format with [`Deborah.Rebekah.JLD2Saver`](@ref), and copy snapshots to the working directory.  
- **Integration**: Connects [`Deborah.Sarah`](@ref) utilities (logging, name parsing, transcoding) and [`Deborah.Rebekah`](@ref)'s reporting tools.

# Key Component
- [`DeborahDocumentRunner.run_DeborahDocument`](@ref)  
  Main entry point. Runs the document-generation workflow for a single ensemble.

# Typical Outputs
- Results snapshot: `results_<overall_name>.jld2` in analysis directory.

# Minimal Usage
```julia
julia> using Deborah
julia> run_DeborahDocument("config_DeborahThreads.toml")
```

# Notes

* Abbreviation dictionary `[abbreviation]` in the [`TOML`](https://toml.io/en/) is parsed with
  [`Deborah.Sarah.StringTranscoder.parse_string_dict`](@ref).
* Input/output codes (`XY_code`, `X_Y`) and model suffixes are built with
  [`Deborah.Sarah.NameParser`](@ref).
* Assumes upstream [`Deborah.DeborahCore`](@ref) runs have produced summary data available for loading.

# See Also

* [`Deborah.DeborahCore`](@ref) — core ML/bias-correction pipeline.
* [`Deborah.Rebekah`](@ref) — reporting/plotting backend.
"""
module DeborahDocument

import ..TOML

using ..Sarah
using ..Rebekah

include("DeborahDocumentRunner.jl")

using .DeborahDocumentRunner

end  # module DeborahDocument