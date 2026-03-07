# ============================================================================
# src/DeborahEstherMiriam/DeborahEstherMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module DeborahEstherMiriam

# `Deborah.DeborahEstherMiriam` — Full pipeline integration: [`Deborah.DeborahCore`](@ref) → [`Deborah.Esther`](@ref) → [`Deborah.Miriam`](@ref).

The `Deborah.DeborahEstherMiriam` module provides the highest-level orchestration layer
in the `Deborah.jl` ecosystem. It unifies **[`Deborah.DeborahCore`](@ref)** (ML-based trace estimation),
**[`Deborah.Esther`](@ref)** (cumulant analysis at the single ensemble), and **[`Deborah.Miriam`](@ref)**
(multi-ensemble reweighting and interpolation).  
This module guarantees that all intermediate results exist (creating them on demand),
and drives single-ensemble-level and multi-ensemble analyses end-to-end.

# Scope & Responsibilities
- **Single-ensemble prerequisites**:  
  - [`MiriamDependencyManager.ensure_ensemble_exists`](@ref)  
    Ensures that each ensemble has complete [`Deborah.DeborahCore`](@ref) ``+`` [`Deborah.Esther`](@ref) outputs, invoking
    [`MiriamDependencyManager.run_Esther_from_Miriam`](@ref) where needed.  
- **Multi-ensemble prerequisites**:  
  - [`MiriamExistenceManager.ensure_multi_ensemble_exists`](@ref)  
    Checks whether required `.dat` outputs for combined ensembles exist; if missing,
    re-runs ensemble-level [`Deborah.Esther`](@ref) ``+`` [`Deborah.Miriam`](@ref) steps automatically.  
- **Full pipeline execution**:  
  - [`DeborahEstherMiriamRunner.run_Deborah_Esther_Miriam`](@ref)  
    Entry point for running the complete [`Deborah.DeborahCore`](@ref) → [`Deborah.Esther`](@ref) → [`Deborah.Miriam`](@ref) process.

# Workflow
1. User provides a multi-ensemble [`TOML`](https://toml.io/en/) config with data paths, bootstrap/jackknife
   parameters, and model specifications.  
2. [`MiriamDependencyManager.ensure_ensemble_exists`](@ref) validates or generates [`Deborah.DeborahCore`](@ref)/[`Deborah.Esther`](@ref) outputs
   for each ensemble.  
3. [`MiriamExistenceManager.ensure_multi_ensemble_exists`](@ref) validates or regenerates multi-ensemble Miriam outputs.  
4. [`DeborahEstherMiriamRunner.run_Deborah_Esther_Miriam`](@ref) executes the [`Deborah.Miriam`](@ref) stage, producing multi-ensemble-reweighted
   and interpolated cumulants across ensembles.

# Outputs
- Single-ensemble-level results: `summary_Esther_*.dat`, trace-based bootstrap/jackknife bundles.  
- Multi-ensemble results: `RWBS`, `RWJK`, `RWP1`, `RWP2`, `Y_BS`, `Y_JK`, `Y_P1`, `Y_P2`
  `.dat` files in analysis directories.  
- Logs: Structured job-aware logging ([`Deborah.Sarah.JobLoggerTools`](@ref)) during each stage.

# Minimal Usage
```julia
julia> using Deborah
julia> run_Deborah_Esther_Miriam("config_Miriam.toml")
```

# Notes

* Abbreviation maps are supported for compact path encoding.
* Both single-ensemble and multi-ensemble runs are guarded: missing files trigger
  regeneration automatically.
* Bootstrap (`N_bs`, `blk_size`, `method`) and jackknife (`bin_size`) configs are
  consistently propagated across Deborah, Esther, and Miriam layers.
* All path/filename construction respects `LBP`/`TRP` splits and ensemble tags.

# See Also

* [`Deborah.DeborahCore`](@ref) — ML trace estimation.
* [`Deborah.DeborahEsther`](@ref) — bridge from [`Deborah.DeborahCore`](@ref) to [`Deborah.Esther`](@ref).
* [`Deborah.Miriam`](@ref) — multi-ensemble reweighting & interpolation.
"""
module DeborahEstherMiriam

import ..TOML
import ..OrderedCollections

using ..Sarah
using ..Esther
using ..Miriam
using ..DeborahEsther

include("MiriamDependencyManager.jl")
include("MiriamExistenceManager.jl")
include("DeborahEstherMiriamRunner.jl")

using .MiriamDependencyManager
using .MiriamExistenceManager
using .DeborahEstherMiriamRunner

end  # module DeborahEstherMiriam