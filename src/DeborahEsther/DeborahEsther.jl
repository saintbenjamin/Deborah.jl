# ============================================================================
# src/DeborahEsther/DeborahEsther.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module DeborahEsther

# `Deborah.DeborahEsther` — Integrated [`Deborah.DeborahCore`](@ref) → [`Deborah.Esther`](@ref) workflow manager.

The `Deborah.DeborahEsther` module provides the bridge layer between **[`Deborah.DeborahCore`](@ref)**
(machine-learning-based bias-corrected trace estimation) and **[`Deborah.Esther`](@ref)**
(cumulant analysis in the single ensemble).  
It ensures that all required trace observables ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` are present,
regenerating them with [`Deborah.DeborahCore`](@ref) if necessary, and then executes the
[`Deborah.Esther`](@ref) stage.

# Scope & Responsibilities
- **Dependency management**:  
  - [`EstherDependencyManager.ensure_TrM_exists`](@ref)  
    Verifies presence of ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` files and auto-invokes [`Deborah.DeborahCore`](@ref) if missing.  
  - [`EstherDependencyManager.run_Deborah_from_Esther`](@ref)  
    Builds a temporary [`TOML`](https://toml.io/en/) config and launches [`Deborah.DeborahCore`](@ref) on demand.  
- **Pipeline orchestration**:  
  - [`DeborahEstherRunner.run_Deborah_Esther`](@ref)  
    Entry point for full [`Deborah.DeborahCore`](@ref) → [`Deborah.Esther`](@ref) runs, given a [`TOML`](https://toml.io/en/) config.  
- **Configuration support**:  
  - [`EstherDependencyManager.generate_toml_dict`](@ref)  
    Constructs a [`TOML`](https://toml.io/en/)-compatible config dictionary for [`Deborah.DeborahCore`](@ref) invocation.

# Typical Workflow
1. User supplies a `config_Esther.toml` including ensemble, features, and models.  
2. [`EstherDependencyManager.ensure_TrM_exists`](@ref) checks whether outputs like `Y_info`, `Y_bc`, `YP_bc` exist.  
3. If missing, [`EstherDependencyManager.run_Deborah_from_Esther`](@ref) creates and executes a matching [`Deborah.DeborahCore`](@ref) config.  
4. Once all ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` are ready, [`DeborahEstherRunner.run_Deborah_Esther`](@ref) launches the [`Deborah.Esther`](@ref) computation.  

# Minimal Usage
```julia
julia> using Deborah
julia> run_Deborah_Esther("config_Esther.toml")
```

# Notes

* [`Deborah.DeborahCore`](@ref) outputs are stored under ensemble/analysis sub-folders with consistent
  naming via [`Deborah.Sarah.NameParser`](@ref).
* Supports abbreviation dictionaries for compact path encodings.
* Bootstrap/jackknife parameters (`N_bs`, `blk_size`, `method`, `bin_size`) are
  propagated from [`Deborah.Esther`](@ref) configs into the temporary [`Deborah.DeborahCore`](@ref) configs.

# See Also

* [`Deborah.DeborahCore`](@ref) — trace estimation with ML/bias correction.
* [`Deborah.Esther`](@ref) — cumulant and transition-point analysis.
"""
module DeborahEsther

using ..Sarah
using ..DeborahCore
using ..Esther

include("EstherDependencyManager.jl")
include("DeborahEstherRunner.jl")

using .EstherDependencyManager
using .DeborahEstherRunner

end  # module DeborahEsther