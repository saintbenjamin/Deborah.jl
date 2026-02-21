# ============================================================================
# src/Elijah/Elijah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Elijah

# `Deborah.Elijah` — Interactive wizards for [`Deborah.DeborahCore`](@ref) / [`Deborah.Esther`](@ref) / [`Deborah.Miriam`](@ref) (and [`Threads`](https://docs.julialang.org/en/v1/manual/multi-threading/) variants).

`Deborah.Elijah` provides [`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/)-based interactive wizards that generate [`TOML`](https://toml.io/en/)
configuration files for the `Deborah.jl` ecosystem:
- **[`Deborah.DeborahCore`](@ref)** (ML-based trace estimation with bias correction),
- **[`Deborah.Esther`](@ref)** (cumulant analysis for single ensemble),
- **[`Deborah.Miriam`](@ref)** (multi-ensemble reweighting/interpolation),
plus their **[`Threads`](https://docs.julialang.org/en/v1/manual/multi-threading/)** batch/parallel runners.

Each wizard guides you through required fields, validates inputs, optionally
loads an abbreviation map, and writes a ready-to-run `config_*.toml`.

# Scope & Responsibilities
- **Guided config authoring**: prompt for paths, features/targets, indices,
  model tags, and statistical settings (bootstrap/jackknife).
- **Ecosystem coverage**:
  - [`DeborahWizardRunner.run_DeborahWizard`](@ref) → `config_Deborah.toml` (single-job). 
  - [`EstherWizardRunner.run_EstherWizard`](@ref)  → `config_Esther.toml`  (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``).
  - [`MiriamWizardRunner.run_MiriamWizard`](@ref)  → `config_Miriam.toml`  (multi-``\\kappa``, solver/trajectory). 
  - [`DeborahThreadsWizardRunner.run_DeborahThreadsWizard`](@ref) → `config_DeborahThreads.toml` (labels``\\times``trains grid).
  - [`EstherThreadsWizardRunner.run_EstherThreadsWizard`](@ref)  → `config_EstherThreads.toml`  (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` across labels``\\times``trains).
  - [`MiriamThreadsWizardRunner.run_MiriamThreadsWizard`](@ref)  → `config_MiriamThreads.toml`  (multi-``\\kappa`` ``\\times`` labels``\\times``trains with solver/trajectory).
- **Abbreviation support**: inline or external [`TOML`](https://toml.io/en/), used to encode feature/target names
  and to derive compact path/name suffixes.
- **[`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/) UX**: clear prompts, defaults, and immediate file emission.

# Public API
- [`DeborahWizardRunner.run_DeborahWizard`](@ref)
- [`EstherWizardRunner.run_EstherWizard`](@ref)
- [`Elijah.MiriamWizardRunner.run_MiriamWizard`](@ref)
- [`DeborahThreadsWizardRunner.run_DeborahThreadsWizard`](@ref)
- [`EstherThreadsWizardRunner.run_EstherThreadsWizard`](@ref)
- [`MiriamThreadsWizardRunner.run_MiriamThreadsWizard`](@ref)

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah

# Single-job configs
julia> run_DeborahWizard()
julia> run_EstherWizard()
julia> run_MiriamWizard()

# Threaded/batch configs
julia> run_DeborahThreadsWizard()
julia> run_EstherThreadsWizard()
julia> run_MiriamThreadsWizard()
```

# Notes

* Wizards persist exactly the sections expected by downstream runners
  (e.g., `[data]`, `[bootstrap]`, `[jackknife]`, and where applicable
  `[input_meta]`, `[deborah]`, `[solver]`, `[trajectory]`).
* Ensemble names for [`Deborah.Esther`](@ref)/[`Deborah.Miriam`](@ref) variants are derived from lattice metadata
  (e.g., `ns`, `nt`, `beta`, `kappa`) to keep paths reproducible.
* All column indices are **``1``-based** to match typical data files.

# See Also

[`Deborah.DeborahCore`](@ref), [`Deborah.Esther`](@ref), [`Deborah.Miriam`](@ref), [`Deborah.DeborahThreads`](@ref), [`Deborah.EstherThreads`](@ref), [`Deborah.MiriamThreads`](@ref).
"""
module Elijah

include("DeborahWizardRunner.jl")
include("EstherWizardRunner.jl")
include("MiriamWizardRunner.jl")
include("DeborahThreadsWizardRunner.jl")
include("EstherThreadsWizardRunner.jl")
include("MiriamThreadsWizardRunner.jl")

using .DeborahWizardRunner
using .EstherWizardRunner
using .MiriamWizardRunner
using .DeborahThreadsWizardRunner
using .EstherThreadsWizardRunner
using .MiriamThreadsWizardRunner

end