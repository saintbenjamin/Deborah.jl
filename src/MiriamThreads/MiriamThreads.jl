# ============================================================================
# src/MiriamThreads/MiriamThreads.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module MiriamThreads

# `Deborah.MiriamThreads` — Threaded dispatcher for multi-job [`Deborah.Miriam`](@ref) (multi-ensemble) runs.

`Deborah.MiriamThreads` parallelizes [`Deborah.Miriam`](@ref) reweighting/interpolation analyses across
`LBP`/`TRP` grids using [Julia Threads](https://docs.julialang.org/en/v1/base/multi-threading/). It parses a batch [`TOML`](https://toml.io/en/) into concrete job
arguments, materializes per-job configs, ensures ensemble/multi-ensemble
prerequisites, and launches computation in batches of [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads).

# Scope & Responsibilities
- **Config expansion**: Turn a single batch [`TOML`](https://toml.io/en/) into a list of [`MiriamThreadsRunner.JobArgsMiriam`](@ref)
  (one per `(label, train)` pair).
- **Per-job config writer**: Serialize [`MiriamThreadsRunner.JobArgsMiriam`](@ref) to a [`TOML`](https://toml.io/en/)-compatible
  `OrderedDict` and save as `config_Miriam_<overall_name>.toml`.
- **Dependency guards**:
  - Ensure single-ensemble prerequisites ([`Deborah.DeborahCore`](@ref)→[`Deborah.Esther`](@ref)) via
    [`Deborah.DeborahEstherMiriam.MiriamDependencyManager.ensure_ensemble_exists`](@ref).
  - Validate multi-ensemble outputs via
    [`Deborah.DeborahEstherMiriam.MiriamExistenceManager.ensure_multi_ensemble_exists`](@ref).
- **Threaded execution**: Dispatch jobs in batches; each job writes its own `.log`.

# Public API
- [`MiriamThreadsRunner.run_MiriamThreads`](@ref)  
- [`MiriamThreadsRunner.run_MiriamThreadsCheck`](@ref)  

# File/Path Conventions
- Output base (abbrev on/off):
```
<location>/<analysis_header>*<multi_ensemble>/
<analysis_header>*<multi_ensemble>_<TrM1..TrM4(+model_suffixes)>/
```
- Per-job directory: `.../<overall_name>/`
- Per-job config: `config_Miriam_<overall_name>.toml`
- Per-job log: `run_Miriam_<overall_name>.log`

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah
julia> run_MiriamThreads("config_MiriamThreads.toml")
# or check-only:
julia> run_MiriamThreadsCheck("config_MiriamThreads.toml")
```

# Notes
* **``1``-based indices**: all `read_column_*` and `index_column` fields are ``1``-based.
* **Abbreviation support**: when `use_abbreviation = true`, compact codes are used to
  build `overall_name` and directory names.
* **Parallel width**: equals [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads); set [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS) to adjust.
* **Resampling**: supports bootstrap `method` ``\\in`` `{"nonoverlapping","moving","circular"}`.

# See Also

* [`Deborah.DeborahEstherMiriam.MiriamDependencyManager`](@ref), 
* [`Deborah.DeborahEstherMiriam.MiriamExistenceManager`](@ref),
* [`Deborah.Miriam.MiriamRunner`](@ref).
"""
module MiriamThreads

using ..Sarah
using ..DeborahEsther
using ..DeborahEstherMiriam
using ..Miriam

include("MiriamThreadsRunner.jl")

using .MiriamThreadsRunner

end  # module MiriamThreads