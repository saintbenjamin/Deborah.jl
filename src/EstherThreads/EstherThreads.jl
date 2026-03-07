# ============================================================================
# src/EstherThreads/EstherThreads.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module EstherThreads

# `Deborah.EstherThreads` — Threaded dispatcher for multi-job [`Deborah.Esther`](@ref) runs.

`Deborah.EstherThreads` parallelizes multiple [`Deborah.Esther`](@ref) computations (across `LBP`/`TRP` grids)
using [Julia Threads](https://docs.julialang.org/en/v1/manual/multi-threading/). It parses a master [`TOML`](https://toml.io/en/) config into per-job arguments, writes
job-specific configs, ensures [`Deborah.DeborahCore`](@ref) prerequisites for ``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)`` are present,
and launches [`Deborah.Esther`](@ref) in batches of [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads) with per-job logging.

# Scope & Responsibilities
- **Config expansion**: [`EstherThreadsRunner.parse_config_EstherThreads`](@ref) → list of jobs over `labels` ``\\times`` `trains`.
- **Per-job config**: [`EstherThreadsRunner.generate_toml_dict`](@ref) builds a [`TOML`](https://toml.io/en/) dict for one [`Deborah.Esther`](@ref) job.
- **Dependency check**: call [`Deborah.DeborahEsther.EstherDependencyManager.ensure_TrM_exists`](@ref)
  to guarantee ``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)`` inputs exist (auto-invokes [`Deborah.DeborahCore`](@ref) if needed).
- **Threaded execution**: spawn tasks in batches; each job writes its own `.log` file.
- **Stable naming**: construct `overall_name`/`output_base` using either full `X_Y` or
  abbreviation codes plus model suffixes.

# Public API
- [`EstherThreadsRunner.run_EstherThreads`](@ref)

# File/Path Conventions
- Output base: `<location>/<analysis_header>_<ensemble>/<analysis_header>_<ensemble>_<TrM1..TrM4(+model_suffixes)>/`
- Per-job directory: `.../<overall_name>/`
- Per-job config: `config_Esther_<overall_name>.toml`
- Per-job log: `run_Esther_<overall_name>.log`

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah
julia> run_EstherThreads("config_EstherThreads.toml")
```

# Notes

* All column indices are **``1``-based**.
* Bootstrap `method` should be `"nonoverlapping"`, `"moving"`, or `"circular"`.
* Parallel width equals [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads); set [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS) to adjust.
* Abbreviation dicts are supported and propagated when `use_abbreviation = true`.
"""
module EstherThreads

import ..TOML
import ..OrderedCollections
import ..Dates
import ..Base.Threads

using ..Sarah
using ..DeborahEsther
using ..Esther

include("EstherThreadsRunner.jl")

using .EstherThreadsRunner

end  # module EstherThreads