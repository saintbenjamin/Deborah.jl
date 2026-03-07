# ============================================================================
# src/DeborahThreads/DeborahThreads.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module DeborahThreads

# `Deborah.DeborahThreads` — Threaded dispatcher for multi-job [`Deborah.DeborahCore`](@ref) runs.

`Deborah.DeborahThreads` parallelizes multiple [`Deborah.DeborahCore`](@ref) jobs (different `LBP`/`TRP` partitions,
or multiple model/config variations) using [Julia Threads](https://docs.julialang.org/en/v1/manual/multi-threading/). It parses a single [`TOML`](https://toml.io/en/) that
enumerates label/train pairs, expands them into per-job arguments, writes per-job configs,
and launches [`Deborah.DeborahCore.DeborahRunner.run_Deborah`](@ref) concurrently with simple batch scheduling.

# Scope & Responsibilities
- **Config expansion**: Parse a multi-job [`TOML`](https://toml.io/en/) and materialize one job per `(LBP, TRP)`.
- **Per-job config writer**: Generate [`TOML`](https://toml.io/en/) dictionaries/files for each job.
- **Threaded execution**: Dispatch jobs in batches of [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads); wait per batch.
- **Run logging**: Record start/end timestamps and errors per job into `run_*.log`.

# Key Components
- [`DeborahThreadsRunner.JobArgsDeborah`](@ref): immutable container of a single [`Deborah.DeborahCore`](@ref) job's arguments.
- [`DeborahThreadsRunner.parse_config_DeborahThreads`](@ref) → `Vector{JobArgsDeborah}`:  
  Expand the global config into concrete jobs.
- [`DeborahThreadsRunner.generate_toml_dict`](@ref) → `Dict`:  
  Build a [`TOML`](https://toml.io/en/)-ready dictionary for one job (data / bootstrap / jackknife / abbreviation).
- [`DeborahThreadsRunner.run_one_job`](@ref):
  Write per-job [`TOML`](https://toml.io/en/), call [`Deborah.DeborahCore.DeborahRunner.run_Deborah`](@ref), and log results.
- [`DeborahThreadsRunner.run_DeborahThreads`](@ref):  
  High-level entry point: parse → batch → spawn → wait.

# File/Path Conventions
- Output base: `<location>/<analysis_header>_<ensemble>/<analysis_header>_<ensemble>_<X_Y_or_code>_<model_suffix>/`
- Per-job directory: `.../<overall_name>/`
- Per-job config: `config_Deborah_<overall_name>.toml`
- Per-job log: `run_Deborah_<overall_name>.log`

# Minimal Usage
```julia
julia> using Deborah
julia> run_DeborahThreads("config_DeborahThreads.toml")
```

# Notes

* Abbreviation support: feature/target names can be encoded via [`Deborah.Sarah.StringTranscoder`](@ref).
* Batching equals [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads); adjust [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS) to change parallel width.
* Column indices in the input spec are **``1``-based**; they are propagated into per-job [`TOML`](https://toml.io/en/)s.
* Exceptions are caught per job and written into the corresponding `.log` file.
"""
module DeborahThreads

import ..TOML
import ..OrderedCollections
import ..Dates
import ..Base.Threads

using ..Sarah
using ..DeborahCore

include("DeborahThreadsRunner.jl")

using .DeborahThreadsRunner

end  # module DeborahThreads