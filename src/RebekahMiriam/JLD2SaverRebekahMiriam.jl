# ============================================================================
# src/RebekahMiriam/JLD2SaverRebekahMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module JLD2SaverRebekahMiriam

import JLD2
import ..Sarah.JobLoggerTools

"""
    save_miriam_results(
        filepath::String,
        summary::Dict,
        summary_trace_meas::Dict,
        summary_moment_meas::Dict,
        summary_cumulant_meas::Dict,
        kappa_list::Vector{String},
        rw_data::Dict,
        nlsolve_status::Dict,
        labels::Vector{String},
        trains::Vector{String}
    ) -> Nothing

Persist Miriam outputs to a single `.jld2` file:

* interpolation results `summary` (keyword-indexed),
* three measurement-style summaries from single-ensemble measurement points:
  `summary_trace_meas`, `summary_moment_meas`, `summary_cumulant_meas`,
* the ``\\kappa`` token list `kappa_list` (e.g., `["13570","13575","13580", ...]`),
* the reweighting payload `rw_data`,
* the [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) convergence payload `nlsolve_status`,
* and index metadata `labels`, `trains`.

# Arguments

* `filepath`:
  Destination [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) path (e.g., `"out/miriam_results.jld2"`). Parent directories must exist.
* `summary`:
  Dictionary produced by [`Deborah.RebekahMiriam.SummaryLoaderRebekahMiriam.load_miriam_summary`](@ref) (keyword-based / interpolation results).
  Typical schema:
  `Dict{Tuple{Symbol,Symbol,Symbol,String}, Matrix{Float64}}`
  with key `(field, stat, tag, keyword)`:
  - `field`   ``\\in`` observed fields (e.g., `:cond`, `:susp`, `:skew`, `:kurt`, `:bind`, `:kappa_t`)
  - `stat`    ``\\in`` (`:avg`, `:err`)
  - `tag`     ``\\in`` input tags (e.g., `:RWP2`)
  - `keyword` ``\\in`` With which the interpolation is done?
  Each matrix has size `(length(labels), length(trains))`.
* `summary_trace_meas`:
  Measurement summary for trace observables (e.g., `:trM1`..`:trM4`).
  Schema:
  `Dict{Tuple{Symbol,Symbol,Symbol,String}, Matrix{Float64}}`
  with key `(field, stat, tag, kappa_str)` and matrices sized `(length(labels), length(trains))`.
* `summary_moment_meas`:
  Measurement summary for moment observables (e.g., `:Q1`,`:Q2`,`:Q3`,`:Q4`).
  Same schema/key shape as `summary_trace_meas`.
* `summary_cumulant_meas`:
  Measurement summary for cumulant observables.
  Same schema/key shape as `summary_trace_meas`. The `field` set depends on `first_block_keys`.
* `kappa_list::Vector{String}`:
  Ordered list of kappa tokens used to index the measurement summaries
  (e.g., `["13570","13575","13580"]`). Each token corresponds to a numeric ``\\kappa`` of `0.<token>`
  (e.g., `"13580"` → `\\kappa = 0.13580`). The order should match how the summaries were populated.
* `rw_data`:
  Reweighting payload returned by [`Deborah.RebekahMiriam.SummaryLoaderRebekahMiriam.load_all_rw_data`](@ref).
* `nlsolve_status`:
  [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) convergence info per `(label, train)` and solver name.
  Schema: `Dict{String, Dict{String, Dict{String, NamedTuple{(:converged, :residual_norm), Tuple{Bool, Float64}}}}}`
  Access pattern: `nlsolve_status[label][train][solver] => (converged, residual_norm)`.
* `labels`:
  Vector of `LBP` label strings; defines the row axis of all matrices.
* `trains`:
  Vector of `TRP` percentage strings; defines the column axis of all matrices.

# Behavior

Writes a single [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) file containing:
* `summary                :: Dict`
* `summary_trace_meas     :: Dict`
* `summary_moment_meas    :: Dict`
* `summary_cumulant_meas  :: Dict`
* `kappa_list             :: Vector{String}`
* `rw_data                :: Dict`
* `nlsolve_status         :: Dict`   # ← added
* `labels                 :: Vector{String}`
* `trains                 :: Vector{String}`

# Notes

* Measurement dictionaries are kappa-indexed with keys `(field, :avg/:err, tag, kappa_str)`,
  where `kappa_str` ``\\in`` `kappa_list`. Downstream plotting/heatmap utilities may display ``\\kappa`` as `0.<kappa_str>`.
* All matrices are indexed as `(label_index, train_index)` and must align with `labels`/`trains`.
* The function overwrites `filepath` if it already exists and does not create parent directories.
* **`nlsolve_status`** enables downstream convergence heatmaps: e.g., mark cells white/black by `converged`,
  and annotate black cells with `residual_norm`.

# Example

```julia
save_miriam_results(
    "out/miriam_results.jld2",
    summary,                 # from load_miriam_summary (legacy, optional)
    summary_trace_meas,      # from load_miriam_summary_for_measurement (traces)
    summary_moment_meas,     # from load_miriam_summary_for_measurement (moments)
    summary_cumulant_meas,   # from load_miriam_summary_for_measurement (cumulants)
    kappa_list,              # e.g., ["13570","13575","13580"]
    rw_data,
    nlsolve_status,
    labels, trains
)

# Later:
data = JLD2.load("out/miriam_results.jld2")
ks_tokens   = data["kappa_list"]           # ["13570","13575","13580", ...]
trace_meas  = data["summary_trace_meas"]
moment_meas = data["summary_moment_meas"]
cum_meas    = data["summary_cumulant_meas"]
nls_status  = data["nlsolve_status"]       # Dict for convergence heatmaps
```

# Returns
`Nothing`. Writes to disk.
"""
function save_miriam_results(
    filepath::String,
    summary::Dict,
    summary_trace_meas::Dict,
    summary_moment_meas::Dict,
    summary_cumulant_meas::Dict,
    kappa_list::Vector{String},
    rw_data::Dict,
    nlsolve_status::Dict,
    labels::Vector{String},
    trains::Vector{String}
)::Nothing
    jobid = nothing
    JobLoggerTools.info_benji("Saving Miriam summary, RW data, NLsolve status, and metadata to $filepath ...", jobid)
    JLD2.jldsave(filepath;
        summary = summary,
        summary_trace_meas = summary_trace_meas,
        summary_moment_meas = summary_moment_meas,
        summary_cumulant_meas = summary_cumulant_meas,
        kappa_list = kappa_list,
        rw_data = rw_data,
        nlsolve_status = nlsolve_status,
        labels  = labels,
        trains  = trains
    )

    JobLoggerTools.info_benji("Save completed.", jobid)
end

end  # module JLD2SaverRebekahMiriam