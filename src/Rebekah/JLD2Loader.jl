# ============================================================================
# src/Rebekah/JLD2Loader.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module JLD2Loader

import JLD2
import ..Sarah.JobLoggerTools

"""
    load_jld2(
        filepath::String
    ) -> NamedTuple

Load `JLD2` data file containing summary statistics, labels, and training identifiers.

This function reads a `.jld2` file and returns a named tuple with keys `:summary`, `:labels`, and `:trains`. 
It is typically used in workflows that do not require reweighting data.

# Arguments
- `filepath::String`: Path to the `.jld2` file to load.

# Returns
- A `NamedTuple` with fields `summary`, `labels`, and `trains`.
"""
function load_jld2(
    filepath::String
)
    data = JLD2.load(filepath)
    return (
        summary = data["summary"],
        labels  = data["labels"],
        trains  = data["trains"]
    )
end

"""
    load_jld2_Miriam(
        filepath::String
    ) -> NamedTuple

Load a consolidated Miriam `JLD2` bundle, including:
- results on multi-ensemble reweighting and transition point interpolation: `summary` (keyword-indexed),
- three measurement-style summaries from single-ensemble measurement points:
  `summary_trace_meas`, `summary_moment_meas`, `summary_cumulant_meas`,
- the kappa token list `kappa_list` (e.g., `["13570", "13575", "13580"]`),
- the reweighting payload `rw_data`,
- **the [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) convergence payload `nlsolve_status` (optional; defaults to empty dict),**
- and index metadata `labels`, `trains`.

This loader is tailored for overlap/error analysis and measurement-based plotting workflows.

# Arguments
- `filepath::String`: Path to the `.jld2` file previously written by `save_miriam_results`.

# Returns
A `NamedTuple` with fields:
- `summary :: Dict`
- `summary_trace_meas :: Dict`
- `summary_moment_meas :: Dict`
- `summary_cumulant_meas :: Dict`
- `kappa_list :: Vector{String}`
- `rw_data :: Dict`
- `nlsolve_status :: Dict` (present if saved; otherwise empty)
- `labels :: Vector{String}`
- `trains :: Vector{String}`

# Notes
- Measurement dictionaries are keyed as `(field, :avg/:err, tag, kappa_str)` where
  `kappa_str` ``\\in`` `kappa_list`. Many plotting utilities render κ as `0.<kappa_str>` (e.g., `"13580"` → `0.13580`).
- All summary matrices are indexed `(label_index, train_index)` and align with `labels`/`trains`.
- `nlsolve_status` enables convergence heatmaps; schema:
  `Dict{String, Dict{String, Dict{String, NamedTuple{(:converged, :residual_norm), Tuple{Bool, Float64}}}}}` with access `nlsolve_status[label][train][solver]`.
- For backward compatibility, if `nlsolve_status` is not found in the file, an empty `Dict()` is returned instead of throwing.

# Example
```julia
bundle = load_jld2_Miriam("out/miriam_results.jld2")
ks_tokens   = bundle.kappa_list
trace_meas  = bundle.summary_trace_meas
moment_meas = bundle.summary_moment_meas
cum_meas    = bundle.summary_cumulant_meas
rw          = bundle.rw_data
nls_status  = bundle.nlsolve_status  # {} if absent in older files
```
"""
function load_jld2_Miriam(
    filepath::String
)
    # 1) Try the normal fast path
    try
    data = JLD2.load(filepath)

        # Backward-compatible default for older files without nlsolve payload
        empty_nls = Dict{String, Dict{String, Dict{String, NamedTuple{(:converged, :residual_norm), Tuple{Bool, Float64}}}}}()

        return (
            summary               = data["summary"],
            summary_trace_meas    = data["summary_trace_meas"],
            summary_moment_meas   = data["summary_moment_meas"],
            summary_cumulant_meas = data["summary_cumulant_meas"],
            kappa_list            = data["kappa_list"],
            rw_data               = data["rw_data"],
            nlsolve_status        = get(data, "nlsolve_status", empty_nls), # ← (safe default)
            labels                = data["labels"],
            trains                = data["trains"]
        )
    catch e
        # 2) On failure, run a targeted scan to locate flattened nodes.
        JobLoggerTools.warn_benji(
            "JLD2.load failed; running diagnostic scan to locate malformed nodes\n" *
            "exception = $(e)\n" *
            "backtrace = $(catch_backtrace())"
        )
        bad_paths = _scan_flat_vectors(filepath)

        if !isempty(bad_paths)
            JobLoggerTools.warn_benji(
                "Found entries where Vector{Float64} appeared instead of Dict{Symbol,Vector{Float64}}\n" *
                "count = $(length(bad_paths))"
            )
            for p in bad_paths
                JobLoggerTools.warn_benji(
                    "Offending path detected: $(p)"
                )
            end
        else
            JobLoggerTools.warn_benji(
                "Diagnostic scan did not find explicit flattened Vector{Float64} entries; the schema mismatch may be deeper or under a different type wrapper."
            )
        end

        # Re-raise with augmented context
        JobLoggerTools.error_benji(
            "Failed to load JLD2 file '$filepath'. See warnings above for offending paths. Original error: $(sprint(showerror, e))"
        )
    end
end

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

"""
    _scan_flat_vectors(
        filepath::AbstractString
    ) -> Vector{String}

Diagnostic helper to inspect the internal schema of a `.jld2` file and locate
possible legacy or malformed entries introduced by older Miriam save formats.

Specifically scans leaf datasets for places where a nested structure expected to
be of the form `Dict{Symbol, Vector{Float64}}` has instead been serialized one
level too shallow as `Vector{Float64}`

This pattern frequently occurs when a `SerializedDict` round-trip produces a
`Vector{Pair{Symbol,Any}}`, or when grouped solver metadata is written without
explicit schema normalization.

# Arguments
- `filepath::AbstractString` : Path to a `.jld2` file opened read-only.

# Returns
- `Vector{String}` : List of dataset paths (group-style, e.g.,
  `"summary/solver1/y"`) showing the locations where the suspicious flattening
  pattern was detected. If no issues are found, returns an empty vector.

# Notes
- Does **not** modify the file; read-only inspection.
- Used to aid migration/repair of legacy bundles and provide context when
  [`JLD2.load`](https://juliaio.github.io/JLD2.jl/stable/basic_usage/#FileIO-interface) throws schema mismatch errors.
"""
function _scan_flat_vectors(
    filepath::AbstractString
)
    bad = String[]
    jldopen(filepath, "r") do f
        _recurse_scan!(bad, f, "")
    end
    return bad
end

"""
    _recurse_scan!(
        bad::Vector{String}, 
        g::JLD2.Group, 
        basepath::AbstractString
    ) -> Nothing

Recursive depth-first traversal used internally by [`_scan_flat_vectors`](@ref).
Walks a [`JLD2.Group`](https://juliaio.github.io/JLD2.jl/stable/basic_usage/#Groups) hierarchy and identifies suspicious values where a nested
dictionary layout appears collapsed (i.e., a value is directly a `Vector{Float64}`
instead of `Dict{Symbol,Vector{Float64}}`).

# Arguments
- `bad::Vector{String}` :
    Mutable accumulator collecting detected offending dataset paths.
- [`g::JLD2.Group`](https://juliaio.github.io/JLD2.jl/stable/basic_usage/#Groups) :
    Current group node being scanned.
- `basepath::AbstractString` :
    Current hierarchical prefix used for reporting full dataset paths.

# Behavior
- For each dataset:
    1. Attempts safe deserialization (`try`/`catch` guarded).
    2. Detects common serialized forms:
        - `Vector{Pair{Symbol,Any}}` (`SerializedDict`-style)
        - `Dict` containers with shallow `Vector{Float64}` leaves.
    3. Appends human-readable paths such as `"rw_data/solver1/w"` when detected.

# Returns
- `Nothing` (side-effect only; modifies `bad`).

# Notes
- A failure to deserialize a leaf does **not** abort scanning.
- This function intentionally avoids mutation or rewriting of [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) nodes; it only reports.
"""
function _recurse_scan!(
    bad::Vector{String}, 
    g::JLD2.Group, 
    basepath::AbstractString
)
    for name in keys(g)
        child = g[name]
        path  = isempty(basepath) ? String(name) : string(basepath, "/", name)

        # If it's a subgroup, descend.
        if child isa JLD2.Group
            _recurse_scan!(bad, child, path)
            continue
        end

        # Try to read payload; some datasets deserialize into Dict/Vector structures.
        # We guard with try/catch so a failing leaf doesn't abort the whole scan.
        try
            val = read(child)

            # Case 1: SerializedDict often comes back as Vector{Pair{Symbol,Any}}
            if val isa Vector{Pair{Symbol,Any}}
                for (k, v) in val
                    if v isa Vector{Float64}
                        push!(bad, string(path, "/", k))
                    end
                end
            # Case 2: Already materialized as a Dict (string/symbol keys both possible)
            elseif val isa AbstractDict
                # We check a couple of common nestings: Dict{Symbol,Any} or Dict{String,Any}
                for (k, v) in val
                    # Offending pattern: value directly a Vector{Float64}
                    if v isa Vector{Float64}
                        push!(bad, string(path, "/", k))
                    # If there is an inner dict, you could further inspect here if needed.
                    end
                end
            end
        catch _
            # Ignore read errors at this node and continue scanning others.
            # (The main goal is to collect as many offending paths as possible.)
        end
    end
end

end  # module JLD2Loader