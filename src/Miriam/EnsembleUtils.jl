# ============================================================================
# src/Miriam/EnsembleUtils.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module EnsembleUtils

import ..Sarah.JobLoggerTools
import ..Ensemble
import ..TOMLConfigMiriam

"""
    trMiT(
        ens::Ensemble.EnsembleStruct{T}, 
        paramT::Ensemble.Params{T}, 
        i::Int
    ) -> Vector{T}

Evaluate the trace moment `trMi` at index `i`, reweighted to a different ``\\kappa`` value.

# Arguments
- [`ens::Ensemble.EnsembleStruct{T}`](@ref Deborah.Miriam.Ensemble.EnsembleStruct): Original ensemble containing trace moments
- [`paramT::Ensemble.Params{T}`](@ref Deborah.Miriam.Ensemble.Params): Target parameter with new ``\\kappa`` value
- `i::Int`: Gauge configuration index

# Returns
- `Vector{T}`: Reweighted trace moment vector of same length as `ens.trMi[i]`

---

# Notes

## Trace Moment Shift via Taylor Expansion

Evaluates the trace moment vector `ens.trMi[i]` at a shifted ``\\kappa`` value using a 4th-order Taylor expansion in the hopping mass.

## Input Format

Each original trace moment is assumed to be of the form:

```math
\\texttt{trMi[i]} = \\left\\{ \\;
12 \\, N_{\\text{f}} \\, V \\, , \\;
\\text{Tr} \\, M^{-1} \\, , \\;
\\text{Tr} \\, M^{-2} \\, , \\;
\\text{Tr} \\, M^{-3} \\, , \\;
\\text{Tr} \\, M^{-4}
\\; \\right \\}
```

Here, the first entry is the rescaling factor, and the remaining entries represent trace moments of the inverse Dirac operator raised to powers 1 through 4.

In this context, we assume that all ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` have already been properly rescaled. (For the rescaling process in [`Deborah.Miriam`](@ref), see [`Deborah.Miriam.MultiEnsembleLoader.generate_trMi_vector`](@ref) or [`Deborah.Miriam.Cumulants.calc_Q`](@ref). Note, however, that the rescaling convention in [`Deborah.Esther`](@ref) is slightly different; to check it, see [`Deborah.Esther.TraceRescaler.rescale_trace`](@ref).)

## Mass Shift

The mass shift is defined by:

```math
\\delta m = \\frac{1}{2} \\left( \\frac{1}{\\kappa} - \\frac{1}{\\kappa_{\\text{target}}} \\right)
```

## Shifted Output

The reweighted trace moments are computed via:

```math
\\begin{aligned}
\\texttt{ret[1]} &= 12 N_f V \\\\
\\texttt{ret[2]} &= \\left[\\text{Tr}\\,M^{-1}\\right]\\left(\\kappa_{\\text{target}}\\right)=\\left[\\text{Tr}\\,M^{-1}\\right]\\left(\\kappa\\right)+\\delta m\\;\\left[\\text{Tr}\\,M^{-2}\\right]\\left(\\kappa\\right)+\\left(\\delta m\\right)^{2}\\;\\left[\\text{Tr}\\,M^{-3}\\right]\\left(\\kappa\\right)+\\left(\\delta m\\right)^{3}\\;\\left[\\text{Tr}\\,M^{-4}\\right]\\left(\\kappa\\right)\\\\
\\texttt{ret[3]} &= \\left[\\text{Tr}\\,M^{-2}\\right]\\left(\\kappa_{\\text{target}}\\right)=\\left[\\text{Tr}\\,M^{-2}\\right]\\left(\\kappa\\right)+2\\;\\left(\\delta m\\right)\\;\\left[\\text{Tr}\\,M^{-3}\\right]\\left(\\kappa\\right)+3\\;\\left(\\delta m\\right)^{2}\\;\\left[\\text{Tr}\\,M^{-4}\\right]\\left(\\kappa\\right)\\\\
\\texttt{ret[4]} &= \\left[\\text{Tr}\\,M^{-3}\\right]\\left(\\kappa_{\\text{target}}\\right)=\\left[\\text{Tr}\\,M^{-3}\\right]\\left(\\kappa\\right)+3\\;\\left(\\delta m\\right)\\;\\left[\\text{Tr}\\,M^{-4}\\right]\\left(\\kappa\\right)\\\\
\\texttt{ret[5]} &= \\left[\\text{Tr}\\,M^{-4}\\right]\\left(\\kappa_{\\text{target}}\\right)=\\left[\\text{Tr}\\,M^{-4}\\right]\\left(\\kappa\\right)
\\end{aligned}
```

This corresponds to a 4th-order Taylor reweighting of trace observables in the hopping parameter ``\\kappa``.
"""
function trMiT(
    ens::Ensemble.EnsembleStruct{T},
    paramT::Ensemble.Params{T},
    i::Int
) where {T}
    # make everything type-stable in T
    half  = T(1//2) 
    oneT  = one(T)
    two   = T(2)
    three = T(3)

    # δm = 1/2 * (1/κ_a - 1/κ_T)
    dm = half * (oneT / ens.param.kappa - oneT / paramT.kappa)

    val = ens.trMi[i]
    ret = Vector{T}(undef, length(val))

    ret[1] = val[1]
    ret[2] = val[2] + dm * (val[3] + dm * (val[4] + dm * val[5]))
    ret[3] = val[3] + dm * (two * val[4] + three * dm * val[5])
    ret[4] = val[4] + three * dm * val[5]     # ← fixed (3, not 6)
    ret[5] = val[5]

    return ret
end

"""
    trMi_rawT(
        ens::Ensemble.EnsembleStruct{T},
        paramT::Ensemble.Params{T},
        i::Int
    ) -> Vector{T}

Utility variant of trMiT that operates on un-rescaled raw trace moments. (For
 details on un-rescaling, see
[`Deborah.Miriam.MultiEnsembleLoader.generate_trMi_raw_vector`](@ref); for details on
rescaling, see [`Deborah.Miriam.MultiEnsembleLoader.generate_trMi_vector`](@ref).)

It reads `ens.trMi_raw[i]` (typically `[1.0, trM1, trM2, trM3, trM4]`) and
applies the same ``\\kappa``-shift logic as [`trMiT`](@ref). The first entry is
preserved as-is.

# Arguments
- [`ens::Ensemble.EnsembleStruct{T}`](@ref
  Deborah.Miriam.Ensemble.EnsembleStruct): Source ensemble providing `ens.trMi_raw[i]`.
- [`paramT::Ensemble.Params{T}`](@ref Deborah.Miriam.Ensemble.Params): Target
  parameters
- `i::Int`: Configuration index into `ens.trMi_raw[i]`.

# Returns
- `Vector{T}`: Reweighted raw trace vector with the same length as
  `ens.trMi_raw[i]`. The first component is passed through unchanged (commonly
  ``1.0``), and entries ``2,3,4,5`` are the raw traces.

# Notes
- When `paramT.kappa == ens.param.kappa`, this returns a **value-identical
  copy** of `ens.trMi_raw[i]`.
- Intended as a lightweight companion to [`trMiT`](@ref) for pipelines that keep
  raw (un-rescaled) traces.
- See also: [`trMiT`](@ref).
"""
function trMi_rawT(
    ens::Ensemble.EnsembleStruct{T}, 
    paramT::Ensemble.Params{T}, 
    i::Int
) where {T}
    # make everything type-stable in T
    half  = T(1//2) 
    oneT  = one(T)
    two   = T(2)
    three = T(3)

    # δm = 1/2 * (1/κ_a - 1/κ_T)
    dm = half * (oneT / ens.param.kappa - oneT / paramT.kappa)

    val = ens.trMi_raw[i]
    ret = Vector{T}(undef, length(val))

    ret[1] = val[1]  # always 1.0
    ret[2] = val[2] + dm * (val[3] + dm * (val[4] + dm * val[5]))
    ret[3] = val[3] + dm * (two * val[4] + three * dm * val[5])
    ret[4] = val[4] + three * dm * val[5]
    ret[5] = val[5]

    return ret
end

"""
    dS(
        ens::Ensemble.EnsembleStruct{T}, 
        paramT::Ensemble.Params{T}, 
        i::Int
    ) -> T

Compute the total action shift ``\\Delta S`` between an ensemble configuration and a target parameter set.
Currently only includes the fermionic part [`dSf`](@ref), which may involve mass (``\\kappa``) reweighting.

# Arguments
- [`ens::Ensemble.EnsembleStruct{T}`](@ref Deborah.Miriam.Ensemble.EnsembleStruct): Source ensemble with original parameters and trace data
- [`paramT::Ensemble.Params{T}`](@ref Deborah.Miriam.Ensemble.Params): Target parameters (includes `kappa`)
- `i::Int`: Gauge configuration index.

# Returns
- `T`: Total action difference ``\\Delta S`` (currently equal to `dSf`)
"""
function dS(
    ens::Ensemble.EnsembleStruct{T}, 
    paramT::Ensemble.Params{T}, 
    i::Int
) :: T where {T}
    return dSf(ens, paramT.csw, paramT.kappa, i)
end

"""
    dSf(
        ens::Ensemble.EnsembleStruct{T}, 
        cswT::T, 
        kappaT::T, 
        i::Int
    ) -> T

Compute the fermionic action shift from mass reweighting only.
Currently ignores clover term variation (`cswT`) and **delegates to [`dSf_m`](@ref)**.

# Arguments
- [`ens::Ensemble.EnsembleStruct{T}`](@ref Deborah.Miriam.Ensemble.EnsembleStruct): Ensemble data containing trace moments `trMi`
- `cswT::T`: Target clover coefficient **(unused)**
- `kappaT::T`: Target hopping parameter
- `i::Int`: Gauge configuration index.

# Returns
- [`dSf_m`](@ref): Estimated shift in fermionic action from mass difference only
"""
function dSf(
    ens::Ensemble.EnsembleStruct{T}, 
    cswT::T, 
    kappaT::T, 
    i::Int
) :: T where {T}
    return dSf_m(ens, kappaT, i)
end

"""
    dSf_m(
        ens::Ensemble.EnsembleStruct{T}, 
        kappaT::T, 
        i::Int
    ) -> T

Compute the derivative of the fermion action ``S_f`` with respect to the mass parameter ``m``,
expanded around the original ``\\kappa`` in the ensemble and evaluated at a target ``\\kappa``.

# Arguments
- [`ens::Ensemble.EnsembleStruct{T}`](@ref Deborah.Miriam.Ensemble.EnsembleStruct): Ensemble containing trace moments `trMi`
- `kappaT::T`: Target hopping parameter (``\\kappa``) for reweighting
- `i::Int`: Gauge configuration index.

# Returns
- `T`: Estimated value of ``\\Delta S_f`` for gauge configuration index `i`

---

# Notes

## Fermion Action Shift via Mass Taylor Expansion

Evaluates the mass-dependent shift in the fermion action ``\\Delta S_f``,  
expanded around the original ``\\kappa`` of the ensemble and evaluated at a target ``\\kappa``.

### Mass Shift

The hopping-mass shift is defined as:

```math
\\delta m = \\frac{1}{2} \\left( \\frac{1}{\\kappa} - \\frac{1}{\\kappa_{\\text{target}}} \\right)
```

### Taylor Expansion of Fermion Action

The action shift is approximated by:

```math
\\Delta S_f (\\kappa_{\\text{target}}) \\approx
  \\frac{(\\delta m)^1}{1} \\left[\\text{Tr}\\,M^{-1}\\right]\\left(\\kappa\\right)
+ \\frac{(\\delta m)^2}{2} \\left[\\text{Tr}\\,M^{-2}\\right]\\left(\\kappa\\right)
+ \\frac{(\\delta m)^3}{3} \\left[\\text{Tr}\\,M^{-3}\\right]\\left(\\kappa\\right)
+ \\frac{(\\delta m)^4}{4} \\left[\\text{Tr}\\,M^{-4}\\right]\\left(\\kappa\\right)
```

Each trace term is obtained from the stored vector `ens.trMi[i]` in the ensemble:

```math
\\texttt{trMi[i]} = \\left\\{ \\;
12 N_f V ,\\;
\\text{Tr}\\, M^{-1} ,\\;
\\text{Tr}\\, M^{-2} ,\\;
\\text{Tr}\\, M^{-3} ,\\;
\\text{Tr}\\, M^{-4}
\\;\\right\\}
```

In this context, we assume that all ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` have already been properly rescaled. (For the rescaling process in [`Deborah.Miriam`](@ref), see [`Deborah.Miriam.MultiEnsembleLoader.generate_trMi_vector`](@ref) or [`Deborah.Miriam.Cumulants.calc_Q`](@ref). Note, however, that the rescaling convention in [`Deborah.Esther`](@ref) is slightly different; to check it, see [`Deborah.Esther.TraceRescaler.rescale_trace`](@ref).)
"""
function dSf_m(
    ens::Ensemble.EnsembleStruct{T}, 
    kappaT::T, 
    i::Int
) where {T}
    val = ens.trMi[i]
    dm = 0.5 * (1.0 / ens.param.kappa - 1.0 / kappaT)
    return dm * (val[2] + dm * (val[3] / 2.0 + dm * (val[4] / 3.0 + dm * val[5] / 4.0)))
end

"""
    flatten_trace_blocks(
        trace_dict::Dict{String, NamedTuple},
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}, Vector{UInt8}, Vector{UInt8}, Vector{Int}}

Flatten per-prefix trace blocks into unified arrays (sorted by configuration) suitable for [`Deborah.Miriam.Ensemble.EnsembleStruct`](@ref).

# Arguments
- `trace_dict::Dict{String, NamedTuple}`:
  Mapping from trace prefix (e.g., `"Y_tr"`, `"Y_bc"`, `"Y_ul"`, `"YP_bc"`, `"YP_ul"`; `"Y_info"` is ignored)
  to a `NamedTuple` with fields:
  - `.values::Vector{Vector{Float64}}` — per-configuration rescaled trace vectors (length 4 or 5).
  - `.values_raw::Vector{Vector{Float64}}` — per-configuration un-rescaled raw trace vectors (length 4 or 5).
  - `.conf_nums::Vector{Int}` — per-configuration IDs.
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `Vector{Vector{Float64}}` (`trMi`):
  Flattened rescaled trace rows (each of length 5).
  If an input row has length 4, a placeholder fifth component is appended as `val[1]^5`
  (do not use this placeholder as a physical observable).
- `Vector{Vector{Float64}}` (`trMi_raw`):
  Flattened un-rescaled raw trace rows (each of length 5).
  If an input row has length 4, a placeholder fifth component is appended as `val_raw[1]^5`.
- `Vector{UInt8}` (`source_tags`):
  Fine-grained source code per row
  - `0` → `"Y_tr"`
  - `1` → `"Y_bc"`, `"YP_bc"`
  - `2` → `"Y_ul"`, `"YP_ul"`
- `Vector{UInt8}` (`secondary_tags`):
  Coarse class tag per row
  - `0` → `"Y_lb"` (originated from `"Y_tr"`, `"Y_bc"`, `"YP_bc"`)
  - `1` → `"Y_ul"` (originated from `"Y_ul"`, `"YP_ul"`)
- `Vector{Int}` (`conf_nums`):
  Configuration numbers aligned with the outputs; rows are globally sorted by configuration.

# Notes
- Concatenates all prefixes present in `trace_dict` except `"Y_info"`, then sorts by configuration number.
- Replacement prefixes (`"YP_*"`) map to the same source/secondary classes as their non-replacement counterparts.
- `.values`, `.values_raw`, and `.conf_nums` must have equal lengths per prefix; mismatches and unknown prefixes raise errors via [`Deborah.Sarah.JobLoggerTools.error_benji`](@ref).
"""
function flatten_trace_blocks(
    trace_dict::Dict{String, NamedTuple},
    jobid::Union{Nothing, String}=nothing
) :: Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}, Vector{UInt8}, Vector{UInt8}, Vector{Int}}

    entries = Vector{NamedTuple{(:conf, :val, :raw, :tag, :sec), Tuple{Int, Vector{Float64}, Vector{Float64}, UInt8, UInt8}}}()

    for (prefix, nt) in trace_dict
        if prefix == "Y_info"
            continue
        end

        tag_id = source_to_tag(prefix, jobid)
        sec_id = secondary_source_to_tag(prefix, jobid)

        values = nt.values
        values_raw = nt.values_raw
        confs  = nt.conf_nums
        JobLoggerTools.assert_benji(length(values) == length(values_raw) == length(confs), "length(values) != length(values_raw) != length(confs)", jobid)

        for i in eachindex(values)
            val = values[i]
            val_raw = values_raw[i]
            if length(val) == 4
                extended = copy(val)
                push!(extended, val[1]^5)  # placeholder for missing 5th
                val = extended
                extended_raw = copy(val_raw)
                push!(extended_raw, val_raw[1]^5)  # placeholder for missing 5th
                val_raw = extended_raw
            elseif length(val) != 5
                JobLoggerTools.error_benji("Invalid trMi length = $(length(val)) at conf = $(confs[i]), prefix = $prefix", jobid)
            end
            push!(entries, (conf = confs[i], val = val, raw = val_raw, tag = tag_id, sec = sec_id))
        end
    end

    sort!(entries; by = x -> x.conf)

    trMi           = [e.val  for e in entries]
    trMi_raw       = [e.raw  for e in entries]
    source_tags    = UInt8[e.tag for e in entries]
    secondary_tags = UInt8[e.sec for e in entries]
    conf_nums      = [e.conf for e in entries]

    return trMi, trMi_raw, source_tags, secondary_tags, conf_nums
end

"""
    source_to_tag(
        prefix::String,
        jobid::Union{Nothing, String}=nothing
    ) -> UInt8

Map a trace prefix string to its source tag (compact `UInt8` code).

# Arguments
- `prefix::String`: Trace data prefix (e.g., `"Y_tr"`, `"Y_bc"`, `"YP_ul"`).
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `UInt8`: Encoded source tag
    - `0` → `"Y_tr"`
    - `1` → `"Y_bc"`, `"YP_bc"`
    - `2` → `"Y_ul"`, `"YP_ul"`

# Throws
- Raises an error via [`Deborah.Sarah.JobLoggerTools.error_benji`](@ref) if `prefix` is unknown.

# Notes
- Replacement prefixes (`"YP_bc"`, `"YP_ul"`) are mapped to the same source
  classes as their non-replacement counterparts.
"""
function source_to_tag(
    prefix::String,
    jobid::Union{Nothing, String}=nothing
)::UInt8
    if     prefix in ["Y_tr"]
        return 0
    elseif prefix in ["Y_bc", "YP_bc"]
        return 1
    elseif prefix in ["Y_ul", "YP_ul"]
        return 2
    else
        JobLoggerTools.error_benji("Unknown trace prefix: $prefix", jobid)
    end
end

"""
    secondary_source_to_tag(
        prefix::String,
        jobid::Union{Nothing, String}=nothing
    ) -> UInt8

Map a trace prefix to a **secondary** class tag used for coarse grouping.

# Arguments
- `prefix::String`: Trace data prefix (e.g., `"Y_tr"`, `"Y_bc"`, `"YP_ul"`).
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `UInt8`: Encoded secondary tag
    - `0` → `"Y_lb"` (for `"Y_tr"`, `"Y_bc"`, `"YP_bc"`)
    - `1` → `"Y_ul"` (for `"Y_ul"`, `"YP_ul"`)

# Throws
- Raises an error via [`Deborah.Sarah.JobLoggerTools.error_benji`](@ref) if `prefix` is unknown.
"""
function secondary_source_to_tag(
    prefix::String,
    jobid::Union{Nothing, String}=nothing
)::UInt8
    if     prefix in ["Y_tr", "Y_bc", "YP_bc"]
        return 0  # Y_lb
    elseif prefix in ["Y_ul", "YP_ul"]
        return 1  # Y_ul
    else
        JobLoggerTools.error_benji("Unknown trace prefix (secondary): $prefix", jobid)
    end
end

"""
    build_ensemble_array_from_trace(
        grouped_data::Dict{String, Dict{String, NamedTuple}},
        cfg::TOMLConfigMiriam.FullConfigMiriam,
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{Ensemble.EnsembleArray{Float64}, Vector{String}}

Build an [`EnsembleArray{Float64}`](@ref Deborah.Miriam.Ensemble.EnsembleArray) (collection of [`EnsembleStruct`](@ref Deborah.Miriam.Ensemble.EnsembleStruct)) and the
corresponding ensemble-name list from grouped trace data.

# Arguments
- `grouped_data::Dict{String, Dict{String, NamedTuple}}`:
  Outer dictionary keyed by ensemble name (e.g., `"L8T4b1.60k13570"`).  
  Each inner dictionary maps trace prefixes (e.g., `"Y_tr"`, `"Y_bc"`, `"Y_ul"`,
  `"YP_bc"`, `"YP_ul"`; `"Y_info"` is ignored) to a `NamedTuple` with fields:
  - `values::Vector{Vector{Float64}}`      — rows of rescaled ``\\text{Tr} \\, M^{-n}`` (length 4 or 5)
  - `values_raw::Vector{Vector{Float64}}`  — rows of un-rescaled raw ``\\text{Tr} \\, M^{-n}`` (length 4 or 5)
  - `conf_nums::Vector{Int}`               — configuration numbers
- [`cfg::TOMLConfigMiriam.FullConfigMiriam`](@ref Deborah.Miriam.TOMLConfigMiriam.FullConfigMiriam):
  Configuration used to determine the ensemble set and to build dummy parameters.
- `jobid::Union{Nothing, String}`:
  Optional job identifier for structured logging.

# Returns
- `(ensemble_array, key_list)`:
  - [`ensemble_array::EnsembleArray{Float64}`](@ref Deborah.Miriam.Ensemble.EnsembleArray):
    Holds one [`Deborah.Miriam.Ensemble.EnsembleStruct`](@ref) per ensemble. Each struct includes:
    - `trMi` (rescaled, rows of length 5; a placeholder may be appended by the loader),
    - `trMi_raw` (un-rescaled, rows of length 5; a placeholder may be appended by the loader),
    - `source_tags` (`0`: `Y_tr`, `1`: `Y_bc/YP_bc`, `2`: `Y_ul`/`YP_ul`),
    - `secondary_tags` (`0`: `Y_lb`, `1`: `Y_ul`), and aligned `conf_nums`.
  - `key_list::Vector{String}`:
    The list of ensemble names sorted lexicographically
    (as in `sort(cfg.data.ensembles)`).

# Notes
- Internally calls [`flatten_trace_blocks`](@ref) which concatenates
  all prefixes (except `"Y_info"`), appends a placeholder 5th component when
  needed for both rescaled and un-rescaled traces, and sorts rows by configuration number.
- Missing or malformed inputs are reported via [`Deborah.Sarah.JobLoggerTools.error_benji`](@ref) /
  [`Deborah.Sarah.JobLoggerTools.warn_benji`](@ref) in the upstream loader/flattening routines.
"""
function build_ensemble_array_from_trace(
    grouped_data::Dict{String, Dict{String, NamedTuple}},
    cfg::TOMLConfigMiriam.FullConfigMiriam,
    jobid::Union{Nothing, String}=nothing
)::Tuple{Ensemble.EnsembleArray{Float64}, Vector{String}}

    ens_list = Ensemble.EnsembleStruct{Float64}[]
    key_list = String[]

    for ens in sort(cfg.data.ensembles)
        trace_dict = grouped_data[ens]

        trMi, trMi_raw, source_tags, secondary_tags, conf_nums = flatten_trace_blocks(trace_dict, jobid)

        nconf = length(trMi)
        dummy_param = build_dummy_param(cfg, ens, jobid)
        dummy_vec = zeros(Float64, nconf)

        ens_obj = Ensemble.EnsembleStruct(
            nconf,
            0.0,
            dummy_param,
            dummy_vec,
            dummy_vec,
            dummy_vec,
            ComplexF64[0.0 + 0.0im for _ in 1:nconf],
            trMi,
            trMi_raw,
            source_tags,
            secondary_tags,
            conf_nums
        )

        push!(ens_list, ens_obj)
        push!(key_list, ens)
    end

    return Ensemble.EnsembleArray(ens_list), key_list
end

"""
    build_dummy_param(
        cfg::TOMLConfigMiriam.FullConfigMiriam, 
        ens::String,
        jobid::Union{Nothing, String}=nothing
    ) -> Params{Float64}

Construct a dummy `Params{Float64}` object from config and ensemble name.

# Arguments
- [`cfg::TOMLConfigMiriam.FullConfigMiriam`](@ref Deborah.Miriam.TOMLConfigMiriam.FullConfigMiriam): Configuration object containing lattice input metadata.
- `ens::String`: Ensemble name string containing encoded ``\\kappa`` (must match `k13580` format).
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- [`Params{Float64}`](@ref Deborah.Miriam.Ensemble.Params): Lattice parameter object with extracted ``\\kappa`` value from ensemble name.
"""
function build_dummy_param(
    cfg::TOMLConfigMiriam.FullConfigMiriam, 
    ens::String,
    jobid::Union{Nothing, String}=nothing
)::Ensemble.Params{Float64}

    # Extract kappa from ensemble name using regex "kXXXXX"
    m = match(r"k(\d{5})", ens)
    if m === nothing
        JobLoggerTools.error_benji("Could not parse kappa from ensemble name: $ens", jobid)
    end
    kappa_raw = m.captures[1]
    kappa = parse(Float64, "0." * kappa_raw)

    return Ensemble.Params(
        cfg.input_meta.ns, 
        cfg.input_meta.nt, 
        cfg.input_meta.nf, 
        cfg.input_meta.beta, 
        cfg.input_meta.csw, 
        kappa
    )
end

end  # module EnsembleUtils