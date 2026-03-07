# ============================================================================
# src/Miriam/MultiEnsembleLoader.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module MultiEnsembleLoader

import ..StatsBase

import ..Sarah.JobLoggerTools
import ..Sarah.DataLoader
import ..TOMLConfigMiriam
import ..PathConfigBuilderMiriam

"""
    parse_kappa_from_ensemble_name(
        ens::String,
        jobid::Union{Nothing, String}=nothing
    ) -> Float64

Extract the ``\\kappa`` value from an ensemble name string.

# Arguments
- `ens::String`: Ensemble name containing a pattern like `kXXXXX` (e.g., `"L8T4b1.60k13580"`)
- `jobid::Union{Nothing, String}`: Optional job identifier for logging.

# Returns
- `Float64`: Parsed ``\\kappa`` value, interpreted as `"0.XXXXX"`

# Throws
- `ErrorException` if the pattern `k\\d+` is not found in the input string.
"""
function parse_kappa_from_ensemble_name(
    ens::String,
    jobid::Union{Nothing, String}=nothing
)::Float64
    m = match(r"k(\d+)", ens)
    m === nothing && JobLoggerTools.error_benji("Failed to parse kappa from ensemble name: $ens", jobid)
    return parse(Float64, "0." * m.captures[1])
end

"""
    generate_trMi_vector(
        trM_raw::Vector{Float64},
        nf::Int,
        ns::Int,
        nt::Int,
        kappa::Float64
    ) -> Vector{Float64}

Convert raw trace values into scaled traces (`trMi`) for one configuration.

# Arguments
- `trM_raw::Vector{Float64}`: Unscaled trace values of length 4 (`raw[1]` to `raw[4]`)
- `nf::Int`: Number of quark f`lavors
- `ns::Int`: Spatial lattice extent
- `nt::Int`: Temporal lattice extent
- `kappa::Float64`: Hopping parameter

# Returns
- `Vector{Float64}` of length 5, representing scaled trace moments:
  - `trMi[1] = ` ``12 \\, N_{\\text{f}} \\, V`` (``V = N_S^3 \\times N_T``)
  - `trMi[2]` to `trMi[5]`: scaled raw values with κ weighting

# Notes
- The scaling factor follows the rule:  
  `trMi[j+1] = trM_raw[j]` ``\\times ( 12 \\, N_{\\text{f}} \\, V ) \\, ( 2 \\, \\kappa )^j`` for ``j = 1,2,3,4``

- In the end, we have
```math
\\texttt{trMi[i]} = \\left\\{ \\;
12 N_f V ,\\;
2\\kappa N_f\\, \\text{Tr}\\, M^{-1} ,\\;
(2\\kappa)^2 N_f\\, \\text{Tr}\\, M^{-2} ,\\;
(2\\kappa)^3 N_f\\, \\text{Tr}\\, M^{-3} ,\\;
(2\\kappa)^4 N_f\\, \\text{Tr}\\, M^{-4}
\\;\\right\\}
```
"""
function generate_trMi_vector(
    trM_raw::Vector{Float64},
    nf::Int,
    ns::Int,
    nt::Int,
    kappa::Float64
)::Vector{Float64}
    V = ns^3 * nt
    factor = 12.0 * nf * V
    trMi = Vector{Float64}(undef, 5)
    trMi[1] = factor
    for j in 1:4
        factor *= 2.0 * kappa
        trMi[j+1] = trM_raw[j] * factor
    end
    return trMi
end

"""
    generate_trMi_raw_vector(
        trM_raw::Vector{Float64}
    ) -> Vector{Float64}

Construct a length-5 vector containing a dummy scaling entry and
the unscaled trace values (`trMi_raw`) for one configuration.

# Arguments
- `trM_raw::Vector{Float64}`: Unscaled trace values of length 4 (`raw[1]` to `raw[4]`).

# Returns
- `Vector{Float64}` of length 5:
  - `trMi_raw[1] = 1.0` (placeholder factor)
  - `trMi_raw[2]` to `trMi_raw[5]` = direct copy of `trM_raw[1:4]` (no rescaling).

# Notes
- This utility is a companion to [`generate_trMi_vector`](@ref) but omits
  all ``\\kappa`` and volume rescaling. It simply places the raw values alongside
  a constant leading entry.
"""
function generate_trMi_raw_vector(
    trM_raw::Vector{Float64}
)::Vector{Float64}
    trMi_raw = Vector{Float64}(undef, 5)
    trMi_raw[1] = 1.0
    for j in 1:4
        trMi_raw[j+1] = trM_raw[j]
    end
    return trMi_raw
end

"""
    load_grouped_trace_data(
        cfg::TOMLConfigMiriam.FullConfigMiriam,
        paths::PathConfigBuilderMiriam.MiriamPathConfig;
        jobid::Union{Nothing, String}=nothing,
        replace_bc::Bool=false,
        replace_ul::Bool=false,
        take_only_lb::Bool=false,
    ) -> Dict{String, Dict{String, NamedTuple}}

Load and organize trace data from disk into a nested dictionary structure, grouped by
ensemble and trace type.

# Arguments
- [`cfg::TOMLConfigMiriam.FullConfigMiriam`](@ref Deborah.Miriam.TOMLConfigMiriam.FullConfigMiriam): Parsed [`TOML`](https://toml.io/en/) configuration containing labels, training
  targets, and bootstrap setup.
- [`paths::PathConfigBuilderMiriam.MiriamPathConfig`](@ref Deborah.Miriam.PathConfigBuilderMiriam.MiriamPathConfig): Auto-generated paths for accessing trace data on disk.

# Keyword Arguments
- `jobid::Union{Nothing, String}=nothing`: Optional job identifier to distinguish
  multiple runs.
- `replace_bc::Bool=false`: If `true`, substitute `"YP_bc"` in place of `"Y_bc"` for
  file reads and keys.
- `replace_ul::Bool=false`: If `true`, substitute `"YP_ul"` in place of `"Y_ul"` for
  file reads and keys.
- `take_only_lb::Bool=false`: If `true`, **only** pick labeled-side traces
  (`"Y_tr"` and `"Y_bc"` or `"YP_bc"` depending on `replace_bc`) and **skip all**
  unlabeled-side traces (`"Y_ul"`/`"YP_ul"`) regardless of `replace_ul`.

# Behavior
- When `take_only_lb = false` (default):
  - The loader reads, in order: `"Y_tr"`,
    then (`"YP_bc"` if `replace_bc` else `"Y_bc"`),
    then (`"YP_ul"` if `replace_ul` else `"Y_ul"`).
- When `take_only_lb = true`:
  - The loader reads **only** `"Y_tr"` and (`"YP_bc"` if `replace_bc` else `"Y_bc"`),
    and **does not** read any `Y_UL` files even if `replace_ul = true`.
- Per-prefix data are sorted by configuration number before being stored.
- Missing files are skipped with a warning.

# Returns
- `Dict{String, Dict{String, NamedTuple}}`: A nested dictionary:
  - **Outer keys**: Ensemble names (e.g., `"L8T4b1.60k13570"`).
  - **Inner keys**: Trace labels (e.g., `"Y_tr"`, `"Y_bc"`, `"Y_ul"`; or `"YP_bc"`,
    `"YP_ul"` when replacements are enabled and not suppressed by `take_only_lb`).
  - **Values**: `NamedTuple` with fields:
    - `values::Vector{Vector{Float64}}`:
      **Scaled** `trMi` vectors of length **5** per configuration
      (``\\left[ 12 \\, N_{\\text{f}} \\, V \\,,\\; \\left( 2 \\, \\kappa \\right)^1 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\,, \\cdots \\,, \\; \\left( 2 \\, \\kappa \\right)^4 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-4} \\right]``).
    - `values_raw::Vector{Vector{Float64}}`:
      **Unscaled** `trMi_raw` vectors of length **5** per configuration
      (typically ``\\left[ 1.0 \\,,\\; \\text{Tr} \\, M^{-1} \\,,\\; \\cdots ,\\; \\text{Tr} \\, M^{-4} \\right]``).
    - `indices::Vector{Int}`:
      Sequential indices `1:N` aligned with `values`.
    - `conf_nums::Vector{Int}`:
      Configuration numbers aligned with `values`.

Additionally, each ensemble contains a `"Y_info"` entry:
- `"Y_info" => ( ... )` with fields:
  - `tags::Vector{String}`:
    Row-wise original tags such as `"LB-TR[j]"`, `"LB-BC[j]"`, or `"UL[j]"`.
  - `secondary_tags::Vector{String}`:
    Row-wise **secondary** tags for coarse grouping:
    `"Y_lb"` when the row came from `"Y_tr"`, `"Y_bc"`, or `"YP_bc"`;
    `"Y_ul"` when the row came from `"Y_ul"` or `"YP_ul"`.
    (If `take_only_lb=true`, this vector will naturally contain only `"Y_lb"`.)
  - `indices::Vector{Int}`:
    Source indices `j` (i.e., which `TrMi` slot contributed).
  - `conf_nums::Vector{Int}`:
    Corresponding configuration numbers.
  - `values::Vector{Float64}`:
    (Reserved/empty placeholder in this loader.)
  - `values_raw::Vector{Float64}`:
    (Reserved/empty placeholder in this loader.)
"""
function load_grouped_trace_data(
    cfg::TOMLConfigMiriam.FullConfigMiriam,
    paths::PathConfigBuilderMiriam.MiriamPathConfig;
    jobid::Union{Nothing, String}=nothing,
    replace_bc::Bool=false,
    replace_ul::Bool=false,
    take_only_lb::Bool=false,
)::Dict{String, Dict{String, NamedTuple}}
    grouped_data = Dict{String, Dict{String, NamedTuple}}()

    for ens in cfg.data.ensembles
        grouped_data[ens] = Dict{String, NamedTuple}()

        # --- Metadata to be stored under "Y_info" ---
        tag_list = String[]     # original row-wise tags: "LB-TR[j]", "LB-BC[j]", "UL[j]"
        conf_list = Int[]
        idx_list = Int[]
        sec_tag_list = String[] # NEW: secondary row-wise tags: "Y_lb" or "Y_ul"

        # --- Decide which prefixes (files) to load ---
        # If take_only_lb=true, we only load Y_tr and Y_bc (or YP_bc if replace_bc=true),
        # and we entirely skip any UL (Y_ul/YP_ul) regardless of replace_ul flag.
        prefix_list =
            if take_only_lb
                ["Y_tr", (replace_bc ? "YP_bc" : "Y_bc")]
            else
                ["Y_tr",
                 (replace_bc ? "YP_bc" : "Y_bc"),
                 (replace_ul ? "YP_ul" : "Y_ul")]
            end

        for prefix in prefix_list
            raw_per_conf = Dict{Int, Vector{Float64}}()

            # Select encoded vs decoded input token list
            inputs = cfg.data.use_abbreviation ? paths.encoded_inputs : paths.decoded_inputs

            for (j, trM) in enumerate(inputs)
                model = paths.model_tags[j]
                base_dir = joinpath(cfg.data.location, "$(cfg.data.analysis_header)_$(ens)")
                target_dir = joinpath(base_dir, "$(cfg.data.analysis_header)_$(ens)_$(trM)_$(model)")
                base_name = "$(ens)_$(trM)_$(model)_LBP_$(cfg.data.LBP)_TRP_$(cfg.data.TRP)"
                path = joinpath(target_dir, base_name, "$(prefix)_$(base_name).dat")

                if isfile(path)
                    raw = DataLoader.try_multi_readdlm(path)
                    for row in eachrow(raw)
                        conf = Int(row[2])
                        val = row[1]

                        # Initialize storage for this configuration if needed
                        if !haskey(raw_per_conf, conf)
                            raw_per_conf[conf] = fill(NaN, 4)
                        end
                        raw_per_conf[conf][j] = val

                        # ---- Original per-row tag (kept as-is) ----
                        tag = prefix == "Y_tr"                             ? "LB-TR[$j]" :
                              (prefix == "Y_bc" || prefix == "YP_bc")      ? "LB-BC[$j]" :
                              (prefix == "Y_ul" || prefix == "YP_ul")      ? "UL[$j]"    :
                              JobLoggerTools.error_benji("Unexpected prefix: $prefix", jobid)
                        push!(tag_list, tag)

                        # ---- NEW: Secondary per-row tag ("Y_lb" or "Y_ul") ----
                        sec_tag = (prefix == "Y_tr" || prefix == "Y_bc" || prefix == "YP_bc") ? "Y_lb" :
                                  (prefix == "Y_ul" || prefix == "YP_ul")                     ? "Y_ul" :
                                  JobLoggerTools.error_benji("Unexpected prefix for sec_tag: $prefix", jobid)
                        push!(sec_tag_list, sec_tag)

                        # Trace meta
                        push!(conf_list, conf)
                        push!(idx_list, j)
                    end
                else
                    JobLoggerTools.warn_benji("Missing file: $path", jobid)
                end
            end

            # Sort trace data by configuration number
            confval_pairs = collect(pairs(raw_per_conf))
            sort!(confval_pairs; by = first)

            # Empty or fake data filter
            if isempty(confval_pairs)
                JobLoggerTools.warn_benji("Empty or missing $prefix data for ensemble $ens — skipping.", jobid)
                continue
            end

            kappa = parse_kappa_from_ensemble_name(ens, jobid)

            values = Vector{Vector{Float64}}()
            values_raw = Vector{Vector{Float64}}()
            conf_nums = Int[]

            for (conf, trM_raw) in confval_pairs
                JobLoggerTools.assert_benji(length(trM_raw) == 4, "Expected 4 trace values for conf $conf", jobid)
                rescaled = generate_trMi_vector(trM_raw, cfg.input_meta.nf, cfg.input_meta.ns, cfg.input_meta.nt, kappa)
                not_rescaled = generate_trMi_raw_vector(trM_raw)
                push!(values, rescaled)
                push!(values_raw, not_rescaled)
                push!(conf_nums, conf)
            end

            indices = collect(1:length(values))

            # Store grouped data per prefix (e.g., "Y_tr", "Y_bc", "Y_ul")
            grouped_data[ens][prefix] = (
                values = values,
                values_raw = values_raw,
                indices = indices,
                conf_nums = conf_nums
            )
        end

        # Store auxiliary info for Y_info (used later in tagging and plotting)
        # Note: when take_only_lb=true, 'secondary_tags' will naturally be all "Y_lb".
        grouped_data[ens]["Y_info"] = (
            values = Float64[],        # kept empty as before
            values_raw = Float64[],    # kept empty as before
            indices = idx_list,        # per-row trM index j
            conf_nums = conf_list,     # per-row conf number
            tags = tag_list,           # original tags
            secondary_tags = sec_tag_list
        )
    end

    return grouped_data
end

end  # module MultiEnsembleLoader