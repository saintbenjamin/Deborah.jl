# ============================================================================
# src/RebekahMiriam/SummaryLoaderRebekahMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SummaryLoaderRebekahMiriam

import ..TOML

import ..Sarah.JobLoggerTools

"""
    extract_all_observables_from_line(
        lines::Vector{String}, 
        keyword::String,
        jobid::Union{Nothing, String}=nothing
    ) -> Dict{Symbol, Float64}

Parse a single summary line ending with the specified keyword and extract observable values.

The line must contain 12 numeric fields followed by a keyword, corresponding to the format:
`kappa_t_avg, kappa_t_err, cond_avg, cond_err, ..., bind_err  keyword`

# Arguments
- `lines`: Vector of strings (typically from `readlines(filename)`).
- `keyword`: Target keyword to match at the end of a line.
- `jobid::Union{Nothing, String}`: Optional job ID for contextual logging.

# Returns
A dictionary mapping symbols like `:cond_avg`, `:cond_err`, etc., to their parsed `Float64` values.

# Errors
Throws an error if no matching line is found.
"""
function extract_all_observables_from_line(
    lines::Vector{String}, 
    keyword::String,
    jobid::Union{Nothing, String}=nothing
)
    for line in lines
        fields = split(strip(line))
        if !isempty(fields) && endswith(fields[end], keyword)
            numbers = parse.(Float64, fields[1:end-1])
            return Dict(
                :kappa_t_avg => numbers[1],
                :kappa_t_err => numbers[2],
                :cond_avg    => numbers[3],
                :cond_err    => numbers[4],
                :susp_avg    => numbers[5],
                :susp_err    => numbers[6],
                :skew_avg    => numbers[7],
                :skew_err    => numbers[8],
                :kurt_avg    => numbers[9],
                :kurt_err    => numbers[10],
                :bind_avg    => numbers[11],
                :bind_err    => numbers[12],
            )
        end
    end
    JobLoggerTools.error_benji("No line ending with '$keyword' found.", jobid)
end

"""
    load_miriam_summary(
        work::String,
        analysis_ensemble::String,
        cumulant_name::String,
        overall_name::String,
        labels::Vector{String},
        trains::Vector{String},
        keywords::Vector{String},
        filetags::Vector{Symbol},
        fields::Vector{Symbol},
        jobid::Union{Nothing, String}=nothing
    ) -> Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}}

Load all observable summaries from `.dat` files for each (`label`, `train`, `filetag`, `keyword`) combination.

Each file is expected to contain labeled lines that end with a keyword (e.g., `"susp"`, `"skew"`), and each such line encodes 12 observable values.

# Arguments
- `work`: Base working directory.
- `analysis_ensemble`: Name of the ensemble (e.g., `"L8T4b1.60"`).
- `cumulant_name`: Name of the observable bundle.
- `overall_name`: Global identifier for filename.
- `labels`: List of labeled set percentages.
- `trains`: List of training set percentages.
- `keywords`: Keywords indicating the interpolation criterion (e.g., `"susp"`).
- `filetags`: Tags for different prediction sources (e.g., `:RWBS`, `:RWP1`).
- `fields`: Observable names (e.g., `:cond`, `:skew`, etc.).
- `jobid::Union{Nothing, String}`: Optional job ID for contextual logging.

# Returns
A nested dictionary indexed by `(field, :avg|:err, filetag, keyword)` mapping to a matrix of shape `(length(labels), length(trains))`.
"""
function load_miriam_summary(
    work::String,
    analysis_ensemble::String,
    cumulant_name::String,
    overall_name::String,
    labels::Vector{String},
    trains::Vector{String},
    keywords::Vector{String},
    filetags::Vector{Symbol},
    fields::Vector{Symbol},
    jobid::Union{Nothing, String}=nothing
)::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64, 2}}

    summary = Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64, 2}}()

    for filetag in filetags, field in fields, keyword in keywords
        summary[(field, :avg, filetag, keyword)] = zeros(length(labels), length(trains))
        summary[(field, :err, filetag, keyword)] = zeros(length(labels), length(trains))
    end

    for (i, label) in enumerate(labels)
        for (j, train) in enumerate(trains)
            location = "$(work)/$(analysis_ensemble)/$(cumulant_name)/$(overall_name)_LBP_$(label)_TRP_$(train)"

            for filetag in filetags
                file = "$(filetag)_$(overall_name)_LBP_$(label)_TRP_$(train).dat"
                filename = "$(location)/$(file)"
                if isfile(filename)
                    lines = readlines(filename)

                    for keyword in keywords
                        try
                            obs = extract_all_observables_from_line(lines, keyword, jobid)
                            for (key, val) in obs
                                strkey = string(key)
                                if endswith(strkey, "_avg")
                                    base = Symbol(replace(strkey, "_avg" => ""))
                                    summary[(base, :avg, filetag, keyword)][i, j] = val
                                elseif endswith(strkey, "_err")
                                    base = Symbol(replace(strkey, "_err" => ""))
                                    summary[(base, :err, filetag, keyword)][i, j] = val
                                else
                                    JobLoggerTools.warn_benji("Unexpected key format: $key", jobid)
                                end
                            end
                        catch e
                            JobLoggerTools.warn_benji("Keyword '$keyword' not found in $filename", jobid)
                        end
                    end
                else
                    JobLoggerTools.warn_benji("Missing file: $filename", jobid)
                end
            end
        end
    end

    return summary
end

"""
    derive_kappa_list(
        ensembles::Vector{String}, 
        multi_ensemble::String
    ) -> Vector{String}

Derive a vector of ``\\kappa`` tokens (e.g., `["13570","13575",...]`) from full ensemble
names by stripping the common prefix `multi_ensemble` and a leading `'k'`.

# Example
- `multi_ensemble = "L8T4b1.60"`
- `ensembles = ["L8T4b1.60k13570","L8T4b1.60k13575"]`
- `=> ["13570","13575"]`
"""
function derive_kappa_list(
    ensembles::Vector{String}, 
    multi_ensemble::String
)::Vector{String}
    out = String[]
    for ens in ensembles
        s = replace(ens, multi_ensemble => "")
        s = replace(s, r"^k" => "")  # strip one leading 'k' if present
        push!(out, s)
    end
    return out
end

"""
    load_miriam_summary_for_measurement(
        work::String,
        analysis_ensemble::String,
        group_name::String,
        overall_name::String,
        labels::Vector{String},
        trains::Vector{String},
        ensembles::Vector{String},   # full ensemble names
        multi_ensemble::String,      # common prefix to strip
        filetags::Vector{Symbol},    # e.g. [:T_BS, :T_JK, :T_P1, :T_P2] or [:Q_BS, :Q_JK, :Q_P1, :Q_P2]
        fields::Vector{Symbol},      # e.g. [:kappa, :trM1, :trM2, :trM3, :trM4] or [:kappa_t, :Q1, :Q2, :Q3, :Q4]
        jobid::Union{Nothing,String}=nothing
    ) -> Tuple{
        Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
        Vector{String}
    }

Load measurement results for all provided `filetags` (source-agnostic; no orig/pred split).
For each (`label`, `train`, `tag`), this scans one `.dat` file row-wise and extracts observable
`(avg, err)` values for all kappas implied by `ensembles`/`multi_ensemble`.

Directory and file layout:
    `<work>/<analysis_ensemble>/<group_name>/<overall_name>_LBP_<label>_TRP_<train>/<tag>_<overall_name>_LBP_<label>_TRP_<train>.dat`

Expected row format inside each file:
    `kappa  (val_key2  err_key2)  (val_key3  err_key3) ...`

Where `fields = [:kappa or :kappa_t, :obs2, :obs3, ...]`.
Only `fields[2:end]` are stored as `(avg, err)` in the returned dictionary.

# Arguments
- `labels`, `trains`: define the `(row, col)` axes for the output matrices.
- `ensembles`, `multi_ensemble`: used to derive `kappa_list::Vector{String}` such as `["13570", ...]`.
- `filetags`: complete set of measurement tags to load (e.g., `[:T_BS,:T_JK,:T_P1,:T_P2]`).
- `fields`: first element must be `:kappa` or `:kappa_t`; the rest are observables to store.

# Returns
A tuple:
1. `Dict{(field, stat, tag, kappa_str) => Matrix{Float64}}` of size `(length(labels), length(trains))`,
   where `field` ``\\in`` `fields[2:end]`, `stat` ``\\in`` `(:avg, :err)`, `tag` ``\\in`` `filetags`, `kappa_str` is a token like `"13580"`.
2. `kappa_list::Vector{String}` in the same order used to fill the dictionary.

# Example
```julia
summary_meas, kappa_list = load_miriam_summary_for_measurement(
    work, analysis_ensemble, group_name, overall_name,
    labels, trains, ensembles, multi_ensemble,
    filetags, fields
)
```
"""
function load_miriam_summary_for_measurement(
    work::String,
    analysis_ensemble::String,
    group_name::String,
    overall_name::String,
    labels::Vector{String},
    trains::Vector{String},
    ensembles::Vector{String},
    multi_ensemble::String,
    filetags::Vector{Symbol},
    fields::Vector{Symbol},
    jobid::Union{Nothing,String}=nothing
)::Tuple{Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}}, Vector{String}}

    # ---- derive kappa tokens from ensembles ----
    kappa_list = derive_kappa_list(ensembles, multi_ensemble)

    # ---- sanity & setup ----
    if isempty(filetags)
        JobLoggerTools.error_benji("filetags must not be empty.", jobid)
    end
    if length(fields) < 2
        JobLoggerTools.error_benji("fields must include at least kappa and one observable.", jobid)
    end
    kappa_key = fields[1]            # :kappa or :kappa_t (semantic only)
    obs_keys  = fields[2:end]        # observables to store

    nLB = length(labels)
    nTR = length(trains)

    # Initialize dictionary with zero matrices
    summary = Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}}()
    for tag in filetags, kappa_str in kappa_list, field in obs_keys
        summary[(field, :avg, tag, kappa_str)] = zeros(nLB, nTR)
        summary[(field, :err, tag, kappa_str)] = zeros(nLB, nTR)
    end

    # Convert "13580" -> 0.13580
    kappa_float(ks::String) = parse(Float64, "0." * ks)

    # Parse all numbers in a line; return empty on failure
    parse_numbers(line::String) = try
        parse.(Float64, split(strip(line)))
    catch
        Float64[]
    end

    # Find row matching a target kappa (first column) within tolerance
    function find_row_numbers_for_kappa(lines::Vector{String}, target::Float64)
        for ln in lines
            nums = parse_numbers(ln)
            if !isempty(nums) && isapprox(nums[1], target; atol=1e-12, rtol=0.0)
                return nums
            end
        end
        return Float64[]
    end

    # ---- main fill loop ----
    for (i_lb, label) in pairs(labels)
        for (i_tr, train) in pairs(trains)
            location = "$(work)/$(analysis_ensemble)/$(group_name)/$(overall_name)_LBP_$(label)_TRP_$(train)"
            for tag in filetags
                file = "$(tag)_$(overall_name)_LBP_$(label)_TRP_$(train).dat"
                filename = "$(location)/$(file)"
                if !isfile(filename)
                    JobLoggerTools.warn_benji("Missing file: $filename", jobid)
                    continue
                end

                lines = readlines(filename)

                for ks in kappa_list
                    target = kappa_float(ks)
                    nums = find_row_numbers_for_kappa(lines, target)
                    if isempty(nums)
                        JobLoggerTools.warn_benji("kappa '$ks' not found in $filename", jobid)
                        continue
                    end

                    expected = 1 + 2*length(obs_keys)
                    if length(nums) < expected
                        JobLoggerTools.warn_benji("Row for kappa '$ks' has $(length(nums)) values (< $expected) in $filename", jobid)
                        continue
                    end

                    # Fill (avg, err) for each observable in order
                    @inbounds for (j, fld) in pairs(obs_keys)
                        idx_avg = 2*j
                        idx_err = 2*j + 1
                        summary[(fld, :avg, tag, ks)][i_lb, i_tr] = nums[idx_avg]
                        summary[(fld, :err, tag, ks)][i_lb, i_tr] = nums[idx_err]
                    end
                end
            end
        end
    end

    return summary, kappa_list
end

"""
    load_rw_data(
        filepaths::Dict{Symbol,String}
    ) -> Dict{Symbol,Dict{Symbol,Vector{Float64}}}

Parse reweighting data files and extract observable values with associated errors.

Each file is expected to have rows like:
`kappa cond cond_err susp susp_err skew skew_err kurt kurt_err bind bind_err`

Lines marked with `#` are treated as headers. Lines following a `# kappa_t` marker are ignored (used to skip interpolation data).

# Arguments
- `filepaths`: Dictionary mapping tag symbols (e.g., `:RWP1`) to file paths.

# Returns
A nested dictionary: `tag => (observable => vector of Float64 values)`.
For each tag, returns vectors for `:cond`, `:cond_err`, ..., `:bind_err`.
"""
function load_rw_data(
    filepaths::Dict{Symbol,String}
)::Dict{Symbol,Dict{Symbol,Vector{Float64}}}
    rw_data = Dict{Symbol, Dict{Symbol, Vector{Float64}}}()

    for (tag, filepath) in filepaths
        open(filepath, "r") do io
            kappa      = Float64[]
            cond       = Float64[]; cond_err = Float64[]
            susp       = Float64[]; susp_err = Float64[]
            skew       = Float64[]; skew_err = Float64[]
            kurt       = Float64[]; kurt_err = Float64[]
            bind       = Float64[]; bind_err = Float64[]

            in_interpolation = false

            for line in eachline(io)
                s = split(strip(line))
                if isempty(s)
                    continue
                elseif startswith(s[1], "#")
                    if occursin("kappa_t", line)
                        in_interpolation = true
                    end
                    continue
                elseif in_interpolation
                    continue
                end

                push!(kappa, parse(Float64, s[1]))
                push!(cond,  parse(Float64, s[2])); push!(cond_err,  parse(Float64, s[3]))
                push!(susp,  parse(Float64, s[4])); push!(susp_err,  parse(Float64, s[5]))
                push!(skew,  parse(Float64, s[6])); push!(skew_err,  parse(Float64, s[7]))
                push!(kurt,  parse(Float64, s[8])); push!(kurt_err,  parse(Float64, s[9]))
                push!(bind,  parse(Float64, s[10])); push!(bind_err,  parse(Float64, s[11]))
            end

            rw_data[tag] = Dict(
                :kappa     => kappa,
                :cond      => cond,      :cond_err => cond_err,
                :susp      => susp,      :susp_err => susp_err,
                :skew      => skew,      :skew_err => skew_err,
                :kurt      => kurt,      :kurt_err => kurt_err,
                :bind      => bind,      :bind_err => bind_err
            )
        end
    end
    return rw_data
end

"""
    load_all_rw_data(
        labels::Vector{String},
        trains::Vector{String},
        tags::Vector{Symbol},
        path_template::Function
    ) -> Dict{String, Dict{String, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}}

Load all raw reweighting data for every (label, train, tag) combination.

This function uses a `path_template(label, train, tag)` to construct file paths, and delegates actual parsing to [`load_rw_data`](@ref).

# `path_template`

In [`Deborah.MiriamDocument.MiriamDocumentRunner.run_MiriamDocument`](@ref), the `path_template` is usually defined as:

```julia
path_template = (label, train, tag) ->
    "\$(location)/\$(analysis_ensemble)/\$(cumulant_name)/" *
    "\$(overall_name)_LBP_\$(label)_TRP_\$(train)/" *
    "\$(String(tag))_\$(overall_name)_LBP_\$(label)_TRP_\$(train).dat"
```

This closure captures `location`, `analysis_ensemble`, `cumulant_name`, and `overall_name` from the surrounding scope,
and generates a full path for each `(label, train, tag)` combination. For example:

```julia
path_template("Plaq", "Rect", :T_BS)
# → /.../<analysis_ensemble>/<cumulant_name>/<overall_name>_LBP_Plaq_TRP_Rect/T_BS_<overall_name>_LBP_Plaq_TRP_Rect.dat
```

# Arguments
- `labels`: List of `LBP` labels (e.g., `["10", "20", ...]`).
- `trains`: List of `TRP` percentages (e.g., `["0", "100"]`).
- `tags`: List of tag symbols indicating file types (e.g., `:Y_BS`, `:RWP2`).
- `path_template`: A function `(label, train, tag) → filepath::String`.

# Returns
Nested dictionary:
`label => train => tag => observable => vector of Float64`
"""
function load_all_rw_data(
    labels::Vector{String},
    trains::Vector{String},
    tags::Vector{Symbol},
    path_template::Function
)::Dict{String, Dict{String, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}}

    rw_data = Dict{String, Dict{String, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}}()

    for label in labels
        rw_data[label] = Dict()
        for train in trains
            filepaths = Dict{Symbol, String}()
            for tag in tags
                filepath = path_template(label, train, tag)
                filepaths[tag] = filepath
            end
            rw_data[label][train] = load_rw_data(filepaths)
        end
    end

    return rw_data
end

using TOML

"""
    load_nlsolve_status_from_info(
        filepath::String;
        solver_prefix::AbstractString = "nlsolve_f_solver_"
    ) -> Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}

Parse a single `infos_Miriam_...LBP_<label>_TRP_<train>.toml` file and extract
[`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) results per solver section.

It scans tables under [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) whose names start with `solver_prefix`
(e.g., `nlsolve_f_solver_FULL-LBOG-ULOG`) and returns, for each such table,
a named tuple `(converged, residual_norm, iterations)`.

- `converged` is parsed from the string `"true"`/`"false"` (case-insensitive).
- `residual_norm` is parsed as `Float64`.
- `iterations` is parsed as `Int`.

Missing fields are handled as:
- `converged` → defaults to false
- `residual_norm` → defaults to `NaN`
- `iterations` → defaults to ``-1`` (sentinel for unavailable)
"""
function load_nlsolve_status_from_info(
    filepath::String;
    solver_prefix::AbstractString = "nlsolve_f_solver_"
)::Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}
    info = TOML.parsefile(filepath)

    nl = get(info, "NLsolve", nothing)
    nl === nothing && return Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}()

    out = Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}()

    # Each entry is a table like NLsolve["nlsolve_f_solver_FULL-LBOG-ULOG"]
    for (k, v) in nl
        startswith(String(k), solver_prefix) || continue

        conv_str = get(v, "converged", nothing)
        resn_str = get(v, "residual_norm", nothing)
        iters_str = get(v, "iterations", nothing)

        conv = conv_str === nothing ? false :
               lowercase(String(conv_str)) == "true"

        resn = resn_str === nothing ? NaN :
               try
                   parse(Float64, String(resn_str))
               catch
                   NaN
               end

        iters = iters_str === nothing ? -1 :
                try
                    parse(Int, String(iters_str))
                catch
                    -1
                end

        out[String(k)] = (converged = conv, residual_norm = resn, iterations = iters)
    end

    return out
end

"""
    load_all_nlsolve_status(
        labels::Vector{String},
        trains::Vector{String},
        path_template::Function;
        solver_prefix::AbstractString = "nlsolve_f_solver_",
        on_missing::Symbol = :warn
    ) -> Dict{String, Dict{String, Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}}}

Load [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) convergence info for every (label, train) combo.

`path_template` must be a function `(label::String, train::String) -> filepath::String`
that returns the infos [`TOML`](https://toml.io/en/) path.

Returns a nested dictionary:
`label => train => solver_name => (converged, residual_norm, iterations)`

Keywords:
- `solver_prefix`: Only collect tables under [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) whose names start with this prefix.
- `on_missing`: What to do if a file is missing or unreadable.
    - `:skip`  → silently skip
    - `:warn`  → print a warning and skip (default)
    - `:error` → rethrow the error
"""
function load_all_nlsolve_status(
    labels::Vector{String},
    trains::Vector{String},
    path_template::Function;
    solver_prefix::AbstractString = "nlsolve_f_solver_",
    on_missing::Symbol = :warn
)::Dict{String, Dict{String, Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}}}

    data = Dict{String, Dict{String, Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}}}()

    for label in labels
        data[label] = Dict{String, Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}}()
        for train in trains
            filepath = path_template(label, train)
            status = Dict{String, NamedTuple{(:converged, :residual_norm, :iterations), Tuple{Bool, Float64, Int}}}()
            try
                status = load_nlsolve_status_from_info(filepath; solver_prefix)
            catch err
                if on_missing === :warn
                    JobLoggerTools.warn_benji(
                        "Failed to load infos TOML\n" *
                        "  label     = $(label)\n" *
                        "  train     = $(train)\n" *
                        "  filepath  = $(filepath)\n" *
                        "  error     = $(err)"
                    )
                elseif on_missing === :error
                    rethrow(err)
                else
                    # :skip → do nothing
                end
            end
            data[label][train] = status
        end
    end

    return data
end

end  # module SummaryLoaderRebekahMiriam