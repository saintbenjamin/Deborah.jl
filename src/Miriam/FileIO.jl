# ============================================================================
# src/Miriam/FileIO.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module FileIO

import Printf: @printf
import ..Sarah.JobLoggerTools
import ..TOMLConfigMiriam
import ..PathConfigBuilderMiriam
import ..Ensemble
import ..EnsembleUtils
import ..MultiEnsembleLoader

"""
    read_ensemble_array_bundle(
        cfg::TOMLConfigMiriam.FullConfigMiriam,
        paths::PathConfigBuilderMiriam.MiriamPathConfig,
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{Ensemble.EnsembleArrayBundle, Vector{String}}

Read and construct a full [`Deborah.Miriam.Ensemble.EnsembleArrayBundle`](@ref) by loading four consistent
array variants (full datasets ``+`` `Y_LB`-only datasets with/without `Y_BC` replacement).
Also prints a bundle summary (when `jobid = nothing`) and can dump raw trace
matrices if requested.

# Arguments
- [`cfg::TOMLConfigMiriam.FullConfigMiriam`](@ref Deborah.Miriam.TOMLConfigMiriam.FullConfigMiriam): Global configuration containing input paths, flags, and analysis settings.
- [`paths::PathConfigBuilderMiriam.MiriamPathConfig`](@ref Deborah.Miriam.PathConfigBuilderMiriam.MiriamPathConfig): Resolved path builder for locating ensemble files.
- `jobid::Union{Nothing, String}`: Optional job ID used for structured logging and timing.

# Loaded Variants
This function internally calls [`read_ensemble_array`](@ref) four times with the following settings
and assigns the corresponding tags:

1. `FULL-LBOG-ULOG` → full data, *no* replacements  
   - `replace_bc = false`, `replace_ul = false`, `take_only_lb = false`  
   - Loads `Y_TR`, `Y_BC`, and `Y_UL` as originally present.

2. `FULL-LBOG-ULML` → full data, **`Y_UL` replaced**  
   - `replace_bc = false`, `replace_ul = true`,  `take_only_lb = false`  
   - `Y_TR` and `Y_BC` are original; `Y_UL` is replaced to `YP_UL`.

3. `LABL-TROG-BCOG` → **`Y_LB`-only**, `Y_BC` original  
   - `replace_bc = false`, `replace_ul = false`, `take_only_lb = true`  
   - Loads only `Y_TR` and `Y_BC` (`Y_UL` is entirely skipped/ignored).

4. `LABL-TROG-BCML` → **`Y_LB`-only**, **`Y_BC` replaced**  
   - `replace_bc = true`,  `replace_ul = false`, `take_only_lb = true`  
   - Loads only `Y_TR` and `YP_BC` (`Y_UL` is entirely skipped/ignored).

> Note: When `take_only_lb = true`, any `replace_ul` setting is effectively ignored,
> and `UL` (`Y_ul`/`YP_ul`) data are not read at all.

# Behavior
- Each call to [`read_ensemble_array`](@ref) is timed and logged via [`Deborah.Sarah.JobLoggerTools.@logtime_benji`](@ref).
- The four [`Deborah.Miriam.Ensemble.EnsembleArray`](@ref)s are combined into a single
  [`Deborah.Miriam.Ensemble.EnsembleArrayBundle`](@ref) along with the tag list:
  `["FULL-LBOG-ULOG", "FULL-LBOG-ULML", "LABL-TROG-BCOG", "LABL-TROG-BCML"]`.
- The `key_list` (common ensemble identifiers) is taken from the first call and reused.
- If `cfg.data.dump_original == true`, the function invokes [`Deborah.Miriam.FileIO.dump_bundle_rawtrM`](@ref)
  to persist raw trace matrices for downstream inspection.
- If `jobid = nothing`, a human-readable summary of the bundle is printed.

# Returns
- `(`[`::Ensemble.EnsembleArrayBundle`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle)`, ::Vector{String})`:
  - The constructed bundle containing the four array variants.
  - The `key_list` used for consistent ensemble identification across variants.
"""
function read_ensemble_array_bundle(
    cfg::TOMLConfigMiriam.FullConfigMiriam,
    paths::PathConfigBuilderMiriam.MiriamPathConfig,
    jobid::Union{Nothing, String}=nothing
)::Tuple{Ensemble.EnsembleArrayBundle, Vector{String}}
    tag_FULL_LBOG_ULOG = "FULL-LBOG-ULOG"
    JobLoggerTools.log_stage_sub1_benji("read_ensemble_array() [$(tag_FULL_LBOG_ULOG)] ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ens_array_1, _, key_list = read_ensemble_array(
            cfg, paths;
            replace_bc=false, 
            replace_ul=false, 
            take_only_lb=false,
            jobid=jobid
        )
    end

    tag_FULL_LBOG_ULML = "FULL-LBOG-ULML"
    JobLoggerTools.log_stage_sub1_benji("read_ensemble_array() [$(tag_FULL_LBOG_ULML)] ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ens_array_2, _,  _ = read_ensemble_array(
            cfg, paths; 
            replace_bc=false, 
            replace_ul=true, 
            take_only_lb=false,
            jobid=jobid
        )
    end

    tag_LABL_TROG_BCOG = "LABL-TROG-BCOG"
    JobLoggerTools.log_stage_sub1_benji("read_ensemble_array() [$(tag_LABL_TROG_BCOG)] ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ens_array_3, _,  _ = read_ensemble_array(
            cfg, paths; 
            replace_bc=false,  
            replace_ul=false, 
            take_only_lb=true,
            jobid=jobid
        )
    end

    tag_LABL_TROG_BCML = "LABL-TROG-BCML"
    JobLoggerTools.log_stage_sub1_benji("read_ensemble_array() [$(tag_LABL_TROG_BCML)] ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ens_array_4, _,  _ = read_ensemble_array(
            cfg, paths; 
            replace_bc=true,  
            replace_ul=false, 
            take_only_lb=true,
            jobid=jobid
        )
    end

    JobLoggerTools.log_stage_sub1_benji("Ensemble.EnsembleArrayBundle() ::",jobid)
    JobLoggerTools.@logtime_benji jobid begin
        ens_array_bundle = Ensemble.EnsembleArrayBundle(
            [ens_array_1,        ens_array_2,        ens_array_3,        ens_array_4       ],
            [tag_FULL_LBOG_ULOG, tag_FULL_LBOG_ULML, tag_LABL_TROG_BCOG, tag_LABL_TROG_BCML]
        )
    end

    if isnothing(jobid)
        JobLoggerTools.log_stage_sub1_benji("print_bundle_summary() ::",jobid)
        JobLoggerTools.@logtime_benji jobid begin
            print_bundle_summary(
                ens_array_bundle, jobid
            )
        end
    end
   
    if cfg.data.dump_original == true
        # Dump raw trace matrices to file
        JobLoggerTools.log_stage_sub1_benji("dump_bundle_rawtrM() ::",jobid)
        JobLoggerTools.@logtime_benji jobid begin
            dump_bundle_rawtrM(
                ens_array_bundle, key_list, paths.my_tex_dir, 
                cfg.data.LBP, cfg.data.TRP
            )
        end
    end

    return ens_array_bundle, key_list
end

"""
    read_ensemble_array(
        cfg::TOMLConfigMiriam.FullConfigMiriam,
        paths::PathConfigBuilderMiriam.MiriamPathConfig;
        replace_bc::Bool=false,
        replace_ul::Bool=false,
        jobid::Union{Nothing, String}=nothing
    ) -> (Ensemble.EnsembleArray{Float64}, Params{Float64}, Vector{String})

Load and construct an [`Deborah.Miriam.Ensemble.EnsembleArray`](@ref) by reading trace data from multiple ensembles.

# Arguments
- [`cfg::TOMLConfigMiriam.FullConfigMiriam`](@ref Deborah.Miriam.TOMLConfigMiriam.FullConfigMiriam): Configuration object with metadata and analysis settings.
- [`paths::PathConfigBuilderMiriam.MiriamPathConfig`](@ref Deborah.Miriam.PathConfigBuilderMiriam.MiriamPathConfig): Path container struct with resolved directory/file paths.
- `replace_bc::Bool=false`: If `true`, use replacement files for `Y_BC` to `YP_BC`
- `replace_ul::Bool=false`: If `true`, use replacement files for `Y_UL` to `YP_UL`.
- `take_only_lb::Bool=false`: If `true`, take files for `Y_LB` only.
- `jobid::Union{Nothing, String}`: Optional job ID for logging and profiling output.

# Returns
- [`ens_array::Ensemble.EnsembleArray{Float64}`](@ref Deborah.Miriam.Ensemble.EnsembleArray): The loaded ensemble trace data.
- [`param::Params{Float64}`](@ref Deborah.Miriam.Ensemble.Params): Shared lattice parameter struct.
- `key_list::Vector{String}`: Names of all ensembles in the array (ordered).
"""
function read_ensemble_array(
    cfg::TOMLConfigMiriam.FullConfigMiriam, 
    paths::PathConfigBuilderMiriam.MiriamPathConfig; 
    replace_bc::Bool=false, 
    replace_ul::Bool=false, 
    take_only_lb::Bool=false,
    jobid::Union{Nothing, String}=nothing
)
    JobLoggerTools.println_benji("Loading all ensemble trace data...", jobid)
    grouped_data = MultiEnsembleLoader.load_grouped_trace_data(cfg, paths; jobid=jobid, replace_bc=replace_bc, replace_ul=replace_ul, take_only_lb=take_only_lb)
    JobLoggerTools.println_benji("Loaded $(length(grouped_data)) ensemble-trace combinations.", jobid)

    ens_array, key_list = EnsembleUtils.build_ensemble_array_from_trace(grouped_data, cfg, jobid)
    JobLoggerTools.println_benji("Built EnsembleArray with $(length(ens_array.data)) ensembles.", jobid)

    param = ens_array.data[1].param  # Assume all ensembles share same params
    return ens_array, param, key_list
end

"""
    print_bundle_summary(
        bundle::Ensemble.EnsembleArrayBundle, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Print a brief summary of an [`Deborah.Miriam.Ensemble.EnsembleArrayBundle`](@ref), including tags and ensemble counts.

# Arguments
- [`bundle::Ensemble.EnsembleArrayBundle`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): Collection of multiple [`Deborah.Miriam.Ensemble.EnsembleArray`](@ref) objects with substitution tags.
- `jobid::Union{Nothing, String}`: Optional job identifier for logging.

# Behavior
- Prints:
    - The tag of the first array
    - Number of ensembles in the first array
    - Total number of arrays
    - Number of ensembles in each array and their corresponding tag
"""
function print_bundle_summary(
    bundle::Ensemble.EnsembleArrayBundle, 
    jobid::Union{Nothing, String}=nothing
)
    JobLoggerTools.println_benji("=== EnsembleArrayBundle Summary ===", jobid)

    JobLoggerTools.println_benji("First tag = $(bundle.tags[1])", jobid)
    JobLoggerTools.println_benji("First EnsembleArray has $(length(bundle.arrays[1].data)) ensembles", jobid)

    JobLoggerTools.println_benji("Number of ensemble arrays: $(length(bundle.arrays))", jobid)
    for (i, arr) in enumerate(bundle.arrays)
        n_ens = length(arr.data)
        tag = bundle.tags[i]
        JobLoggerTools.println_benji("[$i] Tag: $tag → $n_ens ensemble(s)", jobid)
    end
end

"""
    source_tag_to_str(
        tag::UInt8
    ) -> String

Convert a numeric source tag into its corresponding human-readable label.

# Arguments
- `tag::UInt8`: Source tag value (typically from `0` to `2`)

# Returns
- `String`: One of the following depending on the value:
    - `0` → `"Y_tr"`
    - `1` → `"Y_bc"`
    - `2` → `"Y_ul"`
    - Otherwise → "unknown"
"""
function source_tag_to_str(
    tag::UInt8
)::String
    tag == 0 && return "Y_tr"
    tag == 1 && return "Y_bc"
    tag == 2 && return "Y_ul"
    return "unknown"
end

"""
    secondary_tag_to_str(
        tag::UInt8
    ) -> String

Convert a numeric secondary tag into its corresponding human-readable label.

# Arguments
- `tag::UInt8`: Secondary tag value (typically `0` or `1`)

# Returns
- `String`: One of the following depending on the value:
    - `0` → `"Y_lb"`
    - `1` → `"Y_ul"`
    - Otherwise → "unknown"
"""
function secondary_tag_to_str(
    tag::UInt8
)::String
    tag == 0 && return "Y_lb"
    tag == 1 && return "Y_ul"
    return "unknown"
end

"""
    dump_ens_array_rawtrM_by_ensemble(
        ens_array::Ensemble.EnsembleArray,
        key_list::Vector{String},
        outdir::String,
        LBP::Int,
        TRP::Int;
        suffix::String = ""
    ) -> Nothing

Dump raw trace moment data (`trMi`) from each ensemble in `ens_array` to separate `.dat` files.

Each file will be named using the ensemble name and parameters `LBP`, `TRP`, and optional `suffix`,
and stored in the given `outdir`.

# Arguments
- [`ens_array::Ensemble.EnsembleArray`](@ref Deborah.Miriam.Ensemble.EnsembleArray): The container holding all ensemble data
- `key_list::Vector{String}`: Ensemble names (matching order of `ens_array.data`)
- `outdir::String`: Output directory for the dumped `.dat` files
- `LBP::Int`: Ratio of labeled set (for file naming)
- `TRP::Int`: Ratio of training set (for file naming)

# Keyword Arguments
- `suffix::String = ""`: Optional suffix to include in output filenames

# Output
- Writes one file per ensemble in `outdir`. Each line in the file includes:
    - Four normalized trace values (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``),
    - Original configuration number,
    - Source tag as a string (e.g., `"Y_tr"`, `"Y_bc"`, `"Y_ul"`).
"""
function dump_ens_array_rawtrM_by_ensemble(
    ens_array::Ensemble.EnsembleArray,
    key_list::Vector{String},
    outdir::String,
    LBP::Int,
    TRP::Int;
    suffix::String = ""
)::Nothing
    mkpath(outdir)

    for (eidx, ens) in enumerate(ens_array.data)
        ens_name = key_list[eidx]
        base_name = "dump_$(ens_name)"
        filename = suffix == "" ?
            "$(base_name)_LBP_$(LBP)_TRP_$(TRP).dat" :
            "$(base_name)_$(suffix)_LBP_$(LBP)_TRP_$(TRP).dat"
        outpath = joinpath(outdir, filename)

        κ = ens.param.kappa
        nf, ns, nt = ens.param.nf, ens.param.ns, ens.param.nt
        V = ns^3 * nt
        factor = 12.0 * nf * V

        open(outpath, "w") do io
            for i in 1:ens.nconf
                val = ens.trMi[i]
                f = factor
                raw = zeros(4)
                for j in 1:4
                    f *= 2.0 * κ
                    raw[j] = val[j+1] / f
                end
                source = source_tag_to_str(ens.source_tags[i])
                secondary = secondary_tag_to_str(ens.secondary_tags[i])
                conf = ens.conf_nums[i]
                @printf(io, "%.12e\t%.12e\t%.12e\t%.12e\t%d\t%s\t%s\n",
                    raw[1], raw[2], raw[3], raw[4], conf, source, secondary)
            end
        end
    end
end

"""
    dump_bundle_rawtrM(
        bundle::Ensemble.EnsembleArrayBundle,
        key_list::Vector{String},
        outdir::String,
        LBP::Int,
        TRP::Int
    ) -> Nothing

Dump raw trace moment (`trMi`) data from all ensembles in an [`Deborah.Miriam.Ensemble.EnsembleArrayBundle`](@ref).

Each [`Deborah.Miriam.Ensemble.EnsembleArray`](@ref) in the bundle is dumped using its tag name as a suffix, and each ensemble
within the array is saved to a separate `.dat` file in the given `outdir`.

# Arguments
- [`bundle::Ensemble.EnsembleArrayBundle`](@ref Deborah.Miriam.Ensemble.EnsembleArrayBundle): The bundle of ensemble arrays to process
- `key_list::Vector{String}`: Ensemble names (must match order in each array)
- `outdir::String`: Directory to write output `.dat` files
- `LBP::Int`: Ratio of labeled set (used in filenames)
- `TRP::Int`: Ratio of training set (used in filenames)

# Output
- Writes `.dat` files for all ensembles inside each [`Deborah.Miriam.Ensemble.EnsembleArray`](@ref), with filenames tagged by `LBP`, `TRP`, and `tag`.

# Notes
- Internally calls [`dump_ens_array_rawtrM_by_ensemble`](@ref) for each  [`Deborah.Miriam.Ensemble.EnsembleArray`](@ref) in the bundle.
"""
function dump_bundle_rawtrM(
    bundle::Ensemble.EnsembleArrayBundle,
    key_list::Vector{String},
    outdir::String,
    LBP::Int,
    TRP::Int
)::Nothing
    mkpath(outdir)
    for (i, ea) in enumerate(bundle.arrays)
        tag = bundle.tags[i]
        dump_ens_array_rawtrM_by_ensemble(ea, key_list, outdir, LBP, TRP; suffix=tag)
    end
end

end  # module FileIO