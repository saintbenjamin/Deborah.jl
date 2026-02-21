# ============================================================================
# src/Esther/PathConfigBuilderEsther.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PathConfigBuilderEsther

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..TOMLConfigEsther

"""
    struct EstherPathConfig

Configuration container for all file and directory paths used in the [`Deborah.Esther`](@ref) pipeline.

# Fields
- `path::String`: Base path to the ensemble directory.
- `traceM_names::NTuple{4, String}`: Names of the four trace files (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) with model tags.
- `decoded_inputs::NTuple{4, String}`: Original trace identifier strings (e.g., `plaq.dat_rect.dat`).
- `encoded_inputs::NTuple{4, String}`: Abbreviated trace identifier strings (e.g., `Plaq-Rect`).
- `data_Y::NTuple{4, String}`: Observable names (``Y``) used for each trace model.
- `trace_dirs::NTuple{4, String}`: Directories where trace files are located.
- `tr_tex_dirs::NTuple{4, String}`: Full paths to each trace directory including trace file name.
- `analysis_dir::String`: Directory for analysis output files.
- `overall_name::String`: Unique identifier name per job (used in output filenames).
- `my_tex_dir::String`: Path to [``\\LaTeX``](https://www.latex-project.org/) output folder.
- `info_file::String`: Path to summary `.toml` file storing metadata.
"""
struct EstherPathConfig
    path::String
    traceM_names::NTuple{4, String}
    decoded_inputs::NTuple{4, String}
    encoded_inputs::NTuple{4, String}
    data_Y::NTuple{4, String}
    trace_dirs::NTuple{4, String}
    tr_tex_dirs::NTuple{4, String}
    analysis_dir::String
    overall_name::String
    my_tex_dir::String
    info_file::String
end

"""
    build_path_config_Esther(
        data::TOMLConfigEsther.TraceDataConfig,
        abbrev::StringTranscoder.AbbreviationConfig,
        jobid::Union{Nothing, String}=nothing;
        project_root::String = "",
        subdir::String = ""
    ) -> EstherPathConfig

Construct all path-related configuration for a single [`Deborah.Esther`](@ref) job.

# Arguments
- [`data::TOMLConfigEsther.TraceDataConfig`](@ref Deborah.Esther.TOMLConfigEsther.TraceDataConfig): Contains ensemble name, trace settings, and observable metadata.
- [`abbrev::StringTranscoder.AbbreviationConfig`](@ref Deborah.Sarah.StringTranscoder.AbbreviationConfig): Contains abbreviation mappings for trace identifier encoding.
- `jobid::Union{Nothing, String}` (optional): Optional job ID for labeling logs and outputs.
- `project_root::String` (keyword, optional): If non-empty, prepends the project root directory to all generated paths.
- `subdir::String` (keyword, optional): If non-empty, inserted between `data.location` and `collection` in all constructed paths.

# Returns
- [`EstherPathConfig`](@ref): Struct containing all resolved paths, names, and job labels required by the [`Deborah.Esther`](@ref) pipeline.

# Behavior
- Builds standardized trace names with or without abbreviation.
- Resolves directories for each trace (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``).
- Constructs the output analysis directory and summary filename.
- Uses `project_root` as the base path if specified.
- If `subdir` is provided, it is inserted as a subfolder under `data.location`.
- Removes existing info file, if present.

# Notes
This function does not create or write trace data, but prepares the path structure
required for downstream modules (e.g., [`Deborah.Esther.TraceDataLoader`](@ref), [`Deborah.Esther.SummaryWriterEsther`](@ref)).
"""
function build_path_config_Esther(
    data::TOMLConfigEsther.TraceDataConfig,
    abbrev::StringTranscoder.AbbreviationConfig,
    jobid::Union{Nothing, String}=nothing;
    project_root::String = "",
    subdir::String = ""
)::EstherPathConfig

    JobLoggerTools.log_stage_sub1_benji("Models used in Deborah.jl = TrM1: $(data.TrM1_model), TrM2: $(data.TrM2_model), TrM3: $(data.TrM3_model), TrM4: $(data.TrM4_model)", jobid)
    
    TrM1_model_tag = NameParser.model_suffix(data.TrM1_model, jobid)
    TrM2_model_tag = NameParser.model_suffix(data.TrM2_model, jobid)
    TrM3_model_tag = NameParser.model_suffix(data.TrM3_model, jobid)
    TrM4_model_tag = NameParser.model_suffix(data.TrM4_model, jobid)
    model_tags = (TrM1_model_tag, TrM2_model_tag, TrM3_model_tag, TrM4_model_tag)

    if isempty(project_root)
        path = joinpath(data.location, data.ensemble, "")
    else
        path = joinpath(project_root, data.location, data.ensemble, "")
    end    

    data_Y = (data.TrM1_Y, data.TrM2_Y, data.TrM3_Y, data.TrM4_Y)

    TrM1_X_joined = join(data.TrM1_X, "_")
    TrM2_X_joined = join(data.TrM2_X, "_")
    TrM3_X_joined = join(data.TrM3_X, "_")
    TrM4_X_joined = join(data.TrM4_X, "_")

    # Decode trace identifiers
    decoded_M1 = TrM1_X_joined == data.TrM1_Y ? TrM1_X_joined : "$(TrM1_X_joined)_$(data.TrM1_Y)"
    decoded_M2 = TrM2_X_joined == data.TrM2_Y ? TrM2_X_joined : "$(TrM2_X_joined)_$(data.TrM2_Y)"
    decoded_M3 = TrM3_X_joined == data.TrM3_Y ? TrM3_X_joined : "$(TrM3_X_joined)_$(data.TrM3_Y)"
    decoded_M4 = TrM4_X_joined == data.TrM4_Y ? TrM4_X_joined : "$(TrM4_X_joined)_$(data.TrM4_Y)"
    decoded_inputs = (decoded_M1, decoded_M2, decoded_M3, decoded_M4)

    # Encode trace identifiers
    TrM1_str = StringTranscoder.input_encoder_abbrev(data.TrM1_X, data.TrM1_Y, abbrev)
    TrM2_str = StringTranscoder.input_encoder_abbrev(data.TrM2_X, data.TrM2_Y, abbrev)
    TrM3_str = StringTranscoder.input_encoder_abbrev(data.TrM3_X, data.TrM3_Y, abbrev)
    TrM4_str = StringTranscoder.input_encoder_abbrev(data.TrM4_X, data.TrM4_Y, abbrev)
    encoded_inputs = (TrM1_str, TrM2_str, TrM3_str, TrM4_str)

    # Build trace names
    if data.use_abbreviation
        traceM1 = NameParser.build_trace_name(data.ensemble, TrM1_str, data.TrM1_X, data.TrM1_Y, string(data.LBP), string(data.TRP), TrM1_model_tag)
        traceM2 = NameParser.build_trace_name(data.ensemble, TrM2_str, data.TrM2_X, data.TrM2_Y, string(data.LBP), string(data.TRP), TrM2_model_tag)
        traceM3 = NameParser.build_trace_name(data.ensemble, TrM3_str, data.TrM3_X, data.TrM3_Y, string(data.LBP), string(data.TRP), TrM3_model_tag)
        traceM4 = NameParser.build_trace_name(data.ensemble, TrM4_str, data.TrM4_X, data.TrM4_Y, string(data.LBP), string(data.TRP), TrM4_model_tag)
    else
        traceM1 = NameParser.build_trace_name(data.ensemble, decoded_M1, data.TrM1_X, data.TrM1_Y, string(data.LBP), string(data.TRP), TrM1_model_tag)
        traceM2 = NameParser.build_trace_name(data.ensemble, decoded_M2, data.TrM2_X, data.TrM2_Y, string(data.LBP), string(data.TRP), TrM2_model_tag)
        traceM3 = NameParser.build_trace_name(data.ensemble, decoded_M3, data.TrM3_X, data.TrM3_Y, string(data.LBP), string(data.TRP), TrM3_model_tag)
        traceM4 = NameParser.build_trace_name(data.ensemble, decoded_M4, data.TrM4_X, data.TrM4_Y, string(data.LBP), string(data.TRP), TrM4_model_tag)
    end
    traceM_names = (traceM1, traceM2, traceM3, traceM4)

    # Analysis directories
    collection = data.analysis_header * "_" * data.ensemble
    if data.use_abbreviation
        anly_prefixes = map(zip(encoded_inputs, model_tags)) do (encoded, tag)
            data.analysis_header * "_" * data.ensemble * "_" * encoded * "_" * tag
        end
    else
        anly_prefixes = map(zip(decoded_inputs, model_tags)) do (decoded, tag)
            data.analysis_header * "_" * data.ensemble * "_" * decoded * "_" * tag
        end
    end

    trace_dirs = Tuple(map(anly_prefixes) do a
        if isempty(project_root) && isempty(subdir)
            joinpath(data.location, collection, a)
        elseif isempty(project_root)
            joinpath(data.location, subdir, collection, a)
        elseif isempty(subdir)
            joinpath(project_root, data.location, collection, a)
        else
            joinpath(project_root, data.location, subdir, collection, a)
        end
    end)

    tr_tex_dirs = (
        trace_dirs[1] * "/" * traceM1,
        trace_dirs[2] * "/" * traceM2,
        trace_dirs[3] * "/" * traceM3,
        trace_dirs[4] * "/" * traceM4,
    )

    # Main output path and name
    if data.use_abbreviation
        analysis_name = data.analysis_header * "_" * data.ensemble * "_" *
                        join((TrM1_str, TrM1_model_tag, 
                            TrM2_str, TrM2_model_tag, 
                            TrM3_str, TrM3_model_tag, 
                            TrM4_str, TrM4_model_tag), "_")
    else
        analysis_name = data.analysis_header * "_" * data.ensemble * "_" *
                        join((decoded_M1, TrM1_model_tag, 
                            decoded_M2, TrM2_model_tag, 
                            decoded_M3, TrM3_model_tag, 
                            decoded_M4, TrM4_model_tag), "_")
    end
    if isempty(project_root) && isempty(subdir)
        analysis_dir = joinpath(data.location, collection, analysis_name)
    elseif isempty(project_root)
        analysis_dir = joinpath(data.location, subdir, collection, analysis_name)
    elseif isempty(subdir)
        analysis_dir = joinpath(project_root, data.location, collection, analysis_name)
    else
        analysis_dir = joinpath(project_root, data.location, subdir, collection, analysis_name)
    end

    if data.use_abbreviation
        overall_name = data.ensemble * "_" *
                    join((TrM1_str, TrM1_model_tag, 
                            TrM2_str, TrM2_model_tag, 
                            TrM3_str, TrM3_model_tag, 
                            TrM4_str, TrM4_model_tag), "_") *
                    "_LBP_" * string(data.LBP) * "_TRP_" * string(data.TRP)
    else
        overall_name = data.ensemble * "_" *
                    join((decoded_M1, TrM1_model_tag, 
                            decoded_M2, TrM2_model_tag, 
                            decoded_M3, TrM3_model_tag, 
                            decoded_M4, TrM4_model_tag), "_") *
                    "_LBP_" * string(data.LBP) * "_TRP_" * string(data.TRP)
    end

    my_tex_dir   = joinpath(analysis_dir, overall_name)
    mkpath(my_tex_dir)

    info_file = joinpath(my_tex_dir, "infos_Esther_"*overall_name*".toml")
    if isfile(info_file)
        rm(info_file)
    end

    return EstherPathConfig(
        path,
        traceM_names,
        decoded_inputs,
        encoded_inputs,
        data_Y,
        trace_dirs,
        tr_tex_dirs,
        analysis_dir,
        overall_name,
        my_tex_dir,
        info_file,
    )
end

end  # module PathConfigBuilderEsther