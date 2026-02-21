# ============================================================================
# src/DeborahCore/PathConfigBuilderDeborah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PathConfigBuilderDeborah

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..TOMLConfigDeborah

"""
    struct DeborahPathConfig

Holds constructed path-related strings used in the [`Deborah.DeborahCore`](@ref) workflow.

# Fields
- `path::String`         : Base path to the ensemble directory.
- `overall_name::String` : Unique identifier string combining ensemble, inputs, and model tag.
- `analysis_dir::String` : Full path to the analysis output directory.
- `info_file::String`    : Path to the `.toml` file used to store run metadata.
"""
struct DeborahPathConfig
    path::String
    overall_name::String
    analysis_dir::String
    info_file::String
end

"""
    build_path_config_Deborah(
        data::TOMLConfigDeborah.TraceDataConfig,
        abbrev::StringTranscoder.AbbreviationConfig,
        X_list::Vector{String},
        jobid::Union{Nothing, String}=nothing
    ) -> DeborahPathConfig

Constructs path components used by the [`Deborah.DeborahCore`](@ref) pipeline, including the overall analysis name,
result directory, and metadata file location. Encodes input-output identifiers either in full
or using abbreviation.

# Arguments
- [`data::TOMLConfigDeborah.TraceDataConfig`](@ref Deborah.DeborahCore.TOMLConfigDeborah.TraceDataConfig) : Core configuration for the data and model.
- [`abbrev::StringTranscoder.AbbreviationConfig`](@ref Deborah.Sarah.StringTranscoder.AbbreviationConfig) : Dictionary or struct that maps input names to short codes.
- `X_list::Vector{String}` : List of actual input features used in this run.
- `jobid::Union{Nothing, String}` : Optional job ID for logging.

# Returns
- [`PathConfigBuilderDeborah.DeborahPathConfig`](@ref) : Struct containing resolved path strings for this run.
"""
function build_path_config_Deborah(
    data::TOMLConfigDeborah.TraceDataConfig,
    abbrev::StringTranscoder.AbbreviationConfig,
    X_list::Vector{String}, 
    jobid::Union{Nothing, String}=nothing
)::DeborahPathConfig

    JobLoggerTools.log_stage_sub1_benji("Selected model: $(data.model)", jobid)

    model_tag = NameParser.model_suffix(data.model, jobid)
    path = joinpath(data.location, data.ensemble, "")

    suffix = model_tag * "_LBP_" * string(data.LBP) * "_TRP_" * string(data.TRP)
    X_Y = NameParser.make_X_Y(X_list, data.Y)

    # Encode input-output string
    XY_str = StringTranscoder.input_encoder_abbrev(X_list, data.Y, abbrev)

    if data.use_abbreviation
        overall_name = "$(data.ensemble)_$(XY_str)_$(suffix)"
    else
        overall_name = "$(data.ensemble)_$(X_Y)_$(suffix)"
    end

    collection = data.analysis_header * "_" * data.ensemble

    if data.use_abbreviation
        anly_prefixes = collection * "_" * XY_str * "_" * model_tag
        traceM_name = NameParser.build_trace_name(
            data.ensemble, XY_str, data.X, data.Y,
            string(data.LBP), string(data.TRP), model_tag
        )
    else
        anly_prefixes = collection * "_" * X_Y * "_" * model_tag
        traceM_name = NameParser.build_trace_name(
            data.ensemble, X_Y, data.X, data.Y,
            string(data.LBP), string(data.TRP), model_tag
        )
    end

    analysis_dir = joinpath(data.location, collection, anly_prefixes, traceM_name)
    mkpath(analysis_dir)

    info_file = joinpath(analysis_dir, "infos_Deborah_" * overall_name * ".toml")
    if isfile(info_file)
        rm(info_file)
    end

    return DeborahPathConfig(path, overall_name, analysis_dir, info_file)
end

end  # module PathConfigBuilderDeborah