# ============================================================================
# src/DeborahEstherMiriam/MiriamExistenceManager.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MiriamExistenceManager

import TOML
import ..Sarah.JobLoggerTools
import ..Sarah.NameParser
import ..Sarah.StringTranscoder
import ..Miriam.MiriamRunner
import ..MiriamDependencyManager

"""
    ensure_multi_ensemble_exists(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing

Check whether all required output files exist for the multi-ensemble configuration.
If any file is missing, this function automatically triggers both [`Deborah.DeborahEstherMiriam.MiriamDependencyManager.ensure_ensemble_exists`](@ref) 
and [`Deborah.Miriam.MiriamRunner.run_Miriam`](@ref) to regenerate the necessary outputs.

# Arguments
- `toml_path::String`: Path to the configuration [`TOML`](https://toml.io/en/) file.
- `jobid::Union{Nothing, String}`: Optional job ID for logging purposes.

# Returns
- `Nothing`: Performs I/O operations and computations, but does not return a value.
"""
function ensure_multi_ensemble_exists(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)

    JobLoggerTools.println_benji("Reading config file: $toml_path", jobid)

    cfg = TOML.parsefile(toml_path)

    location         = cfg["data"]["location"]
    analysis_header  = cfg["data"]["analysis_header"]
    multi_ensemble   = cfg["data"]["multi_ensemble"]
    TrM1_X           = cfg["data"]["TrM1_X"]
    TrM1_Y           = cfg["data"]["TrM1_Y"]
    TrM1_model       = cfg["data"]["TrM1_model"]
    TrM2_X           = cfg["data"]["TrM2_X"]
    TrM2_Y           = cfg["data"]["TrM2_Y"]
    TrM2_model       = cfg["data"]["TrM2_model"]
    TrM3_X           = cfg["data"]["TrM3_X"]
    TrM3_Y           = cfg["data"]["TrM3_Y"]
    TrM3_model       = cfg["data"]["TrM3_model"]
    TrM4_X           = cfg["data"]["TrM4_X"]
    TrM4_Y           = cfg["data"]["TrM4_Y"]
    TrM4_model       = cfg["data"]["TrM4_model"]
    use_abbreviation = cfg["data"]["use_abbreviation"]
    label            = cfg["data"]["LBP"]
    train            = cfg["data"]["TRP"]
    raw_abbrev       = cfg["abbreviation"]

    abbreviation = StringTranscoder.parse_string_dict(raw_abbrev)
    TrM1_code    = StringTranscoder.input_encoder_abbrev_dict(TrM1_X, TrM1_Y, abbreviation)
    TrM2_code    = StringTranscoder.input_encoder_abbrev_dict(TrM2_X, TrM2_Y, abbreviation)
    TrM3_code    = StringTranscoder.input_encoder_abbrev_dict(TrM3_X, TrM3_Y, abbreviation)
    TrM4_code    = StringTranscoder.input_encoder_abbrev_dict(TrM4_X, TrM4_Y, abbreviation)
    TrM1_X_Y     = NameParser.make_X_Y(TrM1_X, TrM1_Y)
    TrM2_X_Y     = NameParser.make_X_Y(TrM2_X, TrM2_Y)
    TrM3_X_Y     = NameParser.make_X_Y(TrM3_X, TrM3_Y)
    TrM4_X_Y     = NameParser.make_X_Y(TrM4_X, TrM4_Y)
    TrM1_suffix  = NameParser.model_suffix(TrM1_model, jobid)
    TrM2_suffix  = NameParser.model_suffix(TrM2_model, jobid)
    TrM3_suffix  = NameParser.model_suffix(TrM3_model, jobid)
    TrM4_suffix  = NameParser.model_suffix(TrM4_model, jobid)

    if use_abbreviation
        overall_name = "$(multi_ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)_LBP_$(label)_TRP_$(train)"
        output_base = joinpath(location, "$(analysis_header)_$(multi_ensemble)", "$(analysis_header)_$(multi_ensemble)_$(TrM1_code)_$(TrM1_suffix)_$(TrM2_code)_$(TrM2_suffix)_$(TrM3_code)_$(TrM3_suffix)_$(TrM4_code)_$(TrM4_suffix)", overall_name)
    else
        overall_name = "$(multi_ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)_LBP_$(label)_TRP_$(train)"
        output_base = joinpath(location, "$(analysis_header)_$(multi_ensemble)", "$(analysis_header)_$(multi_ensemble)_$(TrM1_X_Y)_$(TrM1_suffix)_$(TrM2_X_Y)_$(TrM2_suffix)_$(TrM3_X_Y)_$(TrM3_suffix)_$(TrM4_X_Y)_$(TrM4_suffix)", overall_name)
    end

    labels = ["RWBS", "RWJK", "RWP1", "RWP2", 
              "Y_BS", "Y_JK", "Y_P1", "Y_P2"]

    good_flag = true
    for label in labels
        check_file = "$(output_base)/$(label)_$(overall_name).dat"
        if !isfile(check_file)
            JobLoggerTools.warn_benji("[Miriam.jl] $(check_file) is not found.", jobid)
            good_flag = false
        end
    end

    if !good_flag
        JobLoggerTools.println_benji("[Miriam.jl] Checking First on $(overall_name) ...", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            MiriamDependencyManager.ensure_ensemble_exists(toml_path, jobid)
        end
        JobLoggerTools.println_benji("[Miriam.jl] Launching Miriam.jl to generate $(overall_name) ...", jobid)
        JobLoggerTools.@logtime_benji jobid begin
            MiriamRunner.run_Miriam(toml_path, jobid)
        end
    end
end

end  # module MiriamExistenceManager