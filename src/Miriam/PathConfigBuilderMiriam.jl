# ============================================================================
# src/Miriam/PathConfigBuilderMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PathConfigBuilderMiriam

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..TOMLConfigMiriam

"""
    struct ResultFileNameConvention

Holds standardized filenames for all statistical output results in [`Deborah.Miriam`](@ref).

# Overview
- **Prefixes**
  - `trc`: trace outputs (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``)
  - `mmt`: moment outputs (``Q_n \\; (n=1,2,3,4)``)
  - `pnt`: non-reweighted cumulants for single ensemble
  - `rwt`: multi-ensemble-reweighted cumulants
- **Scopes**
  - `all`: full set
  - `P1`, `P2`: `P1` / `P2` bootstrap pipelines
- **Error types**
  - `jk`: jackknife estimates
  - `bs`: bootstrap estimates

# Fields
- `trc_all_jk::String`: Filename for **trace** results (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) with **jackknife** errors over the full set.
- `trc_all_bs::String`: Filename for **trace** results (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) with **bootstrap** errors over the full set.
- `trc_P1_bs::String`: Filename for **trace** results (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) with **bootstrap** errors using the **`P1`** pipeline.
- `trc_P2_bs::String`: Filename for **trace** results (``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``) with **bootstrap** errors using the **`P2`** pipeline.

- `mmt_all_jk::String`: Filename for **moment** results (``Q_n \\; (n=1,2,3,4)``) with **jackknife** errors over the full set.
- `mmt_all_bs::String`: Filename for **moment** results (``Q_n \\; (n=1,2,3,4)``) with **bootstrap** errors over the full set.
- `mmt_P1_bs::String`: Filename for **moment** results (``Q_n \\; (n=1,2,3,4)``) with **bootstrap** errors using the **`P1`** pipeline.
- `mmt_P2_bs::String`: Filename for **moment** results (``Q_n \\; (n=1,2,3,4)``) with **bootstrap** errors using the **`P2`** pipeline.

- `pnt_all_jk::String`: Filename for **raw cumulants** with **jackknife** errors over the full set.
- `pnt_all_bs::String`: Filename for **raw cumulants** with **bootstrap** errors over the full set.
- `pnt_P1_bs::String`: Filename for **raw cumulants** with **bootstrap** errors using the **`P1`** pipeline.
- `pnt_P2_bs::String`: Filename for **raw cumulants** with **bootstrap** errors using the **`P2`** pipeline.

- `rwt_all_jk::String`: Filename for **reweighted cumulants** with **jackknife** errors over the full set.
- `rwt_all_bs::String`: Filename for **reweighted cumulants** with **bootstrap** errors over the full set.
- `rwt_P1_bs::String`: Filename for **reweighted cumulants** with **bootstrap** errors using the **`P1`** pipeline.
- `rwt_P2_bs::String`: Filename for **reweighted cumulants** with **bootstrap** errors using the **`P2`** pipeline.
"""
struct ResultFileNameConvention
    trc_all_jk::String
    trc_all_bs::String
    trc_P1_bs::String
    trc_P2_bs::String
    mmt_all_jk::String
    mmt_all_bs::String
    mmt_P1_bs::String
    mmt_P2_bs::String
    pnt_all_jk::String
    pnt_all_bs::String
    pnt_P1_bs::String
    pnt_P2_bs::String
    rwt_all_jk::String
    rwt_all_bs::String
    rwt_P1_bs::String
    rwt_P2_bs::String
end

"""
    struct MiriamPathConfig

Stores all derived path information and file naming conventions for a [`Deborah.Miriam`](@ref) analysis session.

# Fields
- `decoded_inputs::NTuple{4, String}`: Human-readable full input names for each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
- `encoded_inputs::NTuple{4, String}`: Abbreviated short input names for each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
- `model_tags::NTuple{4, String}`: Model tag suffixes (e.g., `"GBM"`) for each ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)``.
- `analysis_dir::String`: Directory path containing the analysis folder.
- `overall_name::String`: Unique string identifying this full analysis run.
- `my_tex_dir::String`: Final result directory under `analysis_dir` (used for plots and outputs).
- `info_file::String`: [`TOML`](https://toml.io/en/) file path to store metadata about the run.
- [`fname::ResultFileNameConvention`](@ref ResultFileNameConvention): Container for result output filenames.
"""
struct MiriamPathConfig
    decoded_inputs::NTuple{4, String}
    encoded_inputs::NTuple{4, String}
    model_tags::NTuple{4, String}
    analysis_dir::String
    overall_name::String
    my_tex_dir::String
    info_file::String
    fname::ResultFileNameConvention
end

"""
    build_path_config_Miriam(
        data::TOMLConfigMiriam.TraceDataConfig, 
        abbrev::StringTranscoder.AbbreviationConfig, 
        jobid::Union{Nothing, String}=nothing
    ) -> MiriamPathConfig

Construct all derived path and filename information for a full [`Deborah.Miriam`](@ref) run.

# Arguments
- [`data::TOMLConfigMiriam.TraceDataConfig`](@ref Deborah.Miriam.TOMLConfigMiriam.TraceDataConfig): Input data source configuration including trace pairs and targets.
- [`abbrev::StringTranscoder.AbbreviationConfig`](@ref Deborah.Sarah.StringTranscoder.AbbreviationConfig): Mapping for short code aliases (e.g., `plaq.dat` → `Plaq`).
- `jobid::Union{Nothing, String}`: Optional job tag for logging output (default = nothing).

# Returns
- [`MiriamPathConfig`](@ref): Fully resolved configuration for analysis paths, result filenames, and trace model structure.

# Behavior
- Builds encoded and decoded trace input keys.
- Constructs full result filenames with [`ResultFileNameConvention`](@ref).
- Ensures result directory is created.
- Removes existing info [`TOML`](https://toml.io/en/) file if present.
"""
function build_path_config_Miriam(
    data::TOMLConfigMiriam.TraceDataConfig,
    abbrev::StringTranscoder.AbbreviationConfig,
    jobid::Union{Nothing, String}=nothing
)::MiriamPathConfig

    JobLoggerTools.log_stage_sub1_benji("Models used in Deborah.jl = TrM1: $(data.TrM1_model), TrM2: $(data.TrM2_model), TrM3: $(data.TrM3_model), TrM4: $(data.TrM4_model)", jobid)
    
    TrM1_model_tag = NameParser.model_suffix(data.TrM1_model, jobid)
    TrM2_model_tag = NameParser.model_suffix(data.TrM2_model, jobid)
    TrM3_model_tag = NameParser.model_suffix(data.TrM3_model, jobid)
    TrM4_model_tag = NameParser.model_suffix(data.TrM4_model, jobid)
    model_tags = (TrM1_model_tag, TrM2_model_tag, TrM3_model_tag, TrM4_model_tag)

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

    # Analysis directories
    collection = data.analysis_header * "_" * data.multi_ensemble

    # Main output path and name
    if data.use_abbreviation
        analysis_name = data.analysis_header * "_" * data.multi_ensemble * "_" *
                        join((TrM1_str, TrM1_model_tag,
                            TrM2_str, TrM2_model_tag, 
                            TrM3_str, TrM3_model_tag, 
                            TrM4_str, TrM4_model_tag), "_")
        overall_name = data.multi_ensemble * "_" *
                        join((TrM1_str, TrM1_model_tag, 
                            TrM2_str, TrM2_model_tag,
                            TrM3_str, TrM3_model_tag, 
                            TrM4_str, TrM4_model_tag), "_") *
                    "_LBP_" * string(data.LBP) * "_TRP_" * string(data.TRP)
    else
        analysis_name = data.analysis_header * "_" * data.multi_ensemble * "_" *
                        join((decoded_M1, TrM1_model_tag, 
                            decoded_M2, TrM2_model_tag, 
                            decoded_M3, TrM3_model_tag, 
                            decoded_M4, TrM4_model_tag), "_")
        overall_name = data.multi_ensemble * "_" *
                        join((decoded_M1, TrM1_model_tag, 
                            decoded_M2, TrM2_model_tag, 
                            decoded_M3, TrM3_model_tag, 
                            decoded_M4, TrM4_model_tag), "_") *
                    "_LBP_" * string(data.LBP) * "_TRP_" * string(data.TRP)
    end

    analysis_dir = joinpath(data.location, collection, analysis_name)
    my_tex_dir   = joinpath(analysis_dir, overall_name)
    mkpath(my_tex_dir)

    # Prepare output filenames using lattice size and beta
    fname_trc_all_jk  = joinpath(my_tex_dir, "T_JK_$(overall_name).dat")
    fname_trc_all_bs  = joinpath(my_tex_dir, "T_BS_$(overall_name).dat")
    fname_trc_P1_bs   = joinpath(my_tex_dir, "T_P1_$(overall_name).dat")
    fname_trc_P2_bs   = joinpath(my_tex_dir, "T_P2_$(overall_name).dat")

    fname_mmt_all_jk  = joinpath(my_tex_dir, "Q_JK_$(overall_name).dat")
    fname_mmt_all_bs  = joinpath(my_tex_dir, "Q_BS_$(overall_name).dat")
    fname_mmt_P1_bs   = joinpath(my_tex_dir, "Q_P1_$(overall_name).dat")
    fname_mmt_P2_bs   = joinpath(my_tex_dir, "Q_P2_$(overall_name).dat")

    fname_pnt_all_jk  = joinpath(my_tex_dir, "Y_JK_$(overall_name).dat")
    fname_pnt_all_bs  = joinpath(my_tex_dir, "Y_BS_$(overall_name).dat")
    fname_pnt_P1_bs   = joinpath(my_tex_dir, "Y_P1_$(overall_name).dat")
    fname_pnt_P2_bs   = joinpath(my_tex_dir, "Y_P2_$(overall_name).dat")

    fname_rwt_all_jk  = joinpath(my_tex_dir, "RWJK_$(overall_name).dat")
    fname_rwt_all_bs  = joinpath(my_tex_dir, "RWBS_$(overall_name).dat")
    fname_rwt_P1_bs   = joinpath(my_tex_dir, "RWP1_$(overall_name).dat")
    fname_rwt_P2_bs   = joinpath(my_tex_dir, "RWP2_$(overall_name).dat")

    fname = ResultFileNameConvention(
        fname_trc_all_jk,
        fname_trc_all_bs,
        fname_trc_P1_bs,
        fname_trc_P2_bs,
        fname_mmt_all_jk,
        fname_mmt_all_bs,
        fname_mmt_P1_bs,
        fname_mmt_P2_bs,
        fname_pnt_all_jk,
        fname_pnt_all_bs,
        fname_pnt_P1_bs,
        fname_pnt_P2_bs,
        fname_rwt_all_jk,
        fname_rwt_all_bs,
        fname_rwt_P1_bs,
        fname_rwt_P2_bs,
    )

    info_file = joinpath(my_tex_dir, "infos_Miriam_"*overall_name*".toml")
    if isfile(info_file)
        rm(info_file)
    end

    return MiriamPathConfig(
        decoded_inputs,
        encoded_inputs,
        model_tags,
        analysis_dir,
        overall_name,
        my_tex_dir,
        info_file,
        fname
    )
end

end