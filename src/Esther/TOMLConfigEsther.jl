# ============================================================================
# src/Esther/TOMLConfigEsther.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module TOMLConfigEsther

import ..TOML

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder

"""
    struct TraceDataConfig

Configuration for trace data input and model settings used in each trace moment (TrM).

# Fields
- `location::String`: Root directory where trace data is stored.
- `ensemble::String`: Ensemble identifier (e.g., `L8T4b1.60k13570`).
- `analysis_header::String`: Identifier prepended to all analysis folders.
- `TrM1_X::Vector{String}`: Input observables for ``\\text{Tr} \\, M^{-1}``.
- `TrM1_Y::String`: Target observable for ``\\text{Tr} \\, M^{-1}``.
- `TrM1_model::String`: Model used for ``\\text{Tr} \\, M^{-1}`` (e.g., "LightGBM").
- `TrM2_X::Vector{String}`: Input observables for ``\\text{Tr} \\, M^{-2}``.
- `TrM2_Y::String`: Target observable for ``\\text{Tr} \\, M^{-2}``.
- `TrM2_model::String`: Model used for ``\\text{Tr} \\, M^{-2}``.
- `TrM3_X::Vector{String}`: Input observables for ``\\text{Tr} \\, M^{-3}``.
- `TrM3_Y::String`: Target observable for ``\\text{Tr} \\, M^{-3}``.
- `TrM3_model::String`: Model used for ``\\text{Tr} \\, M^{-3}``.
- `TrM4_X::Vector{String}`: Input observables for ``\\text{Tr} \\, M^{-4}``.
- `TrM4_Y::String`: Target observable for ``\\text{Tr} \\, M^{-4}``.
- `TrM4_model::String`: Model used for ``\\text{Tr} \\, M^{-4}``.
- `LBP::Int`: Number of bootstrap label partitions.
- `TRP::Int`: Number of bootstrap training partitions.
- `use_abbreviation::Bool`: If true, use abbreviated input IDs (e.g., `Plaq-Rect` instead of `plaq.dat_rect.dat`).
"""
struct TraceDataConfig
    location::String
    ensemble::String
    analysis_header::String
    TrM1_X::Vector{String}
    TrM1_Y::String
    TrM1_model::String
    TrM2_X::Vector{String}
    TrM2_Y::String
    TrM2_model::String
    TrM3_X::Vector{String}
    TrM3_Y::String
    TrM3_model::String
    TrM4_X::Vector{String}
    TrM4_Y::String
    TrM4_model::String
    LBP::Int
    TRP::Int
    use_abbreviation::Bool
end

"""
    struct InputMetaConfig

Lattice configuration and physics metadata.

# Fields
- `ns::Int`: Spatial lattice size.
- `nt::Int`: Temporal lattice size.
- `nf::Int`: Number of quark flavors.
- `beta::Float64`: Gauge coupling.
- `kappa::Float64`: Hopping parameter.
"""
struct InputMetaConfig
    ns::Int
    nt::Int
    nf::Int
    beta::Float64
    kappa::Float64
end

"""
    get_LatVol(
        im::InputMetaConfig
    ) -> Int

Compute the total lattice volume ``V = N_S^3 \\times N_T``.

# Arguments
- `im::InputMetaConfig`: Lattice and physics metadata.

# Returns
- `Int`: Total lattice volume.
"""
get_LatVol(im::InputMetaConfig)::Int = im.ns * im.ns * im.ns * im.nt

"""
    struct JackknifeConfig

Configuration for jackknife resampling.

# Fields
- `bin_size::Int`: Number of configurations per jackknife bin.
"""
struct JackknifeConfig
    bin_size::Int
end

"""
    struct BootstrapConfig

Defines bootstrap resampling parameters (block bootstrap).

# Fields
- `ranseed::Int`   : Random seed for reproducibility.
- `N_bs::Int`      : Number of bootstrap replicates to generate (`N_bs` ``> 0``).
- `blk_size::Int`  : Block length (`blk_size` ``\\ge 1``). Use `1` for i.i.d. bootstrap.
- `method::String` : Block-bootstrap scheme to use. Accepted values:
    - `"nonoverlapping"` : Nonoverlapping Block Bootstrap (NBB). Partition the series
      into disjoint blocks of length `blk_size`, then resample those blocks with replacement
      to reconstruct a series of approximately the original length (last block may be truncated).
    - `"moving"`         : Moving Block Bootstrap (MBB). Candidate blocks are all
      contiguous length-`blk_size` windows; resample these with replacement.
    - `"circular"`       : Circular Block Bootstrap (CBB). Like MBB, but windows wrap
      around the end of the series (circular indexing).

# Notes
- Only the three literal strings above are recognized; other values should raise an error.
- Resampled series length should match the original; if it overshoots, truncate the final block.
- Choose `blk_size` based on dependence strength (larger for stronger autocorrelation).

# Example ([`TOML`](https://toml.io/en/))
```toml
[bootstrap]
ranseed  = 850528
N_bs     = 1000
blk_size = 500
method   = "moving"  # one of: "nonoverlapping", "moving", "circular"
```
"""
struct BootstrapConfig
    ranseed::Int
    N_bs::Int
    blk_size::Int
    method::String
end

"""
    struct FullConfigEsther

Full configuration object passed throughout the [`Deborah.Esther`](@ref) pipeline.

# Fields
- [`data::TraceDataConfig`](@ref TraceDataConfig): Input and model setup.
- [`im::InputMetaConfig`](@ref InputMetaConfig): Lattice/physics metadata.
- [`bs::BootstrapConfig`](@ref BootstrapConfig): Bootstrap settings.
- [`jk::JackknifeConfig`](@ref JackknifeConfig): Jackknife settings.
- [`abbrev::StringTranscoder.AbbreviationConfig`](@ref Deborah.Sarah.StringTranscoder.AbbreviationConfig): Abbreviation mapping for trace names.
"""
struct FullConfigEsther
    data::TraceDataConfig
    im::InputMetaConfig
    bs::BootstrapConfig
    jk::JackknifeConfig
    abbrev::StringTranscoder.AbbreviationConfig
end

"""
    parse_full_config_Esther(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> FullConfigEsther

Parse [`TOML`](https://toml.io/en/) configuration file and return complete [`Deborah.Esther`](@ref) config struct.

# Arguments
- `toml_path::String`: Path to `.toml` file.
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- [`FullConfigEsther`](@ref): Fully parsed configuration object ready for pipeline use.
"""
function parse_full_config_Esther(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)::FullConfigEsther
    JobLoggerTools.println_benji("Reading config file: $toml_path", jobid)
    cfg = TOML.parsefile(toml_path)

    data = cfg["data"]
    im = cfg["input_meta"]
    bs = cfg["bootstrap"]
    jk = cfg["jackknife"]

    return FullConfigEsther(
        TraceDataConfig(
            data["location"], data["ensemble"], data["analysis_header"],
            data["TrM1_X"], data["TrM1_Y"], data["TrM1_model"], 
            data["TrM2_X"], data["TrM2_Y"], data["TrM2_model"],
            data["TrM3_X"], data["TrM3_Y"], data["TrM3_model"],
            data["TrM4_X"], data["TrM4_Y"], data["TrM4_model"],
            data["LBP"], data["TRP"], data["use_abbreviation"]
        ),
        InputMetaConfig(
            im["ns"], im["nt"], im["nf"], 
            im["beta"], im["kappa"]
        ),
        BootstrapConfig(bs["ranseed"], bs["N_bs"], bs["blk_size"], bs["method"]),
        JackknifeConfig(jk["bin_size"]),
        StringTranscoder.abbreviation_map(cfg["abbreviation"], jobid)
    )
end

end  # module TOMLConfigEsther