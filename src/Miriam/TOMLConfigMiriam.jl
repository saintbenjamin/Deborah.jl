# ============================================================================
# src/Miriam/TOMLConfigMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module TOMLConfigMiriam

import TOML
import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder

"""
    struct TraceDataConfig

Configuration for the `[data]` section.

Specifies the ensemble trace file layout, trace data inputs, and modeling targets.

- `location`: Base directory path for data storage
- `multi_ensemble`: Identifier for the current ensemble group
- `ensembles`: List of ensemble names
- `analysis_header`: Base name for generated output
- `TrM1_X`, `TrM2_X`, ...: feature components for each trace
- `TrM1_Y`, `TrM2_Y`, ...: Target trace (scalar) for each data
- `TrM1_model`, ...: Learning model tag for each trace
- `LBP`: Labeled set percentage (as integer, e.g., `5` means ``5\\%`` of data used for labeled set among the total data set)
- `TRP`: Training set percentage (e.g., `95` means ``95\\%`` of data used for training set among the labeled set)
- `dump_original`: Whether to save original prediction data
- `use_abbreviation`: If true, uses abbreviation codes for filenames
"""
struct TraceDataConfig
    location::String
    multi_ensemble::String
    ensembles::Vector{String}
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
    dump_original::Bool
    use_abbreviation::Bool
end

"""
    struct InputMetaConfig

Configuration for the `[input_meta]` section.

Defines the lattice geometry and physics parameters for the ensemble setup.

- `ns`, `nt`: Lattice size (``V = N_S^3 \\times N_T``)
- `nf`: Number of quark flavors ``N_{\\text{f}}``
- `beta`: Gauge coupling ``\\beta``
- `csw`: Clover coefficient ``c_{\\text{SW}}``
- `kappa_list`: List of ``\\kappa`` values used in the simulation
"""
struct InputMetaConfig
    ns::Int
    nt::Int
    nf::Int
    beta::Float64
    csw::Float64
    kappa_list::Vector{Float64}
end

"""
    struct SolverConfig

Configuration for the `[solver]` section.

Parameters that control the trace estimation solver behavior.

- `maxiter`: Maximum number of iterations allowed
- `eps`: Convergence tolerance
"""
struct SolverConfig
    maxiter::Int
    eps::Float64
end

"""
    struct JackknifeConfig

Configuration for the `[jackknife]` section.

- `bin_size`: Number of configurations per jackknife bin
"""
struct JackknifeConfig
    bin_size::Int
end

"""
    struct BootstrapConfig

Defines bootstrap resampling parameters (block bootstrap).

# Fields
- `ranseed::Int`   : Random seed for reproducibility.
- `N_bs::Int`      : Number of bootstrap replicates to generate (`N_bs > 0`).
- `blk_size::Int`  : Block length (`blk_size` `` \\ge 1``). Use `1` for i.i.d. bootstrap.
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

# Example (TOML)
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
    struct TrajectoryConfig

Configuration for the `[trajectory]` section.

- `nkappaT`: Number of ``\\kappa`` values scanned in reweighting
"""
struct TrajectoryConfig
    nkappaT::Int
end

"""
    struct FullConfigMiriam

Top-level configuration struct combining all [`TOML`](https://toml.io/en/) sections.

- `data`: See [`TraceDataConfig`](@ref)
- `input_meta`: See [`InputMetaConfig`](@ref)
- `solver`: See [`SolverConfig`](@ref)
- `jk`: See [`JackknifeConfig`](@ref)
- `bs`: See [`BootstrapConfig`](@ref)
- `traj`: See [`TrajectoryConfig`](@ref)
- `abbrev`: See [`StringTranscoder.AbbreviationConfig`](@ref Deborah.Sarah.StringTranscoder.AbbreviationConfig)
"""
struct FullConfigMiriam
    data::TraceDataConfig
    input_meta::InputMetaConfig
    solver::SolverConfig
    jk::JackknifeConfig
    bs::BootstrapConfig
    traj::TrajectoryConfig
    abbrev::StringTranscoder.AbbreviationConfig
end

"""
    parse_full_config_Miriam(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> FullConfigMiriam

Parses the full configuration dictionary into a [`FullConfigMiriam`](@ref) object.

This function is typically called after loading a `.toml` file using [`TOML.parsefile`](https://docs.julialang.org/en/v1/stdlib/TOML/#TOML.parsefile), 
and it constructs strongly-typed configuration sections used throughout the simulation pipeline.

# Arguments
- `toml_path::String`: Path to the [`TOML`](https://toml.io/en/) configuration file.
- `jobid::Union{Nothing, String}`: Optional job ID for contextual logging.

# Returns
- [`FullConfigMiriam`](@ref): Struct with all configuration sections, used as the central config object.

# Notes
- If a required field is missing or has the wrong type, the function will throw a `KeyError` or `MethodError`.
- Assumes all values in the [`TOML`](https://toml.io/en/) file are well-formed. No defaulting or validation is performed.

## Example
```julia
cfg = parse_full_config_Miriam("config.toml")

println_benji(cfg.input_meta.beta)
println_benji(cfg.solver.maxiter)
```
"""
function parse_full_config_Miriam(
    toml_path::String, 
    jobid::Union{Nothing, String}=nothing
)::FullConfigMiriam
    JobLoggerTools.println_benji("Reading config file: $toml_path", jobid)
    cfg = TOML.parsefile(toml_path)

    data = cfg["data"]
    im = cfg["input_meta"]
    solver = cfg["solver"]
    jk = cfg["jackknife"]
    bs = cfg["bootstrap"]
    traj = cfg["trajectory"]

    return FullConfigMiriam(
        TraceDataConfig(
            data["location"], data["multi_ensemble"], 
            data["ensembles"], data["analysis_header"],
            data["TrM1_X"], data["TrM1_Y"], data["TrM1_model"], 
            data["TrM2_X"], data["TrM2_Y"], data["TrM2_model"],
            data["TrM3_X"], data["TrM3_Y"], data["TrM3_model"],
            data["TrM4_X"], data["TrM4_Y"], data["TrM4_model"],            
            data["LBP"], data["TRP"], data["dump_original"], 
            data["use_abbreviation"]
        ),
        InputMetaConfig(
            im["ns"], im["nt"], im["nf"], im["beta"],
            im["csw"], im["kappa_list"]
        ),
        SolverConfig(solver["maxiter"], solver["eps"]),
        JackknifeConfig(jk["bin_size"]),
        BootstrapConfig(bs["ranseed"], bs["N_bs"], bs["blk_size"], bs["method"]),
        TrajectoryConfig(traj["nkappaT"]),
        StringTranscoder.abbreviation_map(cfg["abbreviation"])
    )
end

end  # module TOMLConfigMiriam