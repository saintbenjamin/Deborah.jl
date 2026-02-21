# ============================================================================
# src/DeborahCore/TOMLConfigDeborah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module TOMLConfigDeborah

import TOML
import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder

"""
    struct TraceDataConfig

Holds metadata and parameters for trace data input/output configuration  
used in [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) machine-learning workflows.

# Fields
- `location::String`  
    Base directory containing raw trace data files.

- `ensemble::String`  
    Ensemble identifier (e.g., `"L8T4k13580"`).

- `analysis_header::String`  
    Prefix for analysis directories (e.g., `"analysis_..."`).

- `X::Vector{String}`  
    List of input file names (e.g., `["plaq.dat", "rect.dat"]`).

- `Y::String`  
    Output file name (e.g., `"pbp.dat"`).

- `model::String`  
    Machine learning model identifier (e.g., `"Ridge"`, `"LightGBM"`).

- `read_column_X::Vector{Int}`  
    List of ``1``-based column indices to read from each file in `X`.

- `read_column_Y::Int`  
    ``1``-based column index to read from output file `Y`.

- `index_column::Int`  
    ``1``-based column index for reading configuration indices (usually `1`).

- `LBP::Int`  
    Percentage (``0 < x < 100``) of the total configurations to assign as the labeled set.

- `TRP::Int`  
    Percentage (``0 \\le x \\le 100``) of the labeled set that is used as the training set.  
    (i.e., ``\\texttt{(training set)} = \\texttt{(total set)} \\times \\dfrac{\\texttt{LBP}}{100} \\times \\dfrac{\\texttt{TRP}}{100}``)

- `IDX_shift::Int`  
    Offset applied to align the index of input `X` and output `Y` (e.g., `0` or `1`).

- `dump_X::Bool`  
    Whether to dump preprocessed input matrix `X` to disk.

- `use_abbreviation::Bool`  
    Whether to abbreviate variable names for output directory or file naming.
"""
struct TraceDataConfig
    location::String
    ensemble::String
    analysis_header::String
    X::Vector{String}
    Y::String
    model::String
    read_column_X::Vector{Int}
    read_column_Y::Int
    index_column::Int
    LBP::Int
    TRP::Int
    IDX_shift::Int
    dump_X::Bool
    use_abbreviation::Bool
end

"""
    struct JackknifeConfig

Defines jackknife resampling parameters.

# Fields
- `bin_size::Int` : Size of bins to use for jackknife error estimation.
"""
struct JackknifeConfig
    bin_size::Int
end

"""
    struct BootstrapConfig

Defines bootstrap resampling parameters (block bootstrap).

# Fields
- `ranseed::Int`   : Random seed for reproducibility.
- `N_bs::Int`      : Number of bootstrap replicates to generate (`N_bs` `` > 0``).
- `blk_size::Int`  : Block length (`blk_size ≥ 1`). Use `1` for i.i.d. bootstrap.
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
method   = "nonoverlapping"  # one of: "nonoverlapping", "moving", "circular"
```
"""
struct BootstrapConfig
    ranseed::Int
    N_bs::Int
    blk_size::Int
    method::String
end

"""
    struct FullConfigDeborah

Aggregate configuration struct used in the [`Deborah.DeborahCore`](@ref) pipeline.

# Fields
- [`data::TraceDataConfig`](@ref TraceDataConfig) : Data input and model setup config.
- [`bs::BootstrapConfig`](@ref BootstrapConfig) : Bootstrap-specific parameters.
- [`jk::JackknifeConfig`](@ref JackknifeConfig) : Jackknife-specific parameters.
- [`abbrev::StringTranscoder.AbbreviationConfig`](@ref Deborah.Sarah.StringTranscoder.AbbreviationConfig) : Abbreviation dictionary or struct.
"""
struct FullConfigDeborah
    data::TraceDataConfig
    bs::BootstrapConfig
    jk::JackknifeConfig
    abbrev::StringTranscoder.AbbreviationConfig
end

"""
    parse_full_config_Deborah(
        toml_path::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> FullConfigDeborah

Parse a [`TOML`](https://toml.io/en/) configuration file into a [`TOMLConfigDeborah.FullConfigDeborah`](@ref) object.

# Arguments
- `toml_path::String` : Path to the [`TOML`](https://toml.io/en/) configuration file.
- `jobid::Union{Nothing, String}` : Optional job ID for logging.

# Returns
- [`TOMLConfigDeborah.FullConfigDeborah`](@ref) : Struct containing all parsed configuration sections.
"""
function parse_full_config_Deborah(
    toml_path::String,
    jobid::Union{Nothing, String}=nothing
)::FullConfigDeborah
    JobLoggerTools.println_benji("Reading config file: $toml_path", jobid)

    cfg = TOML.parsefile(toml_path)
    data = cfg["data"]
    bs   = cfg["bootstrap"]
    jk   = cfg["jackknife"]

    return FullConfigDeborah(
        TraceDataConfig(
            String(data["location"]),
            String(data["ensemble"]),
            String(data["analysis_header"]),
            Vector{String}(data["X"]),
            String(data["Y"]),
            String(data["model"]),
            Vector{Int}(data["read_column_X"]),
            Int(data["read_column_Y"]),
            Int(data["index_column"]),
            Int(data["LBP"]),
            Int(data["TRP"]),
            Int(data["IDX_shift"]),
            Bool(data["dump_X"]),
            Bool(data["use_abbreviation"])
        ),
        BootstrapConfig(
            Int(bs["ranseed"]),
            Int(bs["N_bs"]),
            Int(bs["blk_size"]),
            String(bs["method"])
        ),
        JackknifeConfig(
            Int(jk["bin_size"])
        ),
        StringTranscoder.abbreviation_map(cfg["abbreviation"])
    )
end

end  # module TOMLConfigDeborah