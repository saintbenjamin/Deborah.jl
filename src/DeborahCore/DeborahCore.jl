# ============================================================================
# src/DeborahCore/DeborahCore.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module DeborahCore

# `Deborah.DeborahCore` — Bias-corrected ML pipeline.

`Deborah.DeborahCore` provides the end-to-end workflow to learn regression models for trace-like
observables, apply bias correction, and materialize study-ready artifacts (vectorized `X`/`Y`
bundles, and summary tables). It orchestrates configuration parsing,
path/name construction, dataset partitioning, feature/target preparation, multiple ML
backends ([`Ridge`](https://juliaai.github.io/MLJ.jl/stable/models/RidgeRegressor_MLJLinearModels/#RidgeRegressor_MLJLinearModels)/[`Lasso`](https://juliaai.github.io/MLJ.jl/stable/models/LassoRegressor_MLJLinearModels/#LassoRegressor_MLJLinearModels)/[`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) variants), and result writing/printing.

# Scope & Responsibilities
- **Configuration & paths**: parse a single [`TOML`](https://toml.io/en/) into strongly-typed structs and construct
  reproducible analysis paths and filenames.
- **Data preparation**: split labeled data into training/bias-correction sets; vectorize
  `X`/`Y` bundles for ML; emit `LB`/`TR`/`BC`/`UL` artifacts for inspection and reuse.
- **Model training**:
  Run baseline and machine-learning training sequences.
  [`Ridge`](https://juliaai.github.io/MLJ.jl/stable/models/RidgeRegressor_MLJLinearModels/#RidgeRegressor_MLJLinearModels) and
  [`Lasso`](https://juliaai.github.io/MLJ.jl/stable/models/LassoRegressor_MLJLinearModels/#LassoRegressor_MLJLinearModels)
  are provided via [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/).
  [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM)
  is available either through [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/)
  or via [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl).
  The **[`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/)-based [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) branch (internally referred to as `MiddleGBM`)**
  additionally supports optional hyperparameter scanning/tuning and
  per-split evaluation (`TR`/`BC`/`UL`).
- **Outputs & reporting**: write machine predictions (flattened/matrix forms), residual
  plots (`MiddleGBM` option), and jackknife/bootstrap summaries for downstream tools.

# Key Components
- [`Deborah.DeborahCore.TOMLConfigDeborah`](@ref) - parse/run config structs: `TraceDataConfig`, `BootstrapConfig`, `JackknifeConfig`, `FullConfigDeborah`; `parse_full_config_Deborah(...)`.
- [`Deborah.DeborahCore.PathConfigBuilderDeborah`](@ref) - build stable output layout: `DeborahPathConfig`; `build_path_config_Deborah(...)`.
- [`Deborah.DeborahCore.DatasetPartitionerDeborah`](@ref) — compute `LB`/`TR`/`BC`/`UL` indices and counts.
- [`Deborah.DeborahCore.XYMLInfoGenerator`](@ref) / [`Deborah.DeborahCore.XYMLVectorizer`](@ref) — split & dump `LB`/`TR`/`BC`/`UL` blocks; flatten and
  reshape `X`/`Y` for ML I/O (vector ``\\Leftrightarrow`` ``N_\\text{cnf} \\times N_\\text{src}`` matrix).
- [`Deborah.DeborahCore.FeaturePipeline`](@ref) / [`Deborah.DeborahCore.MLInputPreparer`](@ref) — assemble feature tables (`NamedTuple` form) and
  target vectors per split.
- [`Deborah.DeborahCore.BaselineSequence`](@ref) — non-ML baselines and scaffolding.
- [`Deborah.DeborahCore.MLSequence`](@ref) — model runners:
  - [``Deborah.DeborahCore.MLSequenceRidge`](@ref) ([`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) [`Ridge`](https://juliaai.github.io/MLJ.jl/stable/models/RidgeRegressor_MLJLinearModels/#RidgeRegressor_MLJLinearModels)), 
  - [``Deborah.DeborahCore.MLSequenceLasso`](@ref) ([`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) [`Lasso`](https://juliaai.github.io/MLJ.jl/stable/models/LassoRegressor_MLJLinearModels/#LassoRegressor_MLJLinearModels)),
  - [``Deborah.DeborahCore.MLSequenceLightGBM`](@ref) ([`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM)),
  - [``Deborah.DeborahCore.MLSequenceMiddleGBM`](@ref) ([`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) ``+`` learning curves/tuning; residual plots),
  - [``Deborah.DeborahCore.MLSequencePyCallLightGBM`](@ref) ([`Python` `LightGBM`](https://github.com/microsoft/LightGBM) via [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl)),
- [`Deborah.DeborahCore.SummaryWriterDeborah`](@ref) / [`Deborah.DeborahCore.ResultPrinterDeborah`](@ref) — persist and print `Y`/`YP`, bias, and
  `P1`/`P2` summaries (jackknife & bootstrap).
- [`Deborah.DeborahCore.DeborahRunner`](@ref) — glue code to execute the full pipeline end-to-end.

# Public API (typical entry points)
- [`DeborahRunner.run_Deborah`](@ref)

# Minimal Usage
```julia
julia> using Deborah
julia> run_DeborahWizard()
julia> run_Deborah("config_Deborah.toml")
```

# Notes

* **Splits**: labeled set (`LB`) is partitioned into `TR` (training) and `BC` (bias correction)
  according to `LBP` and `TRP`; `UL` is the remaining unlabeled set.
* **Shapes**: ML expects flattened vectors; helper utils convert between flattened and
  ``N_\\text{cnf} \\times N_\\text{src}`` matrices for analysis and file dumps.
* **[`MiddleGBM` (`LightGBM`)](@ref Deborah.DeborahCore.MLSequenceMiddleGBM)**: can auto-produce learning curves and residual plots; ensure
  plotting dependencies are available when `jobid === nothing`.
* **[`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl) backend**: requires [`Python` `LightGBM`](https://github.com/microsoft/LightGBM) available to the [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl) environment.

# See Also

- [`Deborah.Sarah`](@ref): shared logging, naming, formatting
- [`Deborah.Esther`](@ref): cumulant estimation at the single ensemble
- [`Deborah.Miriam`](@ref): cumulant estimation with multi-ensemble reweighting
"""
module DeborahCore

using ..Sarah
using ..Rebekah

include("TOMLConfigDeborah.jl")
include("PathConfigBuilderDeborah.jl")
include("DatasetPartitionerDeborah.jl")
include("XYMLInfoGenerator.jl")
include("XYMLVectorizer.jl")
include("FeaturePipeline.jl")
include("MLInputPreparer.jl")
include("BaselineSequence.jl")
include("MLSequenceLasso.jl")
include("MLSequenceRidge.jl")
include("MLSequenceLightGBM.jl")
include("MLSequenceMiddleGBM.jl")
include("MLSequencePyCallLightGBM.jl")
include("MLSequence.jl")
include("SummaryWriterDeborah.jl")
include("ResultPrinterDeborah.jl")
include("DeborahRunner.jl")

using .TOMLConfigDeborah
using .PathConfigBuilderDeborah
using .DatasetPartitionerDeborah
using .XYMLInfoGenerator
using .XYMLVectorizer
using .FeaturePipeline
using .MLInputPreparer
using .BaselineSequence
using .MLSequenceLasso
using .MLSequenceRidge
using .MLSequenceLightGBM
using .MLSequenceMiddleGBM
using .MLSequencePyCallLightGBM
using .MLSequence
using .SummaryWriterDeborah
using .ResultPrinterDeborah
using .DeborahRunner

end  # module DeborahCore