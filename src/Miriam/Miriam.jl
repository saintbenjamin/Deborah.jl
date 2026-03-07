# ============================================================================
# src/Miriam/Miriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Miriam

# `Deborah.Miriam` — Multi-ensemble reweighting & interpolation for lattice-QCD cumulants.

`Deborah.Miriam` aggregates single-ensemble outputs, performs multi-ensemble reweighting and interpolation
along ``\\kappa``-trajectories, and emits jackknife/bootstrap summaries for traces, moments,
and higher-order cumulants (``\\Sigma``, ``\\chi``, ``S``, ``K``, ``B``). It is the multi-ensemble layer that
sits downstream of [`Deborah.Esther`](@ref)/[`Deborah.DeborahCore`](@ref) and prepares analysis-ready text files and [`JLD2`](https://juliaio.github.io/JLD2.jl/stable)
snapshots for reporting and figure generation.

# Scope & Responsibilities
- **Configuration**: parse [`TOML`](https://toml.io/en/) into strongly-typed structs (`[data]`,
  `[input_meta]`, `[solver]`, `[bootstrap]`, `[jackknife]`, `[trajectory]`,
  and `[abbreviation]`).
- **Reweighting curves & transition points**: scan ``\\kappa``, build bootstrap distributions
  for `OG`/`P1`/`P2` estimators, and locate transition ``\\kappa`` via susceptibility/skewness/kurtosis
  probes; write per-``\\kappa`` tables and final ``\\kappa_t`` with errors.
- **Resampling outputs**:
  - **Bootstrap**: traces (``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)``), moments (``Q_n \\; (n=1,2,3,4)``), and cumulants → three files (`OG`/`P1`/`P2`) with
    ``\\mu \\pm \\sigma`` per ``\\kappa``.
  - **Jackknife**: traces, moments, and cumulants → ``\\mu \\pm \\sigma`` tables per ``\\kappa``.
- **Path-ing & names**: stable, abbreviation-aware filenames/dirs driven by the config.

# Key Components
- [`Deborah.Miriam.TOMLConfigMiriam`](@ref) — types & [`TOMLConfigMiriam.parse_full_config_Miriam`](@ref) for `[data]`,
  `[input_meta]`, `[solver]`, `[bootstrap]`, `[jackknife]`, `[trajectory]`,
  and `[abbreviation]`.
- [`ReweightingCurveBundle.reweighting_curve_bundle!`](@ref) — ``\\kappa``-scan with bootstrap `OG`/`P1`/`P2`,
  transition-point interpolation, and file emission.
- [`Deborah.Miriam.WriteBSOutput`](@ref) — `write_bs_traces`, `write_bs_moments`, `write_bs_cumulants`
  (bootstrap mean & error per ``\\kappa`` for `OG`/`P1`/`P2`).
- [`Deborah.Miriam.WriteJKOutput`](@ref) — `write_jk_traces`, `write_jk_moments`, `write_jk_cumulants`
  (jackknife mean & error per ``\\kappa``).

# Public API (typical entry points)
- [`MiriamRunner.run_Miriam`](@ref)  (orchestrated run: load → resample → reweight/interpolate → write tables).

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah
julia> run_Miriam("config_Miriam.toml")
```

# Notes

* **Estimators**: `OG` (original), `P1` (bias-corrected), `P2` (weighted `LB` and `P1`). 
* **Bootstrap**: supports nonoverlapping / moving / circular block schemes; reuse of
  bootstrap indices ensures fair `OG`/`P1`/`P2` comparison.
* **Jackknife**: delete-1 with binning; volume factor ``V = N_S^3 \\times N_T`` used where needed. 
* All column indices are **``1``-based**; abbreviation maps, when enabled, shorten tokens
  in names and paths consistently. 

# See Also

* [`Deborah.DeborahCore`](@ref): trace estimation
* [`Deborah.Esther`](@ref): single ensemble cumulants
* [`Deborah.DeborahEstherMiriam`](@ref): pipeline bridge 
* [`Deborah.MiriamDocument`](@ref): report layer
"""
module Miriam

import ..TOML
import ..StatsBase
import ..Printf
import ..OrderedCollections
import ..NLsolve
import ..Statistics

using ..Sarah

include("TOMLConfigMiriam.jl")
include("PathConfigBuilderMiriam.jl")
include("MultiEnsembleLoader.jl")
include("Ensemble.jl")
include("EnsembleUtils.jl")
include("FileIO.jl")
include("Cumulants.jl")
include("WriteJKOutput.jl")
include("Interpolation.jl")
include("Reweighting.jl")
include("ReweightingCurve.jl")
include("ReweightingBundle.jl")
include("CumulantsBundleUtils.jl")
include("CumulantsBundle.jl")
include("WriteBSOutput.jl")
include("ReweightingCurveBundle.jl")
include("MiriamRunner.jl")

using .TOMLConfigMiriam
using .PathConfigBuilderMiriam
using .MultiEnsembleLoader
using .Ensemble
using .EnsembleUtils
using .FileIO
using .Cumulants
using .WriteJKOutput
using .Interpolation
using .Reweighting
using .ReweightingCurve
using .ReweightingBundle
using .CumulantsBundleUtils
using .CumulantsBundle
using .WriteBSOutput
using .ReweightingCurveBundle
using .MiriamRunner

end  # module Miriam