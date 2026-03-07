# ============================================================================
# src/Rahab/Rahab.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Rahab

# `Deborah.Rahab` — Pre-scouting visualization & ``T=0`` utilities for [`Deborah.DeborahCore`](@ref)/[`Deborah.Esther`](@ref)/[`Deborah.Miriam`](@ref).

`Deborah.Rahab` provides exploratory analysis tools designed for *pre-scouting* an ensemble
before committing to heavy [`Deborah.DeborahCore`](@ref)/[`Deborah.Esther`](@ref)/[`Deborah.Miriam`](@ref) calculations. Its functions help
assess correlations among observables, check configuration histories, estimate
  autocorrelations and ``\\tau_{\\text{int}}``, and probe how bootstrap/jackknife errors depend on
block/bin sizes — thereby guiding the choice of near-optimal resampling windows.
In addition, `Deborah.Rahab` includes ``T=0`` scaling utilities and spectroscopy
converters for baseline checks.

# Scope & Responsibilities
- **Correlation heatmaps**: assemble per-observable samples and render annotated
  correlation matrices to reveal inter-observable dependencies.
- **Original vs `ML` histograms**: compare `OG` data (`Y_tr` ``\\oplus`` `Y_bc` ``\\oplus`` `Y_ul`) against `ML`
  predictions (`Y_tr` ``\\oplus`` `Y_bc` ``\\oplus`` `YP_ul`) with shared binning; outputs both plots and
  numerical bin files for further diagnostics.
- **Observable histories & ``\\tau_{\\text{int}}``**: visualize per-configuration time histories,
  compute autocorrelation functions ``\\rho(\\Delta)``, and estimate ``\\tau_{\\text{int}}`` with standard windows
  or truncations, highlighting regions of slow decorrelation.
- **Block/bin scans & NSR**: sweep bootstrap block sizes or jackknife bin sizes,
  plot ``\\mu\\pm\\sigma`` bands with blocked averages, show NSR curves, and mark regimes with no
  discarded samples — useful for selecting stable resampling parameters.
- **Zero-temperature scaling**: provide ``g_0^2(\\beta)``, ``\\kappa_c(g_0^2)``, and quark mass ``m_q \\, a``
  relations; include weighted-least-squares fits, linear/quadratic extrapolations,
  and spectroscopy table I/O for ``\\beta``-slice analyses.

# Role in the Ecosystem
While [`Deborah.DeborahCore`](@ref)/[`Deborah.Esther`](@ref)/[`Deborah.Miriam`](@ref) carry out the main ML, cumulant, and multi-ensemble reweighting
pipelines, **Rahab acts as reconnaissance**: it allows researchers to examine the
statistical structure of a single ensemble (correlations, histories, resampling
behaviors) to inform downstream parameter choices and detect issues early.

# Typical Workflow
```julia
julia> using Deborah.Rahab

# 1. Correlation scan
julia> Rahab.CorrPlot.plot_corr_matrix("Y_bundle.jld2"; savepdf=true)

# 2. Histogram of original vs ML
julia> Rahab.HistogramOrigML.plot_histogram_orig_vs_ml(traces, idx=3, nbins=50)

# 3. Observable history + autocorr
julia> Rahab.ObservableHistory.plot_history_and_tauint("obs.dat")

# 4. Block/bin scan with NSR
julia> Rahab.BlockBinScan.scan_block_jackknife(data; maxbin=50)

# 5. Zero-T scaling
julia> Rahab.ZeroTemperatureScale.fit_kappa_c_beta(beta_vals, kappa_vals)
```

# See Also

* [`Deborah.DeborahCore`](@ref), 
* [`Deborah.Esther`](@ref), 
* [`Deborah.Miriam`](@ref), 
* [`Deborah.Sarah`](@ref).
"""
module Rahab

import ..DelimitedFiles
import ..Statistics
import ..PyPlot
import ..Printf
import ..ProgressMeter
import ..Random

using ..Sarah

include("CorrPlot.jl")
include("HistogramOrigML.jl")
include("ObservableHistory.jl")
include("BlockBinScan.jl")
include("ZeroTemperatureScale.jl")

using .CorrPlot
using .HistogramOrigML
using .ObservableHistory
using .BlockBinScan
using .ZeroTemperatureScale

end  # module Rahab