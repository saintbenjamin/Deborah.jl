# ============================================================================
# src/Esther/Esther.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Esther

# `Deborah.Esther` — Higher-order cumulant analysis for lattice-QCD trace data.

`Deborah.Esther` implements the statistical pipeline to extract chiral-condensate
observables from trace moments ``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)``: loading
and rescaling traces, computing ``Q``-moments and derived cumulants, running
bootstrap/jackknife, and emitting formatted summaries and files ready for
downstream reporting.

# Scope & Responsibilities
- **Configuration & paths**
  - [`Deborah.Esther.TOMLConfigEsther`](@ref): parse a single [`TOML`](https://toml.io/en/) into strongly-typed structs
    (`TraceDataConfig`, `InputMetaConfig`, `BootstrapConfig`, `JackknifeConfig`,
    `FullConfigEsther`), including abbreviation maps.
  - [`Deborah.Esther.PathConfigBuilderEsther`](@ref): construct reproducible output layouts and names.
  - [`Deborah.Esther.DatasetPartitionerEsther`](@ref): prepare (`labels` ``\\times`` `trains`) partitions and indices.

- **Trace I/O & normalization**
  - [`Deborah.Esther.TraceDataLoader`](@ref): load `Y_*` / `YP_*` series for ``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)`` into label-keyed
    dictionaries (`Y_info`, `Y_tr`, `Y_bc`, `Y_ul`, `Y_lb`, `YP_tr`, `YP_bc`,
    `YP_ul`).
  - [`Deborah.Esther.TraceRescaler`](@ref): apply ``V``/``\\kappa`` rescaling for each power ``p``:
    ``[\\mathrm{Tr}\\,M^{-p}]_{\\mathrm{rescaled}} = 12V(2\\kappa)^p [\\mathrm{Tr}\\,M^{-p}]_{\\mathrm{in}}``.

- **Moment & cumulant computation**
  - [`Deborah.Esther.SingleQMoment`](@ref): compute per-configuration ``Q_n \\; (n=1,2,3,4)`` from ``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)`` and ``N_{\\text{f}}``.
  - [`Deborah.Esther.QMomentCalculator`](@ref): vectorized ``Q``-moment builders over resamples.
  - [`Deborah.Esther.SingleCumulant`](@ref): per-resample chiral condensate / susceptibility / skewness / kurtosis
    from ``Q_n \\; (n=1,2,3,4)`` with the standard formulas.
  - [`Deborah.Esther.BootstrapDerivedCalculator`](@ref): derive observables from bootstrap-averaged ``Q``-data.

- **Resampling & error analysis**
  - [`Deborah.Esther.JackknifeRunner`](@ref): delete-``1`` jackknife with binning support (bin size from gauge configuration).
  - Bootstrap methods: nonoverlapping / moving / circular (config-driven).

- **Reporting & output**
  - [`Deborah.Esther.SummaryWriterEsther`](@ref): write `summary_Esther_<overall_name>.dat`
    for models (``\\text{Tr}\\,M^{-n} \\; (n=1,2,3,4)``), ``Q_n \\; (n=1,2,3,4)``, and derived observables with
    `(mean, std)` per label/tag.
  - [`Deborah.Esther.ResultPrinterEsther`](@ref): print console tables grouping jackknife/ bootstrap
    by tags (e.g., `Y:JK`, `Y:BS`, `Y_P1`, `Y_P2`, etc.).

- **Execution**
  - [`Deborah.Esther.EstherRunner`](@ref): end-to-end runner (config → load/scale traces → ``Q``-moments →
    cumulants → resampling → summaries).

# Public API (typical entry point)
- [`EstherRunner.run_Esther`](@ref)

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah
julia> run_Esther("config_Esther.toml")
```

# Notes

* Column indices in configs are **``1``-based** and propagated as-is.
* Output directory scheme is derived from `analysis_header` and `ensemble`
  to keep results reproducible across runs.
* Abbreviation maps (when enabled) shorten feature/target tokens consistently
  across file and folder names. 

# See Also

* [`Deborah.DeborahCore`](@ref) (trace estimation with ML/bias correction),
* [`Deborah.DeborahEsther`](@ref) (bridge runner), 
* [`Deborah.EstherDocument`](@ref) (report layer),
* [`Deborah.EstherThreads`](@ref) (threaded batch runner).
"""
module Esther

import ..TOML
import ..Printf

using ..Sarah

include("TOMLConfigEsther.jl")
include("DatasetPartitionerEsther.jl")
include("PathConfigBuilderEsther.jl")
include("TraceDataLoader.jl")
include("TraceRescaler.jl")
include("SingleQMoment.jl")
include("SingleCumulant.jl")
include("QMomentCalculator.jl")
include("BootstrapDerivedCalculator.jl")
include("JackknifeRunner.jl")
include("SummaryWriterEsther.jl")
include("ResultPrinterEsther.jl")
include("EstherRunner.jl")

using .TOMLConfigEsther
using .DatasetPartitionerEsther
using .PathConfigBuilderEsther
using .TraceDataLoader
using .TraceRescaler
using .SingleQMoment
using .SingleCumulant
using .QMomentCalculator
using .BootstrapDerivedCalculator
using .JackknifeRunner
using .SummaryWriterEsther
using .ResultPrinterEsther
using .EstherRunner

end  # module Esther