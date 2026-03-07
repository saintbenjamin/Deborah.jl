# ============================================================================
# src/RebekahMiriam/RebekahMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module RebekahMiriam

# `Deborah.RebekahMiriam` — Plotting, comparison, and [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) I/O helpers tailored to [`Deborah.Miriam`](@ref) outputs.

`Deborah.RebekahMiriam` provides a [`Deborah.Miriam`](@ref)-specific companion to [`Deborah.Rebekah`](@ref):
- loads multi-ensemble [`Deborah.Miriam`](@ref) summaries (interpolation style and
  measurement-at-single-``\\kappa`` style),
- computes overlap/ERR dictionaries and similarity metrics (Bhattacharyya,
  Hellinger/JSD) against a baseline,
- renders `CHK`&`ERR`/`BC`/`JSD` heatmaps on the (`LBP` ``\\times`` `TRP`) grid,
- plots `P1`/`P2` vs original along `TRP` at fixed `LBP`,
- visualizes full reweighting curves with uncertainty bands and interpolation points,
- and persists consolidated results into a single `.jld2` file.

# Scope & Responsibilities
- **Comparison utilities**  
  Build per-observable dictionaries of overlap codes (`CHK` ``\\in`` `{0,1,2}`) and error ratios
  vs. a reference tag (e.g., `RWBS`), and compute Bhattacharyya/Hellinger matrices for
  (`LBP` ``\\times`` `TRP`) cells.
- **Heatmaps ([`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl))**  
  Render paired `CHK`/`ERR` heatmaps and single-map `BC`/`JSD` heatmaps, with [``\\LaTeX``](https://www.latex-project.org/) ticks,
  optional annotations, contour lines, and cropped PDF export.
- **`PXvsBS` curves**  
  Flatten [`Deborah.Miriam`](@ref) matrices to a legacy-like dict and plot `P1`/`P2`/original vs `TRP` at
  fixed `LBP`, for either interpolation or measurement-at-single-``\\kappa`` workflows.
- **Reweighting curve plot**  
  Show continuous ``\\kappa``-scans (`RWBS`/`RWP1`/`RWP2`) with bands, discrete point estimates,
  and final interpolated values at the chosen criterion.
- **[`JLD2`](https://juliaio.github.io/JLD2.jl/stable) saving**  
  Persist interpolation summaries *and* ``\\kappa``-indexed measurement summaries,
  plus `kappa_list`, `rw_data`, `labels`, `trains` into one `.jld2`.
- **Summary loading**  
  Read `.dat` summaries into `(field, :avg|:err, tag, keyword)` or `(… , kappa_str)`
  matrices; derive ``\\kappa`` token lists from ensemble names.

# Key Components
- [`ComparisonRebekahMiriam.build_overlap_and_error_dicts`](@ref),  
  [`ComparisonRebekahMiriam.build_overlap_and_error_dicts_for_measurements`](@ref),  
  [`ComparisonRebekahMiriam.build_bhattacharyya_dicts`](@ref) — construct `CHK`/`ERR`/`BC` grids.
- [`HeatmapsRebekahMiriam.render_overlap_and_error_heatmaps`](@ref),  
  [`HeatmapsRebekahMiriam.render_bhattacharyya_heatmap`](@ref),  
  [`HeatmapsRebekahMiriam.render_jsd_heatmap`](@ref) — `CHK`/`ERR`/`BC`/`JSD` visuals with [``\\LaTeX``](https://www.latex-project.org/) axes and optional contours.
- [`PXvsBSPlotterRebekahMiriam.plot_PX_BS_vs_trains`](@ref),  
  [`PXvsBSPlotterRebekahMiriam.plot_PX_BS_vs_trains_for_measurements`](@ref) — `P1`/`P2` vs `BS` curves across `TRP`.
- [`ReweightingPlotRebekahMiriam.plot_reweighting_pyplot`](@ref) — ``\\kappa``-scan curves ``+`` interpolated points (`RWBS`/`P1`/`P2`).
- [`SummaryLoaderRebekahMiriam.load_miriam_summary`](@ref).
- [`JLD2SaverRebekahMiriam.save_miriam_results`](@ref) — single-file persistence.

# Public API (typical entry points)
- Loading: [`SummaryLoaderRebekahMiriam.load_miriam_summary`](@ref) / [`SummaryLoaderRebekahMiriam.load_miriam_summary_for_measurement`](@ref) → `(Dict, kappa_list?)`.
- Comparison: [`ComparisonRebekahMiriam.build_overlap_and_error_dicts`](@ref), [`ComparisonRebekahMiriam.build_bhattacharyya_dicts`](@ref). 
- Plotting: [`HeatmapsRebekahMiriam.render_bhattacharyya_heatmap`](@ref), [`PXvsBSPlotterRebekahMiriam.plot_PX_BS_vs_trains`](@ref), [`ReweightingPlotRebekahMiriam.plot_reweighting_pyplot`](@ref). 
- Saving: [`JLD2SaverRebekahMiriam.save_miriam_results`](@ref).

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah
julia> using Deborah.RebekahMiriam

# Load interpolation summary and build diagnostics
julia> new_dict = SummaryLoaderRebekahMiriam.load_miriam_summary(work, ens, group, overall, labels, trains, ["skew","kurt"], [:RWBS,:RWP1,:RWP2], [:cond,:susp,:skew,:kurt])
julia> chk, err = ComparisonRebekahMiriam.build_overlap_and_error_dicts(new_dict, [:cond,:susp,:skew,:kurt], ["skew","kurt"], [:RWP1,:RWP2], :RWBS, labels, trains)  # CHK/ERR
julia> bc, _ = ComparisonRebekahMiriam.build_bhattacharyya_dicts(new_dict, [:cond], ["kurt"], [:RWP2], :RWBS, labels, trains)

# Visualize heatmaps
julia> HeatmapsRebekahMiriam.render_overlap_and_error_heatmaps(chk[(:cond,:RWP2,"kurt")], err[(:cond,:RWP2,"kurt")], N_lb_arr, N_tr_arr, :cond, :RWP2, "kurt", overall, figs_dir; save_file=true)

# P1/P2 vs BS along TRP at fixed LBP
julia> flat = PXvsBSPlotterRebekahMiriam.build_flat_plot_dict(:kurt, :RWBS, :RWP1, :RWP2, "kurt", new_dict)
julia> PXvsBSPlotterRebekahMiriam.plot_PX_BS_vs_trains("kurt", "RWBS", "RWP1", "RWP2", 25, flat, trains_int, labels_int, "kurt", overall, figs_dir; save_file=true)

# Save a consolidated JLD2
julia> JLD2SaverRebekahMiriam.save_miriam_results("out/miriam_results.jld2", legacy_summary, trace_meas, moment_meas, cumulant_meas, kappa_list, rw_data, labels, trains)
```

# Notes

* **Two workflows** are supported: (i) interpolation (keyword = guiding cumulant like `"kurt"`),
  (ii) measurement-at-``\\kappa`` (keyword = ``\\kappa`` token string like `"13580"`). Plotters/loaders handle both.
* Heatmaps assume integer percent ticks for `LBP`/`TRP` and can annotate cells; PDFs are auto-cropped if [`pdfcrop`](https://ctan.org/pkg/pdfcrop) is available. 
* All matrices are shaped `(num_labels, num_trains)` and must align with `labels`/`trains` metadata; `CHK` uses categorical codes, `ERR` is a ratio. 
"""
module RebekahMiriam

import ..PyPlot
import ..PyCall
import ..Printf
import ..JLD2
import ..PlotlyJS
import ..TOML

using ..Sarah
using ..Rebekah

include("ComparisonRebekahMiriam.jl")
include("HeatmapsRebekahMiriam.jl")
include("JLD2SaverRebekahMiriam.jl")
include("PXvsBSPlotterRebekahMiriam.jl")
include("ReweightingPlotRebekahMiriam.jl")
include("SummaryLoaderRebekahMiriam.jl")

using .ComparisonRebekahMiriam
using .HeatmapsRebekahMiriam
using .JLD2SaverRebekahMiriam
using .PXvsBSPlotterRebekahMiriam
using .ReweightingPlotRebekahMiriam
using .SummaryLoaderRebekahMiriam

end