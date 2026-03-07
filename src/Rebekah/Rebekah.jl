# ============================================================================
# src/Rebekah/Rebekah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
    module Rebekah

# `Deborah.Rebekah` — Plotting and I/O utilities for [`Deborah.DeborahCore`](@ref)/[`Deborah.Esther`](@ref)/[`Deborah.Miriam`](@ref) results.

`Deborah.Rebekah` is the lightweight plotting + I/O layer in the `Deborah.jl` ecosystem.
It provides helpers to 

1. load consolidated [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) result bundles, 
2. set publication-grade [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)/[``\\LaTeX``](https://www.latex-project.org/) styles, and 
3. visualize grid scans and method comparisons (e.g., overlap/ERR heatmaps, Bhattacharyya-coefficient/JSD maps,
and `P1`/`P2` vs. baseline curves).

# Scope & Responsibilities
- **Result loading**  
  - [`JLD2Loader.load_jld2`](@ref) → `(summary, labels, trains)` for single-stage reports.
  - [`JLD2Loader.load_jld2_Miriam`](@ref) → rich multi-ensemble bundle with
    `summary*` dicts, `kappa_list`, `rw_data`, `labels`, `trains`.
- **Plot style ([``\\LaTeX``](https://www.latex-project.org/))**  
  - [`PyPlotLaTeX.set_pyplot_latex_style`](@ref) and a dedicated correlation-matrix style
    preset for consistent paper figures.
- **Heatmaps**  
  - Overlap/ERR pair: [`Heatmaps.render_overlap_and_error_heatmaps`](@ref).  
  - Bhattacharyya coefficient (BC): [`Heatmaps.render_bhattacharyya_heatmap`](@ref).  
  - Jensen-Shannon divergence (JSD): [`Heatmaps.render_jsd_heatmap`](@ref).
- **`P1`/`P2` vs Baseline curves**  
  - At fixed `LBP`: [`PXvsBSPlotter.plot_PX_BS_vs_trains`](@ref).  
  - At fixed `TRP`: [`PXvsBSPlotter.plot_PX_BS_vs_labels`](@ref).

# Public API (selected)
- [`JLD2Loader.load_jld2`](@ref)
- [`JLD2Loader.load_jld2_Miriam`](@ref)
- [`PyPlotLaTeX.set_pyplot_latex_style`](@ref)
- [`Heatmaps.render_overlap_and_error_heatmaps`](@ref)
  [`Heatmaps.render_bhattacharyya_heatmap`](@ref)
  [`Heatmaps.render_jsd_heatmap`](@ref)
- [`PXvsBSPlotter.plot_PX_BS_vs_trains`](@ref)  
  [`PXvsBSPlotter.plot_PX_BS_vs_labels`](@ref)

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah
julia> using Deborah.Rebekah

# 1) Load a consolidated results bundle (e.g., from Miriam)
julia> b = Rebekah.JLD2Loader.load_jld2_Miriam("results_<overall_name>.jld2")

# 2) Set LaTeX plotting style (optional)
julia> Rebekah.PyPlotLaTeX.set_pyplot_latex_style(0.5)

# 3) Heatmaps for a chosen observable/method
julia> using Deborah.Rebekah.Heatmaps
julia> Heatmaps.render_bhattacharyya_heatmap(bc, N_lb_arr, N_tr_arr, :TrM1, :Y_P2, bname, "figs/"; save_file=true)

# 4) Compare P1/P2 vs baseline along TRP (fixed LBP)
julia> using Deborah.Rebekah.PXvsBSPlotter
julia> plot_PX_BS_vs_trains("TrM1", "Y_BS", "P1", "P2", 20, new_dict, trains, labels, bname, "figs/"; save_file=true)
```

# Notes

* Heatmap utilities accept integer percentage grids for `LBP`/`TRP` axes and optionally
  annotate each cell with numeric values; PDFs are auto-cropped when [`pdfcrop`](https://ctan.org/pkg/pdfcrop) exists. 
* The [`Deborah.Rebekah.PXvsBSPlotter`](@ref) expects pre-aligned matrices in `new_dict` keyed by
  `"KEY:Y_*:(avg|err)"` (or `"Y_*"` when `KEY == "Deborah"`). 
* Style presets modify global [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams); call them once per session. 

# See Also

* [`Deborah.DeborahDocument`](@ref), 
* [`Deborah.EstherDocument`](@ref), 
* [`Deborah.MiriamDocument`](@ref) (document/report layers).
"""
module Rebekah

import ..JLD2
import ..PyPlot
import ..PyCall

using ..Sarah

include("SummaryLoader.jl")
include("JLD2Saver.jl")
include("Comparison.jl")

include("JLD2Loader.jl")
include("PyPlotLaTeX.jl")
include("Heatmaps.jl")
include("PXvsBSPlotter.jl")

using .SummaryLoader
using .JLD2Saver
using .Comparison

using .JLD2Loader
using .PyPlotLaTeX
using .Heatmaps
using .PXvsBSPlotter

end  # module Rebekah