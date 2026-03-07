# ============================================================================
# src/Rebekah/PyPlotLaTeX.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PyPlotLaTeX

import ..PyPlot

"""
    set_pyplot_latex_style(
        scale::Float64=0.5
    ) -> Nothing

Configure [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) with [``\\LaTeX``](https://www.latex-project.org/) rendering and appropriate font settings for publications.

This function modifies [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) to enable [``\\LaTeX``](https://www.latex-project.org/)-based text rendering and adjust 
font sizes, marker sizes, and line widths for consistent visual output.  
Useful for generating high-quality plots for papers or presentations.

# Arguments
- `scale::Float64`: Scaling factor for font sizes and figure dimensions. Default is `0.5`.

# Side Effects
- Modifies [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)'s global rendering configuration via [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams).
"""
function set_pyplot_latex_style(
    scale::Float64=0.5
)
    rcParams = PyPlot.matplotlib["rcParams"]
    rcParams.update(PyPlot.matplotlib["rcParamsDefault"])
    rcParams.update(Dict(
        "figure.figsize" => (16 * scale, 12 * scale),
        "font.size" => 24 * scale,
        "axes.labelsize" => 24 * scale,
        "legend.fontsize" => 24 * scale,
        "lines.markersize" => 18 * scale,
        "lines.linewidth" => 4 * scale,
        "font.family" => "lmodern",
        "text.usetex" => true,
        "text.latex.preamble" => raw"\usepackage{lmodern}"
    ))
end

"""
    set_pyplot_latex_style_corrmat() -> Nothing

Configure [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl)/[`matplotlib`](https://matplotlib.org/stable/) style for correlation-matrix and heatmap-style plots.

This helper sets a clean, serif-based style with Computer Modern math fonts, inward
ticks, black axes/text, and white figure background. It also configures a custom
color cycle suitable for multi-line overlays on top of correlation/overlap panels.

Unlike [`set_pyplot_latex_style`](@ref), this function **does not** enable full
external [``\\LaTeX``](https://www.latex-project.org/) rendering via [`text.usetex`](https://matplotlib.org/stable/users/explain/text/usetex.html), but instead relies on
[`mathtext.fontset = "cm"`](https://matplotlib.org/stable/users/explain/text/mathtext.html) for inline math support. This makes it lighter-weight
and convenient for interactive exploration or batch heatmap generation.

# Arguments
- *(none)*

# Effects
- Updates global [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) via `PyPlot.PyDict(PyPlot.matplotlib."rcParams")`:
    - `mathtext.fontset = "cm"`
    - `font.family = "serif"`
    - `font.size = 16`
    - `xtick.labelsize = 12`, `ytick.labelsize = 12`
    - tick directions set to `"in"` and colors set to black
    - figure/axes background set to white, edge/label/text colors set to black
    - grid linestyle set to `":"`
    - `savefig.*` parameters tuned for tight bounding boxes and higher DPI
    - `axes.prop_cycle` set to a custom color cycle for multi-series plots

# Returns
- `Nothing` (style is applied globally as a side effect).

# Notes
- Intended for correlation matrices, overlap/error heatmaps, or summary panels
  where a compact serif layout with consistent tick labeling is preferred.
- Can be safely combined with other style helpers, but the last call wins for
  overlapping [`matplotlib.rcParams`](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams) keys.
"""
function set_pyplot_latex_style_corrmat()

    new_params = Dict(
        "mathtext.fontset" => "cm",
        "font.family" => "serif",
        "font.size" => 16,
        "xtick.labelsize" => 12,
        "ytick.labelsize" => 12,
        "xtick.direction" => "in",
        "ytick.direction" => "in",
        "xtick.color" => "k",
        "ytick.color" => "k",
        "figure.facecolor" => "w",
        "figure.edgecolor" => "w",
        "axes.facecolor" => "w",
        "axes.edgecolor" => "k",
        "axes.labelcolor" => "k",
        "axes.titlepad" => 10.0,
        "text.color" => "k",
        "grid.linestyle" => ":",
        "savefig.facecolor" => "w",
        "savefig.edgecolor" => "w",
        "savefig.bbox" => "tight",
        "savefig.dpi" => 150,
        "savefig.pad_inches" => 0.05
    )

    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["mathtext.fontset"] = new_params["mathtext.fontset"]
    rcParams["font.family"] = new_params["font.family"]
    rcParams["font.size"] = new_params["font.size"]
    rcParams["xtick.labelsize"] = new_params["xtick.labelsize"]
    rcParams["ytick.labelsize"] = new_params["ytick.labelsize"]
    rcParams["xtick.direction"] = new_params["xtick.direction"]
    rcParams["ytick.direction"] = new_params["ytick.direction"]
    rcParams["xtick.color"] = new_params["xtick.color"]
    rcParams["ytick.color"] = new_params["ytick.color"]
    rcParams["figure.facecolor"] = new_params["figure.facecolor"]
    rcParams["figure.edgecolor"] = new_params["figure.edgecolor"]
    rcParams["axes.facecolor"] = new_params["axes.facecolor"]
    rcParams["axes.edgecolor"] = new_params["axes.edgecolor"]
    rcParams["axes.labelcolor"] = new_params["axes.labelcolor"]
    rcParams["text.color"] = new_params["text.color"]
    rcParams["grid.linestyle"] = new_params["grid.linestyle"]
    rcParams["savefig.facecolor"] = new_params["savefig.facecolor"]
    rcParams["savefig.edgecolor"] = new_params["savefig.edgecolor"]
    rcParams["savefig.bbox"] = new_params["savefig.bbox"]
    rcParams["savefig.dpi"] = new_params["savefig.dpi"]
    rcParams["savefig.pad_inches"] = new_params["savefig.pad_inches"];
    # Cycle of color map for plot
    color_cycle = PyPlot.matplotlib.cycler(color=[ "#B07AA2", "#FF9DA7", "#9C755F", "#BAB0AC", "#286ab0", "#f98f25", "#E15759", "#76B7B2", "#59A14E", "#EDC949",])
    new_params["axes.prop_cycle"] = color_cycle
    rcParams["axes.prop_cycle"] = new_params["axes.prop_cycle"];
end

end  # module PyPlotLaTeX