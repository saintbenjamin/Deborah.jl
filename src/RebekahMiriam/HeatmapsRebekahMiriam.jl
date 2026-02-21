# ============================================================================
# src/RebekahMiriam/HeatmapsRebekahMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module HeatmapsRebekahMiriam

import PyPlot
import PyCall
import Printf: @sprintf
import ..Sarah.JobLoggerTools

"""
    render_overlap_and_error_heatmaps(
        chk_arr::Array{Int, 2},
        err_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Render `CHK` and `ERR` heatmaps for a specific observable, method, and interpolation origin.

This function displays two side-by-side heatmaps using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl):
- Left: `CHK` matrix (categorical values `0`, `1`, `2` rendered as black-gray-white).
- Right: `ERR` matrix (real-valued error ratios with color scale from `1.0` to `7.0`).
  Values outside this range are explicitly labeled on the plot.

Each heatmap uses `N_lb_arr` (labeled set percentages) on the ``y``-axis
and `N_tr_arr` (training set percentages) on the ``x``-axis.
Axis ticks are [``\\LaTeX``](https://www.latex-project.org/)-formatted, and colorbars are included for both subplots.

The title of the `CHK` panel reflects the observable `key` and the interpolation origin `keyword`.

# Arguments
- `chk_arr::Array{Int,2}`: Matrix of overlap quality indicators (`0`, `1`, or `2`).
- `err_arr::Array{Float64,2}`: Matrix of error ratios for the same grid.
- `N_lb_arr::Vector{Int}`: Labeled set percentage values for ``y``-axis ticks.
- `N_tr_arr::Vector{Int}`: Training set percentage values for ``x``-axis ticks.
- `key::Symbol`: Target observable being evaluated (e.g., `:kurt`, `:cond`).
- `pred_tag::Symbol`: Identifier for the prediction method (e.g., `:RWP1`, `:RWP2`).
- `keyword::String`: Indicates the origin cumulant used to determine the interpolation point.  
  For example, `"skew"` or `"kurt"` means that the interpolation target ``\\kappa_t`` was selected  
  based on the behavior of the skewness or kurtosis, respectively.  
  Once ``\\kappa_t`` is determined using this origin cumulant, a different cumulant--such as the condensate—may be evaluated at that same point.  
  In other words, `keyword` identifies which cumulant guided the interpolation,  
  even though the reported result may pertain to a different observable.
- `overall_name::String`: Suffix string used for naming the saved figure.
- `figs_dir::String`: Directory to save the output figure.

# Keyword Arguments
- `key_tex::String = ""`: Optional [``\\LaTeX``](https://www.latex-project.org/) string for labeling the `CHK` subplot.
- `save_file::Bool = false`: If `true`, saves the figure as a PDF and crops it using [`pdfcrop`](https://ctan.org/pkg/pdfcrop) if available.

# Side Effects
- Displays a formatted figure using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).
- If `save_file=true`, writes `heatmap_<key>_<keyword>_<pred_tag>_<overall_name>.pdf` to `figs_dir`.
"""
function render_overlap_and_error_heatmaps(
    chk_arr::Array{Int, 2},
    err_arr::Array{Float64, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool=false
)

    fig, axs = PyPlot.subplots(1, 2, figsize=(10, 5), dpi=500)

    # === CHK colormap: 0=black, 1=gray, 2=white ===
    chk_colors = ["black", "gray", "white"]
    cmap_chk = PyPlot.matplotlib.colors.ListedColormap(chk_colors)
    bounds_chk = [0, 1, 2, 3]
    norm_chk = PyPlot.matplotlib.colors.BoundaryNorm(bounds_chk, cmap_chk.N)

    im1 = axs[1].imshow(chk_arr, cmap=cmap_chk, norm=norm_chk, origin="lower", aspect="auto")
    axs[1].set_title(key_tex * " (\$\\kappa_t\$ with " * keyword * ")")
    axs[1].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[1].set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")
    axs[1].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[1].set_yticks(collect(0:length(N_lb_arr)-1))
    axs[1].set_xticklabels(N_tr_arr)
    axs[1].set_yticklabels(N_lb_arr)

	cbar1 = fig.colorbar(im1, ax=axs[1], ticks=[0.5, 1.5, 2.5], shrink=0.75, aspect=20, pad=0.02)
	cbar1.ax.set_yticklabels(["0", "1", "2"])
    #cbar1.set_label("Overlap Quality")
	
    # === ERR heatmap ===
    cmap_err = PyPlot.get_cmap("gist_rainbow")
    im2 = axs[2].imshow(err_arr, cmap=cmap_err, origin="lower", vmin=1.0, vmax=7.0, aspect="auto")
    axs[2].set_title(pred_tag)
    axs[2].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[2].set_yticks([])
    axs[2].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[2].set_xticklabels(N_tr_arr)

    for i in 1:size(err_arr, 1), j in 1:size(err_arr, 2)
        val = round(err_arr[i,j]; digits=2)
	    # if val < 1.0 || val > 7.0
        pe = PyCall.pyimport("matplotlib.patheffects")
        # outline = [pe.withStroke(linewidth=0.2, foreground="white")]
        
        color_val = (val < 2.0 || val > 4.0) ? "white" : "black"

        # axs[2].text(j - 1, i - 1, "$val",
        #     ha="center", va="center",
        #     color=color_val, fontsize=9,
        #     path_effects=outline)
        axs[2].text(j - 1, i - 1, "$val",
            ha="center", va="center",
            color=color_val, fontsize=9)
	    # end
    end

	cbar2 = fig.colorbar(im2, ax=axs[2], shrink=0.75, aspect=20, pad=0.02)
    #cbar2.set_label("Error Ratio")
	
    fig.tight_layout()
    display(fig)

    basename = "heatmap_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end


"""
    overlay_nlsolve_marks!(
        ax,
        conv_arr::Array{Bool,2},
        iter_arr::Array{Int,2};
        iter_threshold::Int = 10,
        color::AbstractString = "white",
        lw::Real = 0.8,
        pad::Float64 = 0.48
    ) -> Nothing

Overlay per-cell convergence marks on an existing heatmap (typically the `ERR`
panel in [`render_overlap_and_error_heatmaps`](@ref)).

For each cell `(i,j)` this function draws:

- a **red X** if the corresponding [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) call did **not** converge, and  
- a **single diagonal segment** (top-right → bottom-left) if it converged but
  required at least `iter_threshold` iterations.

Cells that converged with fewer than `iter_threshold` iterations are left
unmarked.

The coordinates are assumed to match the usual `imshow`/`text` convention used
in [`render_overlap_and_error_heatmaps`](@ref), i.e. cell centers located at `(j-1, i-1)`.

# Arguments
- `ax`:
    `PyPlot` axes object on which to draw the marks. This should be the same
    axis that already displays the heatmap (e.g., the `ERR` panel).
- `conv_arr::Array{Bool,2}`:
    Boolean matrix encoding [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) convergence status:
    - `true`  → solver converged,
    - `false` → solver failed to converge.
- `iter_arr::Array{Int,2}`:
    Integer matrix with the same shape as `conv_arr`, holding the number of
    iterations used for each solve.

# Keyword Arguments
- `iter_threshold::Int = 10`:
    Minimum iteration count to trigger a “slow convergence” mark. Cells with
    `conv_arr[i,j] == true` and `iter_arr[i,j] ≥ iter_threshold` receive a
    single diagonal segment.
- `color::AbstractString = "white"`:
    Color of the diagonal mark used for “slow but converged” cells.
- `lw::Real = 0.8`:
    Line width for both X-marks and diagonal segments.
- `pad::Float64 = 0.48`:
    Half-length of each line segment inside a cell. The segment endpoints are
    placed at `(cx ± pad, cy ± pad)` where `(cx, cy) = (j-1, i-1)` is the cell
    center.

# Behavior
- For `!conv_arr[i,j]`:
    - Draw two red diagonals across the cell, forming an X.
- For `conv_arr[i,j] && iter_arr[i,j] ≥ iter_threshold`:
    - Draw a single diagonal (top-right → bottom-left) in `color`.
- For other cells:
    - No overlay is drawn.

# Returns
- `Nothing` (modifies `ax` in place as a side effect).

# Notes
- This function assumes that the heatmap pixels align with integer grid cells
  as in `imshow(...; origin="lower")` with tick/label logic matching
  [`render_overlap_and_error_heatmaps`](@ref).
- For best visibility, call [`overlay_nlsolve_marks!`](@ref) **after** rendering the
  base heatmap but **before** calling `tight_layout`/`savefig`, so that marks
  are included in the final layout.
"""
function overlay_nlsolve_marks!(
    ax,
    conv_arr::Array{Bool,2},
    iter_arr::Array{Int,2};
    iter_threshold::Int = 10,
    color::AbstractString = "white",
    lw::Real = 0.8,
    pad::Float64 = 0.48  # half-size of the line segment inside each cell
)
    nL, nT = size(conv_arr)
    JobLoggerTools.assert_benji(
        size(iter_arr) == size(conv_arr),
        "size(iter_arr) must equal size(conv_arr)"
    )

    for i in 1:nL, j in 1:nT
        cx = j - 1       # cell center x (matches your imshow/text usage)
        cy = i - 1       # cell center y

        if !conv_arr[i,j]
            # draw an X: two diagonals across the cell
            ax.plot([cx - pad, cx + pad], [cy - pad, cy + pad]; color="red", linewidth=lw, zorder=1)
            ax.plot([cx + pad, cx - pad], [cy - pad, cy + pad]; color="red", linewidth=lw, zorder=1)
        elseif iter_arr[i,j] ≥ iter_threshold
            # draw a single diagonal (top-right → bottom-left)
            ax.plot([cx + pad, cx - pad], [cy + pad, cy - pad]; color=color, linewidth=lw, zorder=1)
        end
    end
    return nothing
end

"""
    render_overlap_type_b_and_error_heatmaps(
        ovl_arr::Array{Float64, 2},
        err_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        vmax_ovl::Float64 = 3.0
    ) -> Nothing

Render Type-B distance (`ovl_arr`) and `ERR` heatmaps for a specific observable, method,
and interpolation origin.

Left panel visualizes the **Type-B distance** (continuous) defined as  
```math
d = \\frac{\\lvert \\mu_{\\text{orig}} - \\mu_{\\text{pred}} \\rvert}{\\max(\\sigma_{\\text{orig}}, \\sigma_{\\text{floor}})}.
```  
Color map is tuned so that:
- ``d = 0`` → white,
- ``d \\approx 1`` → vivid color (attention threshold),
- ``1 - \\texttt{vmax\\_ovl}`` → smoothly intensifying,
- ``d > \\texttt{vmax\\_ovl}`` → shown with an extended colorbar (triangle) yet numbers are still annotated.

Right panel is the same `ERR` heatmap as before (``[1.0,7.0]`` by default).

Both panels place `N_lb_arr` on the y-axis and `N_tr_arr` on the x-axis. Cell values are annotated.

# Arguments
- `ovl_arr`: Type-B distance matrix (`Float64`).
- `err_arr`: Error-ratio matrix (`Float64`).
- `N_lb_arr`, `N_tr_arr`: Percent tick lists for ``y``/``x`` axes.
- `key`, `pred_tag`, `keyword`: Labels for titles/filenames.
- `overall_name`, `figs_dir`: Suffix and output directory.

# Keywords
- `key_tex`: Optional [``\\LaTeX``](https://www.latex-project.org/) label for the left subplot title.
- `save_file`: If `true`, save PDF and crop via [`pdfcrop`](https://ctan.org/pkg/pdfcrop) if available.
- `vmax_ovl`: Upper bound for Type-B color scaling (default `3.0`).

# Side Effects
- Displays a figure via [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).
- If `save_file=true`, writes `heatmap_typeB_<key>_<keyword>_<pred_tag>_<overall_name>.pdf`.
"""
function render_overlap_type_b_and_error_heatmaps(
    ovl_arr::Array{Float64, 2},
    err_arr::Array{Float64, 2},
    conv_arr::Array{Bool,2},      # ← added
    iter_arr::Array{Int,2},       # ← added
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    vmax_ovl::Float64 = 3.0,
    iter_threshold::Int = 10      # ← added (for overlays)
)
    # Figure & axes
    fig, axs = PyPlot.subplots(1, 2, figsize=(10, 5), dpi=500)

    # ==== OVL (Type-B) colormap ====
    # Base colormap with near-white low end, strong reds at high end
    cmap_ovl = PyPlot.get_cmap("YlOrRd").copy()
    # Ensure exact white at the very bottom, and a strong over-color beyond vmax
    cmap_ovl.set_under("white")
    cmap_ovl.set_over("#7f0000")  # dark red for > vmax
    im1 = axs[1].imshow(
        ovl_arr, cmap=cmap_ovl, origin="lower", vmin=0.0, vmax=vmax_ovl, aspect="auto"
    )

    keyword_tex = ""
    if keyword == "susp"
        keyword_tex = "\$\\chi(\\kappa_t)\$"
    elseif keyword == "skew"
        keyword_tex = "\$S(\\kappa_t)\$"
    elseif keyword == "kurt"
        keyword_tex = "\$K(\\kappa_t)\$"
    else
        keyword_tex = keyword
    end
    # axs[1].set_title(key_tex * " (from " * keyword_tex * " interpolation)")
    axs[1].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[1].set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")
    axs[1].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[1].set_yticks(collect(0:length(N_lb_arr)-1))
    axs[1].set_xticklabels(N_tr_arr)
    axs[1].set_yticklabels(N_lb_arr)

    # Annotate ovl values per cell
    # pe = PyCall.pyimport("matplotlib.patheffects")
    # outline = [pe.withStroke(linewidth=0.2, foreground="white")]
    for i in 1:size(ovl_arr, 1), j in 1:size(ovl_arr, 2)
        val = round(ovl_arr[i, j]; digits=2)
        text_color = val >= 2.0 ? "white" : "black"
        fontweight = (text_color == "white") ? "bold" : "normal"
        # axs[1].text(j - 1, i - 1, string(val),
        #     ha="center", va="center", color=text_color, fontsize=9, path_effects=outline)
        axs[1].text(j - 1, i - 1, string(val),
            ha="center", va="center", color=text_color, fontsize=9, fontweight=fontweight)
    end

    # Colorbar (extend right for > vmax_ovl)
    cbar1 = fig.colorbar(
        im1, ax=axs[1], shrink=0.75, aspect=20, pad=0.02, extend="max"
    )
    # Emphasize checkpoints around 0, 0.5, 1, 2, 3
    cbar1.set_ticks([0.0, 0.5, 1.0, 2.0, vmax_ovl])
    cbar1.set_ticklabels(["0", "0.5", "1", "2", string(vmax_ovl)])

    # ==== ERR heatmap (as before) ====
    cmap_err = PyPlot.get_cmap("gist_rainbow")
    im2 = axs[2].imshow(err_arr, cmap=cmap_err, origin="lower", vmin=1.0, vmax=7.0, aspect="auto")
    pred_tag_map = Dict(
        "RWP1" => "\$\\mathcal{P}1\$",
        "RWP2" => "\$\\mathcal{P}2\$"
    )
    right_title = get(pred_tag_map, string(pred_tag), string(pred_tag))

    # axs[2].set_title(right_title)
    axs[2].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[2].set_yticks([])
    axs[2].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[2].set_xticklabels(N_tr_arr)

    for i in 1:size(err_arr, 1), j in 1:size(err_arr, 2)
        val = round(err_arr[i, j]; digits=2)
        color_val = (val < 1.41 || val > 4.0) ? "white" : "black"
        fontweight = (color_val == "white") ? "bold" : "normal"
        # axs[2].text(j - 1, i - 1, string(val),
        #     ha="center", va="center", color="black", fontsize=9, path_effects=outline)
        axs[2].text(j - 1, i - 1, string(val),
            ha="center", va="center", color=color_val, fontsize=9, fontweight=fontweight)
    end

    overlay_nlsolve_marks!(axs[1], conv_arr, iter_arr; iter_threshold=iter_threshold, color="white", lw=0.9, pad=0.48)

    overlay_nlsolve_marks!(axs[2], conv_arr, iter_arr; iter_threshold=iter_threshold, color="white", lw=0.9, pad=0.48)

    cbar2 = fig.colorbar(im2, ax=axs[2], shrink=0.75, aspect=20, pad=0.02)
    # cbar2.set_label("Error Ratio")

    fig.tight_layout()
    display(fig)

    # Save (optional)
    basename = "heatmap_typeB_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end

"""
    render_bhattacharyya_heatmap(
        bc_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        annotate::Bool = true,
        cmap_name::String = "viridis",
        levels::Vector{Float64} = [0.6, 0.8, 0.9],
        draw_contours::Bool = true
    ) -> Nothing

Render a single heatmap for the Bhattacharyya coefficient (``\\mathrm{BC}`` in ``[0,1]``) for a given
observable, method, and interpolation origin (`keyword`).

- Colormap spans `0.0` (worst) to `1.0` (best).
- Optional numeric annotations per cell.
- Optional contour lines at given `levels`.

# Arguments
- `bc_arr::Array{Float64,2}`: ``\\mathrm{BC}`` matrix in ``[0,1]``.
- `N_lb_arr::Vector{Int}`   : ``y``-axis tick labels (`LBP` ``\\%``).
- `N_tr_arr::Vector{Int}`   : ``x``-axis tick labels (`TRP` ``\\%``).
- `key::Symbol`             : Observable symbol (e.g., `:kurt`, `:cond`).
- `pred_tag::Symbol`        : Prediction method (e.g., `:RWP1`, `:RWP2`).
- `keyword::String`         : Interpolation origin cumulant identifier (e.g., `"skew"`, `"kurt"`).
- `overall_name::String`    : Suffix used in the output filename.
- `figs_dir::String`        : Output directory to save the figure.

# Keywords
- `key_tex::String=""`      : [``\\LaTeX``](https://www.latex-project.org/) label for ``y``-axis title part.
- `save_file::Bool=false`   : If true, saves a cropped PDF into `figs_dir`.
- `annotate::Bool=true`     : Print ``\\mathrm{BC}`` values in each cell.
- `cmap_name::String="viridis"` : [`Matplotlib` colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) name.
- `levels::Vector{Float64}=[0.6,0.8,0.9]` : Contour levels for ``\\mathrm{BC}``.
- `draw_contours::Bool=true` : Draw contour overlays if true.

# Side Effects
- Displays the formatted figure.
- If `save_file=true`, writes `heatmap_bc_<key>_<keyword>_<pred_tag>_<overall_name>.pdf` to `figs_dir`.
"""
function render_bhattacharyya_heatmap(
    bc_arr::Array{Float64, 2},
    conv_arr::Array{Bool,2},
    iter_arr::Array{Int,2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    annotate::Bool = true,
    cmap_name::String = "viridis",
    levels::Vector{Float64} = [0.6, 0.8, 0.9],
    draw_contours::Bool = false,
    iter_threshold::Int = 10
) :: Nothing

    fig, ax = PyPlot.subplots(1, 1, figsize=(5.6, 5.0), dpi=500)

    # Clamp defensively to [0,1] in case of tiny numeric drift
    bc_safe = clamp.(bc_arr, 0.0, 1.0)

    # Heatmap
    cmap = PyPlot.get_cmap(cmap_name)
    im = ax.imshow(bc_safe; cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")

    # Title and axis labels (match your style & ``\\LaTeX`` usage)
    # Left: observable (``\\LaTeX`` if provided) + keyword context, Right: method tag
    left_title  = isempty(key_tex) ? string(key) : key_tex

    pred_tag_map = Dict(
        "RWP1" => "\$\\mathcal{P}1\$",
        "RWP2" => "\$\\mathcal{P}2\$"
    )
    right_title = get(pred_tag_map, string(pred_tag), string(pred_tag))

    keyword_tex = ""
    if keyword == "susp"
        keyword_tex = "\$\\chi(\\kappa_t)\$"
    elseif keyword == "skew"
        keyword_tex = "\$S(\\kappa_t)\$"
    elseif keyword == "kurt"
        keyword_tex = "\$K(\\kappa_t)\$"
    else
        keyword_tex = keyword
    end
    # ax.set_title(left_title * " (from " * keyword_tex * " interpolation)  |  " * right_title)

    ax.set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    ax.set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")

    ax.set_xticks(collect(0:length(N_tr_arr)-1))
    ax.set_yticks(collect(0:length(N_lb_arr)-1))
    ax.set_xticklabels(N_tr_arr)
    ax.set_yticklabels(N_lb_arr)

    # Optional contour overlays at BC thresholds
    if draw_contours && !isempty(levels)
        try
            cs = ax.contour(bc_safe; levels=levels, colors="white",
                            linewidths=0.6, origin="lower")
            ax.clabel(cs; inline=true, fontsize=7, fmt="%.2f")
        catch err
            JobLoggerTools.warn_benji("Contour drawing failed: $err")
        end
    end

    # Optional numeric annotations
    if annotate
        pe = PyCall.pyimport("matplotlib.patheffects")
        # outline = [pe.withStroke(linewidth=0.1, foreground="white")]
        nrow, ncol = size(bc_safe)
        for i in 1:nrow, j in 1:ncol
            v = bc_safe[i, j]
            txt = string(round(v; digits=2))
            # Ensure readability: dark text on light cells, light text on dark cells
            txtcolor = v < 0.945 ? "white" : "black"
            fontweight = (txtcolor == "white") ? "bold" : "normal"
            # ax.text(j - 1, i - 1, txt;
            #         ha="center", va="center",
            #         color=txtcolor, fontsize=9, fontweight=fontweight,
            #         path_effects=outline, zorder=3)
            ax.text(j - 1, i - 1, txt;
                    ha="center", va="center",
                    color=txtcolor, fontsize=9, fontweight=fontweight,
                    zorder=3)
        end
    end

    overlay_nlsolve_marks!(ax, conv_arr, iter_arr; iter_threshold=iter_threshold, color="white", lw=0.9, pad=0.48)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.80, aspect=22, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    # cbar.set_label("Bhattacharyya Coefficient")

    fig.tight_layout()
    display(fig)

    # Save (optional) with pdfcrop integration
    basename = "heatmap_bc_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end

"""
    render_jsd_heatmap(
        jsd_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        annotate::Bool = true,
        cmap_name::String = "bwr",
        levels::Vector{Float64} = [0.2, 0.4, 0.6, 0.8],
        draw_contours::Bool = false
    ) -> Nothing

Render a single heatmap for the Jensen-Shannon divergence (``\\mathrm{JSD}`` in ``[0,1]``) for a given
observable, method, and interpolation origin (`keyword`).

- Lower is better (`0` = identical, `1` = worst).
- Colormap spans `0.0` to `1.0`.
- Optional numeric annotations per cell.
- Optional contour lines at given `levels`.

# Arguments
- `jsd_arr::Array{Float64,2}`: ``\\mathrm{JSD}`` matrix in ``[0,1]``.
- `N_lb_arr::Vector{Int}`   : ``y``-axis tick labels (`LBP` ``\\%``).
- `N_tr_arr::Vector{Int}`   : ``x``-axis tick labels (`TRP` ``\\%``).
- `key::Symbol`              : Observable (e.g., `:kurt`, `:cond`).
- `pred_tag::Symbol`         : Prediction method (e.g., `:RWP1`, `:RWP2`).
- `keyword::String`          : Interpolation origin cumulant identifier (e.g., `"skew"`, `"kurt"`).
- `overall_name::String`     : Suffix used in the output filename.
- `figs_dir::String`         : Output directory to save the figure.

# Keywords
- `key_tex::String=""`       : [``\\LaTeX``](https://www.latex-project.org/) label for y-axis title part.
- `save_file::Bool=false`    : If true, saves a cropped PDF into `figs_dir`.
- `annotate::Bool=true`      : Print ``\\mathrm{JSD}`` values in each cell.
- `cmap_name::String="bwr"`  : [`Matplotlib` colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) (default mirrors your `BC` function).
- `levels::Vector{Float64}=[0.2,0.4,0.6,0.8]` : Contour levels for ``\\mathrm{JSD}``.
- `draw_contours::Bool=false` : Draw contour overlays if true.

# Side Effects
- Displays the formatted figure.
- If `save_file=true`, writes `heatmap_jsd_<key>_<keyword>_<pred_tag>_<overall_name>.pdf` to `figs_dir`.
"""
function render_jsd_heatmap(
    jsd_arr::Array{Float64, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    annotate::Bool = true,
    cmap_name::String = "Greens",
    levels::Vector{Float64} = [0.2, 0.4, 0.6, 0.8],
    draw_contours::Bool = false
) :: Nothing

    fig, ax = PyPlot.subplots(1, 1, figsize=(5.6, 5.0), dpi=500)

    # Clamp defensively to [0,1]
    jsd_safe = clamp.(jsd_arr, 0.0, 1.0)

    # Heatmap
    cmap = PyPlot.get_cmap(cmap_name)
    im = ax.imshow(jsd_safe; cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")

    # Title and axis labels
    left_title  = isempty(key_tex) ? string(key) : key_tex
    right_title = string(pred_tag)
    ax.set_title(left_title * " (\$\\kappa_t\$ with " * keyword * ")  |  " * right_title)

    ax.set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    ax.set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")

    ax.set_xticks(collect(0:length(N_tr_arr)-1))
    ax.set_yticks(collect(0:length(N_lb_arr)-1))
    ax.set_xticklabels(N_tr_arr)
    ax.set_yticklabels(N_lb_arr)

    # Optional contour overlays
    if draw_contours && !isempty(levels)
        try
            cs = ax.contour(jsd_safe; levels=levels, colors="white",
                            linewidths=0.6, origin="lower")
            ax.clabel(cs; inline=true, fontsize=7, fmt="%.2f")
        catch err
            JobLoggerTools.warn_benji("Contour drawing failed: $err")
        end
    end

    # Optional numeric annotations
    if annotate
        pe = PyCall.pyimport("matplotlib.patheffects")
        outline = [pe.withStroke(linewidth=0.1, foreground="white")]
        nrow, ncol = size(jsd_safe)
        for i in 1:nrow, j in 1:ncol
            v = jsd_safe[i, j]
            txt = string(round(v; digits=2))
            # Low JSD (dark with many colormaps) → white text; otherwise black
            txtcolor = v > 0.75 ? "white" : "black"
            fontweight = (txtcolor == "white") ? "bold" : "normal"
            ax.text(j - 1, i - 1, txt;
                    ha="center", va="center",
                    color=txtcolor, fontsize=9, fontweight=fontweight,
                    path_effects=outline)
        end
    end

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.80, aspect=22, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    # cbar.set_label("Jensen–Shannon Divergence (base-2)")

    fig.tight_layout()
    display(fig)

    # Save (optional) with pdfcrop
    basename = "heatmap_jsd_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end

"""
    render_overlap_and_error_heatmaps_for_measurements(
        chk_arr::Array{Int, 2},
        err_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Render `CHK` and `ERR` heatmaps for a measurement-based evaluation at a fixed ``\\kappa`` at the single ensemble.

This variant mirrors [`render_overlap_and_error_heatmaps`](@ref) but interprets `keyword` as a
kappa token string (e.g., `"13580"`). The `CHK` panel title displays this as
``\\kappa = \\texttt{0.<keyword>}`` (e.g., ``\\kappa = 0.13580``), indicating that both `CHK` and `ERR` are
evaluated at that specific measurement point.

Two side-by-side heatmaps are produced using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl):
- Left: `CHK` matrix (categorical values `0`, `1`, `2` rendered as black-gray-white).
- Right: `ERR` matrix (error ratios) with a color scale from `1.0` to `7.0`.
  Values are overlaid as text for readability.

Axes:
- ``y``-axis: `N_lb_arr` (labeled-set percentages),
- ``x``-axis: `N_tr_arr` (training-set percentages).
Both subplots include ticks and colorbars.

# Arguments
- `chk_arr::Array{Int,2}`: Overlap quality map (`0`, `1`, `2`) on the (`LB`, `TR`) grid.
- `err_arr::Array{Float64,2}`: Error-ratio map on the same grid.
- `N_lb_arr::Vector{Int}`: ``y``-axis tick values (`LB` percentages).
- `N_tr_arr::Vector{Int}`: ``x``-axis tick values (`TR` percentages).
- `key::Symbol`: Observable being plotted (e.g., `:trM1`, `:Q2`).
- `pred_tag::Symbol`: Method/tag being visualized (e.g., `:T_P1`, `:Q_P2`).
- `keyword::String`: ``\\kappa`` token (e.g., `"13580"`) used for lookup in summaries and rendered as "``\\kappa = \\texttt{0.<keyword>}``".
- `overall_name::String`: Suffix used in the output filename.
- `figs_dir::String`: Directory to write the figure file.

# Keyword Arguments
- `key_tex::String = ""`: Optional [``\\LaTeX``](https://www.latex-project.org/) label for the `CHK` subplot (e.g., pretty-printed observable name).
- `save_file::Bool = false`: If `true`, saves `heatmap_<key>_<keyword>_<pred_tag>_<overall_name>.pdf`
  to `figs_dir` and, if available, crops it with [`pdfcrop`](https://ctan.org/pkg/pdfcrop).

# Side Effects
- Displays the figure via [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).
- Optionally writes a cropped PDF to disk when `save_file=true`.

# Notes
- This function is intended for the measurement pipeline where results are indexed by discrete ``\\kappa`` values of single ensemble
  Ensure that `keyword` matches the kappa token used in the summary dictionaries (e.g., `"13580"`).
- Color scale for `ERR` is fixed to `[1.0, 7.0]`; text annotations show the numeric values inside cells.
"""
function render_overlap_and_error_heatmaps_for_measurements(
    chk_arr::Array{Int, 2},
    err_arr::Array{Float64, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool=false
)

    fig, axs = PyPlot.subplots(1, 2, figsize=(10, 5), dpi=500)

    # === CHK colormap: 0=black, 1=gray, 2=white ===
    chk_colors = ["black", "gray", "white"]
    cmap_chk = PyPlot.matplotlib.colors.ListedColormap(chk_colors)
    bounds_chk = [0, 1, 2, 3]
    norm_chk = PyPlot.matplotlib.colors.BoundaryNorm(bounds_chk, cmap_chk.N)

    im1 = axs[1].imshow(chk_arr, cmap=cmap_chk, norm=norm_chk, origin="lower", aspect="auto")
    axs[1].set_title(key_tex * " (\$\\kappa = \$0." * keyword * ")")
    axs[1].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[1].set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")
    axs[1].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[1].set_yticks(collect(0:length(N_lb_arr)-1))
    axs[1].set_xticklabels(N_tr_arr)
    axs[1].set_yticklabels(N_lb_arr)

	cbar1 = fig.colorbar(im1, ax=axs[1], ticks=[0.5, 1.5, 2.5], shrink=0.75, aspect=20, pad=0.02)
	cbar1.ax.set_yticklabels(["0", "1", "2"])
    #cbar1.set_label("Overlap Quality")
	
    # === ERR heatmap ===
    cmap_err = PyPlot.get_cmap("gist_rainbow")
    im2 = axs[2].imshow(err_arr, cmap=cmap_err, origin="lower", vmin=1.0, vmax=7.0, aspect="auto")
    axs[2].set_title(pred_tag)
    axs[2].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[2].set_yticks([])
    axs[2].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[2].set_xticklabels(N_tr_arr)

    for i in 1:size(err_arr, 1), j in 1:size(err_arr, 2)
        val = round(err_arr[i,j]; digits=2)
	    # if val < 1.0 || val > 7.0
        pe = PyCall.pyimport("matplotlib.patheffects")
        outline = [pe.withStroke(linewidth=0.2, foreground="white")]
        
        axs[2].text(j - 1, i - 1, "$val",
            ha="center", va="center",
            color="black", fontsize=9,
            path_effects=outline)
	    # end
    end

	cbar2 = fig.colorbar(im2, ax=axs[2], shrink=0.75, aspect=20, pad=0.02)
    #cbar2.set_label("Error Ratio")
	
    fig.tight_layout()
    display(fig)

    basename = "heatmap_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end

"""
    render_overlap_type_b_and_error_heatmaps_for_measurements(
        ovl_arr::Array{Float64, 2},
        err_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        vmax_ovl::Float64 = 3.0
    ) -> Nothing

Render Type-B distance (`ovl_arr`) and `ERR` heatmaps for a measurement-based evaluation
at a fixed ``\\kappa`` within a single ensemble.

This mirrors [`render_overlap_and_error_heatmaps_for_measurements`](@ref) but uses a continuous
Type-B metric on the left panel:
```math
d \\equiv \\frac{|\\mu_{\\text{orig}} - \\mu_{\\text{pred}}|}
               {\\max(\\sigma_{\\text{orig}}, \\sigma_{\\text{floor}})}.
```
Color mapping is tuned for rapid visual assessment:

* ``d = 0`` → white,
* ``d \\approx 1`` → vivid color (attention threshold),
* ``1 - \\texttt{vmax\\_ovl}`` → smoothly intensifying,
* ``d > \\texttt{vmax\\_ovl}`` → shown via extended colorbar (triangle), with numeric annotations retained.

The title shows the kappa token as ``\\kappa = \\texttt{0.<keyword>}`` (e.g., `0.13580`).

# Arguments

* `ovl_arr`: Type-B distance matrix (`Float64`) on the (`LB`, `TR`) grid.
* `err_arr`: Error-ratio matrix (`Float64`) on the same grid.
* `N_lb_arr`, `N_tr_arr`: Percent tick lists for ``y``/``x`` axes.
* `key`, `pred_tag`: Observable/method labels.
* `keyword`: Kappa token string (e.g., `"13580"`) rendered as ``\\kappa = \\texttt{0.<keyword>}``.
* `overall_name`, `figs_dir`: Suffix and output directory for the saved figure.

# Keywords

* `key_tex`: Optional [`\\LaTeX`](https://www.latex-project.org/) label prefix for the left subplot title.
* `save_file`: If `true`, saves `heatmap_typeB_<key>_<keyword>_<pred_tag>_<overall_name>.pdf`
  and crops with [`pdfcrop`](https://ctan.org/pkg/pdfcrop) if available.
* `vmax_ovl`: Upper bound for Type-B color scaling (default `3.0`).

# Side Effects

* Displays the figure via [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl); optionally writes a cropped PDF.

# Notes

* Designed for the measurement pipeline where results are indexed by discrete ``\\kappa`` values.
* The `ERR` panel uses the fixed range ``[1.0, 7.0]`` with per-cell numeric overlays.
"""
function render_overlap_type_b_and_error_heatmaps_for_measurements(
    ovl_arr::Array{Float64, 2},
    err_arr::Array{Float64, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    vmax_ovl::Float64 = 3.0
)
    fig, axs = PyPlot.subplots(1, 2, figsize=(10, 5), dpi=500)

    # ==== OVL (Type-B) heatmap: 0 -> white, 1 accent, >vmax extend ====

    cmap_ovl = PyPlot.get_cmap("YlOrRd").copy()
    cmap_ovl.set_under("white")
    cmap_ovl.set_over("#7f0000")  # deep red for values > vmax_ovl
    im1 = axs[1].imshow(
        ovl_arr; cmap=cmap_ovl, origin="lower", vmin=0.0, vmax=vmax_ovl, aspect="auto"
    )
    # axs[1].set_title(key_tex * " (\$\\kappa = \$0." * keyword * ")")
    axs[1].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[1].set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")
    axs[1].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[1].set_yticks(collect(0:length(N_lb_arr)-1))
    axs[1].set_xticklabels(N_tr_arr)
    axs[1].set_yticklabels(N_lb_arr)

    # Per-cell annotations for OVL

    # pe = PyCall.pyimport("matplotlib.patheffects")
    # outline = [pe.withStroke(linewidth=0.2, foreground="white")]
    for i in 1:size(ovl_arr, 1), j in 1:size(ovl_arr, 2)
        val = round(ovl_arr[i, j]; digits=2)
        text_color = val >= 2.0 ? "white" : "black"
        fontweight = (text_color == "white") ? "bold" : "normal"
        # axs[1].text(j - 1, i - 1, string(val);
        #     ha="center", va="center", color="black", fontsize=9, path_effects=outline)
        axs[1].text(j - 1, i - 1, string(val);
            ha="center", va="center", color=text_color, fontsize=9, fontweight=fontweight)
    end

    cbar1 = fig.colorbar(
        im1, ax=axs[1], shrink=0.75, aspect=20, pad=0.02, extend="max"
    )
    cbar1.set_ticks([0.0, 0.5, 1.0, 2.0, vmax_ovl])
    cbar1.set_ticklabels(["0", "0.5", "1", "2", string(vmax_ovl)])

    # ==== ERR heatmap (same spec as usual) ====

    cmap_err = PyPlot.get_cmap("gist_rainbow")
    im2 = axs[2].imshow(err_arr; cmap=cmap_err, origin="lower", vmin=1.0, vmax=7.0, aspect="auto")
    # axs[2].set_title(pred_tag)
    axs[2].set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    axs[2].set_yticks([])
    axs[2].set_xticks(collect(0:length(N_tr_arr)-1))
    axs[2].set_xticklabels(N_tr_arr)

    for i in 1:size(err_arr, 1), j in 1:size(err_arr, 2)
        val = round(err_arr[i, j]; digits=2)
        color_val = (val < 1.41 || val > 4.0) ? "white" : "black"
        fontweight = (color_val == "white") ? "bold" : "normal"
        # axs[2].text(j - 1, i - 1, string(val);
        #     ha="center", va="center", color="black", fontsize=9, path_effects=outline)
        axs[2].text(j - 1, i - 1, string(val);
            ha="center", va="center", color=color_val, fontsize=9, fontweight=fontweight)
    end

    cbar2 = fig.colorbar(im2, ax=axs[2], shrink=0.75, aspect=20, pad=0.02)

    fig.tight_layout()
    display(fig)

    basename = "heatmap_typeB_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end
    return nothing
end

"""
    render_bhattacharyya_heatmap_for_measurements(
        bc_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        annotate::Bool = true,
        cmap_name::String = "viridis",
        levels::Vector{Float64} = [0.6, 0.8, 0.9],
        draw_contours::Bool = true
    ) -> Nothing

Render a single-panel heatmap of the Bhattacharyya coefficient (``\\mathrm{BC}`` in ``[0,1]``)
for a measurement-based evaluation at a fixed ``\\kappa`` for single ensemble.

- Title shows the kappa token as ``\\kappa = \\texttt{0.<keyword>}``.
- Colormap fixed to ``[0,1]`` across figures for comparability.
- Optional per-cell annotations and contour overlays.

# Arguments
- `bc_arr::Array{Float64,2}` : ``\\mathrm{BC}`` matrix (`LB` ``\\times`` `TR` grid), values expected in ``[0,1]``.
- `N_lb_arr::Vector{Int}`    : ``y``-axis tick labels (`LB` percentages).
- `N_tr_arr::Vector{Int}`    : ``x``-axis tick labels (`TR` percentages).
- `key::Symbol`              : Observable (e.g., `:trM1`, `:Q2`).
- `pred_tag::Symbol`         : Method/tag (e.g., `:T_P1`, `:Q_P2`).
- `keyword::String`          : ``\\kappa`` token (e.g., `"13580"`) rendered as "``\\kappa = \\texttt{0.<keyword>}``".
- `overall_name::String`     : Suffix for output file.
- `figs_dir::String`         : Output directory.

# Keywords
- `key_tex::String=""`       : Optional [``\\LaTeX``](https://www.latex-project.org/) label for ``y``-axis/title left part.
- `save_file::Bool=false`    : If true, saves/crops PDF.
- `annotate::Bool=true`      : Print numeric ``\\mathrm{BC}`` per cell.
- `cmap_name::String="viridis"` : [`Matplotlib` colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) name.
- `levels::Vector{Float64}=[0.6,0.8,0.9]` : Contour levels.
- `draw_contours::Bool=true` : Draw contour overlays.

# Side Effects
- Displays the figure.
- If `save_file=true`, writes `heatmap_bc_<key>_<keyword>_<pred_tag>_<overall_name>.pdf`.
"""
function render_bhattacharyya_heatmap_for_measurements(
    bc_arr::Array{Float64, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    annotate::Bool = true,
    cmap_name::String = "viridis",
    levels::Vector{Float64} = [0.6, 0.8, 0.9],
    draw_contours::Bool = false
) :: Nothing

    fig, ax = PyPlot.subplots(1, 1, figsize=(5.6, 5.0), dpi=500)

    # Defensive clamp to [0,1]
    bc_safe = clamp.(bc_arr, 0.0, 1.0)

    # Heatmap
    cmap = PyPlot.get_cmap(cmap_name)
    im = ax.imshow(bc_safe; cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")

    # Title & axes (mirror your measurement renderer, with κ token)
    left_title  = isempty(key_tex) ? string(key) : key_tex
    right_title = string(pred_tag)
    # ax.set_title(left_title * " (\$\\kappa = \$0." * keyword * ")  |  " * right_title)

    ax.set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    ax.set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")

    ax.set_xticks(collect(0:length(N_tr_arr)-1))
    ax.set_yticks(collect(0:length(N_lb_arr)-1))
    ax.set_xticklabels(N_tr_arr)
    ax.set_yticklabels(N_lb_arr)

    # Optional contours
    if draw_contours && !isempty(levels)
        try
            cs = ax.contour(bc_safe; levels=levels, colors="white",
                            linewidths=0.6, origin="lower")
            ax.clabel(cs; inline=true, fontsize=7, fmt="%.2f")
        catch err
            JobLoggerTools.warn_benji("Contour drawing failed: $err")
        end
    end

    # Optional per-cell annotations
    if annotate
        pe = PyCall.pyimport("matplotlib.patheffects")
        # outline = [pe.withStroke(linewidth=0.1, foreground="white")]
        nrow, ncol = size(bc_safe)
        for i in 1:nrow, j in 1:ncol
            v = bc_safe[i, j]
            txt = string(round(v; digits=2))
            txtcolor = v < 0.945 ? "white" : "black"
            fontweight = (txtcolor == "white") ? "bold" : "normal"
            # ax.text(j - 1, i - 1, txt;
            #         ha="center", va="center",
            #         color=txtcolor, fontsize=9,
            #         path_effects=outline)
            ax.text(j - 1, i - 1, txt;
                    ha="center", va="center",
                    color=txtcolor, fontsize=9,
                    fontweight=fontweight)
        end
    end

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.80, aspect=22, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    # cbar.set_label("Bhattacharyya Coefficient")

    fig.tight_layout()
    display(fig)

    # Save with pdfcrop (optional)
    basename = "heatmap_bc_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end

"""
    render_jsd_heatmap_for_measurements(
        jsd_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        annotate::Bool = true,
        cmap_name::String = "viridis",
        levels::Vector{Float64} = [0.2, 0.4, 0.6, 0.8],
        draw_contours::Bool = true
    ) -> Nothing

Render a single-panel heatmap of the Jensen-Shannon divergence (``\\mathrm{JSD}`` in ``[0,1]``)
for a measurement-based evaluation at a fixed κ (no interpolation).

- Title shows the kappa token as "``\\kappa = \\texttt{0.<keyword>}``".
- Lower is better (`0` = identical, `1` = worst).
- Colormap fixed to `[0,1]` across figures for comparability.
- Optional per-cell annotations and contour overlays.

# Arguments
- `jsd_arr::Array{Float64,2}` : ``\\mathrm{JSD}`` matrix (`LB` ``\\times`` `TR` grid) in ``[0,1]``.
- `N_lb_arr::Vector{Int}`     : ``y``-axis tick labels (`LB` percentages).
- `N_tr_arr::Vector{Int}`     : ``x``-axis tick labels (`TR` percentages).
- `key::Symbol`               : Observable (e.g., `:trM1`, `:Q2`).
- `pred_tag::Symbol`          : Method/tag (e.g., `:T_P1`, `:Q_P2`).
- `keyword::String`           : ``\\kappa`` token (e.g., `"13580"`) rendered as "``\\kappa = \\texttt{0.<keyword>}``".
- `overall_name::String`      : Suffix for output file.
- `figs_dir::String`          : Output directory.

# Keywords
- `key_tex::String=""`        : Optional [``\\LaTeX``](https://www.latex-project.org/) label for ``y``-axis/title left part.
- `save_file::Bool=false`     : If true, saves/crops PDF.
- `annotate::Bool=true`       : Print numeric ``\\mathrm{JSD}`` per cell.
- `cmap_name::String="viridis"` : [`Matplotlib` colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) name.
- `levels::Vector{Float64}=[0.2,0.4,0.6,0.8]` : Contour levels.
- `draw_contours::Bool=true`  : Draw contour overlays.

# Side Effects
- Displays the figure.
- If `save_file=true`, writes `heatmap_jsd_<key>_<keyword>_<pred_tag>_<overall_name>.pdf`.
"""
function render_jsd_heatmap_for_measurements(
    jsd_arr::Array{Float64, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    keyword::String,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    annotate::Bool = true,
    cmap_name::String = "viridis",
    levels::Vector{Float64} = [0.2, 0.4, 0.6, 0.8],
    draw_contours::Bool = false
) :: Nothing

    fig, ax = PyPlot.subplots(1, 1, figsize=(5.6, 5.0), dpi=500)

    # Clamp defensively
    jsd_safe = clamp.(jsd_arr, 0.0, 1.0)

    # Heatmap
    cmap = PyPlot.get_cmap(cmap_name)
    im = ax.imshow(jsd_safe; cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")

    # Title & axes (κ token)
    left_title  = isempty(key_tex) ? string(key) : key_tex
    right_title = string(pred_tag)
    ax.set_title(left_title * " (\$\\kappa = \$0." * keyword * ")  |  " * right_title * "  (JSD↓)")

    ax.set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    ax.set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")

    ax.set_xticks(collect(0:length(N_tr_arr)-1))
    ax.set_yticks(collect(0:length(N_lb_arr)-1))
    ax.set_xticklabels(N_tr_arr)
    ax.set_yticklabels(N_lb_arr)

    # Optional contours
    if draw_contours && !isempty(levels)
        try
            cs = ax.contour(jsd_safe; levels=levels, colors="white",
                            linewidths=0.6, origin="lower")
            ax.clabel(cs; inline=true, fontsize=7, fmt="%.2f")
        catch err
            JobLoggerTools.warn_benji("Contour drawing failed: $err")
        end
    end

    # Optional per-cell annotations
    if annotate
        pe = PyCall.pyimport("matplotlib.patheffects")
        outline = [pe.withStroke(linewidth=0.35, foreground="black")]
        nrow, ncol = size(jsd_safe)
        for i in 1:nrow, j in 1:ncol
            v = jsd_safe[i, j]
            txt = string(round(v; digits=2))
            txtcolor = v < 0.5 ? "white" : "black"
            ax.text(j - 1, i - 1, txt;
                    ha="center", va="center",
                    color=txtcolor, fontsize=8,
                    path_effects=outline)
        end
    end

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.80, aspect=22, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    # cbar.set_label("Jensen–Shannon Divergence (base-2)")

    fig.tight_layout()
    display(fig)

    # Save with pdfcrop (optional)
    basename = "heatmap_jsd_$(key)_$(keyword)_$(pred_tag)_$(overall_name)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end

"""
    sci_round(
        x::Real
    ) -> String

Convert a floating-point number `x` into a simplified scientific-notation string
without any fractional part in the mantissa.

The mantissa is rounded to the nearest integer and followed by the exponent in
``\\texttt{e} \\pm \\texttt{n}`` format. This produces compact expressions like `"2e-13"` instead of
`"2.1e-13"` or `"1.0e-13"`.

- If `x = 2.1e-13`, returns `"2e-13"`.
- If `x = 3.5e-13`, returns `"4e-13"`.
- If `x = -9.8e5`, returns `"-1e+6"`.
- If `x = 0`, returns `"0"`.

# Returns
A `String` representing the integer-rounded scientific notation of `x`.
"""
function sci_round(
    x::Real
)
    if x == 0
        return "0"
    end
    exp = floor(Int, log10(abs(x)))
    base = round(Int, x / 10.0^exp)
    return string(sign(x) < 0 ? "-" : "", base, "e", exp >= 0 ? "+" : "", exp)
end

"""
    build_nlsolve_grids(
        nlsolve_status::Dict,
        labels::Vector{String},
        trains::Vector{String},
        solver_name::AbstractString;
        default_converged::Bool = false,
        default_residual::Float64 = NaN,
        default_iterations::Int = -1
    ) -> Tuple{Array{Bool,2}, Array{Float64,2}, Array{Int,2}}

Construct 2D grids of [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) outcomes for a given `solver_name`, shaped as
`(length(labels), length(trains))` to match your heatmap conventions.

- `conv_arr[i,j]` is `true` if the solver converged for `(labels[i], trains[j])`.
- `resn_arr[i,j]` is the corresponding residual norm (`Float64`).
- `iter_arr[i,j]` is the number of iterations (`Int`).

If a cell is missing (no file, no solver section, or missing fields), the
`default_converged`, `default_residual`, and `default_iterations` values are used.
"""
function build_nlsolve_grids(
    nlsolve_status::Dict,
    labels::Vector{String},
    trains::Vector{String},
    solver_name::AbstractString;
    default_converged::Bool = false,
    default_residual::Float64 = NaN,
    default_iterations::Int = -1
)::Tuple{Array{Bool,2}, Array{Float64,2}, Array{Int,2}}

    nl = nlsolve_status  # alias
    nL = length(labels)
    nT = length(trains)

    conv_arr = fill(default_converged, nL, nT)
    resn_arr = fill(default_residual,  nL, nT)
    iter_arr = fill(default_iterations, nL, nT)

    for (ii, lab) in pairs(labels)
        row = get(nl, lab, nothing)
        row === nothing && continue
        for (jj, tr) in pairs(trains)
            cell = get(row, tr, nothing)
            cell === nothing && continue
            st = get(cell, String(solver_name), nothing)
            st === nothing && continue
            # st :: NamedTuple{(:converged, :residual_norm, :iterations), ...}
            conv_arr[ii, jj] = st.converged
            resn_arr[ii, jj] = st.residual_norm
            iter_arr[ii, jj] = st.iterations
        end
    end

    return conv_arr, resn_arr, iter_arr
end

"""
    collect_available_solvers(
        nlsolve_status::Dict
    ) -> Vector{String}

Return a sorted list of all solver names present within the nested
`nlsolve_status` dictionary structure.

This function scans every `(label, train)` entry and collects the keys of each
solver section (e.g., `"Broyden"`, `"TrustRegion"`, `"Newton"`), ensuring
uniqueness and alphabetic order.

Typical usage:
```julia
available = collect_available_solvers(nlsolve_status)
println_benji(available)
# → ["Broyden", "Newton", "TrustRegion"]
```

# Arguments
- `nlsolve_status::Dict`: Nested dictionary of [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) results, where each
`(label, train)` cell contains a dictionary of solver names mapped to
`NamedTuples` of outcome data.

# Returns
- A sorted `Vector{String}` listing all solver names that appear in the given
  `nlsolve_status` dictionary.
"""
function collect_available_solvers(
    nlsolve_status::Dict
)::Vector{String}
    solvers = Set{String}()
    for (_, row) in nlsolve_status
        for (_, cell) in row
            # cell :: Dict{String, NamedTuple{...}}
            union!(solvers, keys(cell))
        end
    end
    return sort!(collect(solvers))
end

"""
    render_nlsolve_convergence_heatmap(
        conv_arr::Array{Bool, 2},
        resn_arr::Array{Float64, 2},
        iter_arr::Array{Int, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        solver_name::AbstractString,
        overall_name::AbstractString,
        figs_dir::AbstractString;
        save_file::Bool = false,
        show_all_residuals::Bool = false,
        iter_threshold::Int = 10
    ) -> Nothing

Render an [`NLsolve.jl`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/) convergence heatmap across labeled (`LBP`) and training (`TRP`)
percentage grids.

Visualization rules:
- Cell color encodes convergence: white (`true`) vs black (`false`).
- Residual norms are printed in white text on failed (black) cells. When
  `show_all_residuals=true`, residuals are printed on all cells (in white),
  remaining invisible on white cells by design.
- If a cell converged and `iterations ≥ iter_threshold`, the iteration count is
  annotated in black text inside the white cell.

# Arguments
- `conv_arr`: Boolean matrix of convergence results.
- `resn_arr`: Matrix of residual norms, same shape as `conv_arr`.
- `iter_arr`: Matrix of iteration counts, same shape as `conv_arr`.
- `N_lb_arr`, `N_tr_arr`: Axis tick values for `LBP`% (``y``) and `TRP`% (``x``).
- `solver_name`: Solver identifier (e.g., `"nlsolve_f_solver_FULL-LBOG-ULOG"`).
- `overall_name`: Suffix used in the output filename.
- `figs_dir`: Directory to save the figure.

# Keywords
- `save_file`: If true, saves as PDF and crops with [`pdfcrop`](https://ctan.org/pkg/pdfcrop) if available.
- `show_all_residuals`: If true, prints residuals on all cells; otherwise only
  on failed cells.
- `iter_threshold`: Minimum iteration count that triggers iteration annotation
  on converged cells.

# Returns
`Nothing`. Displays (and optionally saves) the rendered heatmap.
"""
function render_nlsolve_convergence_heatmap(
    conv_arr::Array{Bool, 2},
    resn_arr::Array{Float64, 2},
    iter_arr::Array{Int, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    solver_name::AbstractString,
    overall_name::AbstractString,
    figs_dir::AbstractString;
    save_file::Bool = false,
    show_all_residuals::Bool = false,
    iter_threshold::Int = 10
) :: Nothing

    JobLoggerTools.assert_benji(
        size(conv_arr) == size(resn_arr),
        "conv_arr and resn_arr must have the same size"
    )
    JobLoggerTools.assert_benji(
        size(conv_arr) == size(iter_arr),
        "conv_arr and iter_arr must have the same size"
    )
    JobLoggerTools.assert_benji(
        size(conv_arr, 1) == length(N_lb_arr),
        "row size must match N_lb_arr"
    )
    JobLoggerTools.assert_benji(
        size(conv_arr, 2) == length(N_tr_arr),
        "col size must match N_tr_arr"
    )

    # Bool -> Int (false=0, true=1)
    conv_int = Int.(conv_arr)

    fig, ax = PyPlot.subplots(1, 1, figsize=(5.2, 5.0), dpi=500)

    # 0=black, 1=white
    cmap_bw   = PyPlot.matplotlib.colors.ListedColormap(["black", "white"])
    bounds_bw = [0, 0.5, 1]
    norm_bw   = PyPlot.matplotlib.colors.BoundaryNorm(bounds_bw, cmap_bw.N)

    im = ax.imshow(conv_int, cmap=cmap_bw, norm=norm_bw, origin="lower", aspect="auto")

    # Title (strip prefix if present)
    prefix = "nlsolve_f_solver_"
    ttl = startswith(solver_name, prefix) ? solver_name[length(prefix)+1:end] : solver_name
    # ax.set_title(ttl)

    # Axes / ticks
    ax.set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    ax.set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")
    ax.set_xticks(collect(0:length(N_tr_arr)-1))
    ax.set_yticks(collect(0:length(N_lb_arr)-1))
    ax.set_xticklabels(N_tr_arr)
    ax.set_yticklabels(N_lb_arr)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], shrink=0.8, aspect=24, pad=0.02)
    cbar.ax.set_yticklabels(["No", "OK"])

    # Text overlays
    # NOTE: If you use LaTeX strings like \texttt{...}, ensure PyPlot.rc("text", usetex=true).
    for i in 1:size(resn_arr, 1), j in 1:size(resn_arr, 2)
        converged = conv_arr[i, j]
        residual  = resn_arr[i, j]
        iters     = iter_arr[i, j]

        # 1) Residuals on failed cells (or everywhere if requested) — white text
        # if (!converged) || show_all_residuals
        #     txt_res = isnan(residual) ? "NaN" : @sprintf("%.1e", residual)
        #     ax.text(j - 1, i - 1, "\$\\texttt{$txt_res}\$";
        #             ha="center", va="center",
        #             color="white", fontsize=6)
        # end
        if (!converged) || show_all_residuals
            txt_res = isnan(residual) ? "NaN" : sci_round(residual)
            ax.text(j - 1, i - 1, "\$\\texttt{$txt_res}\$";
                    ha="center", va="center",
                    color="white", fontsize=8)
        end

        # 2) Iteration count on converged cells if over threshold — black text
        if converged && iters >= iter_threshold
            ax.text(j - 1, i - 1, "\$\\texttt{$(string(iters))}\$";
                    ha="center", va="center",
                    color="black", fontsize=8)
        end
    end

    fig.tight_layout()
    display(fig)

    if save_file
        base = "nlsolve_conv_$(replace(solver_name, '/' => '-'))_$(overall_name)"
        resfile = joinpath(figs_dir, base * ".pdf")
        cropped = joinpath(figs_dir, base * "-crop.pdf")
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    return nothing
end

end  # module HeatmapsRebekahMiriam