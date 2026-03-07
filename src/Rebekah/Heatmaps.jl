# ============================================================================
# src/Rebekah/Heatmaps.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Heatmaps

import ..PyPlot
import ..PyCall

import ..Sarah.JobLoggerTools

"""
    render_overlap_and_error_heatmaps(
        chk_arr::Array{Int, 2},
        err_arr::Array{Float64, 2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Render side-by-side heatmaps showing overlap (`CHK`) and error ratio (`ERR`) matrices
for a given observable and prediction method.

This function generates a two-panel static figure using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl):
- Left panel (`CHK`): overlap matrix with categorical values (`0` = black, `1` = gray, `2` = white).
- Right panel (`ERR`): error ratio matrix, colormapped from `1.0` to `7.0`.
  If any values fall outside this range, the out-of-range values are shown as overlaid text.

Axis ticks are labeled using `N_tr_arr` (``x``-axis) and `N_lb_arr` (``y``-axis),
with [``\\LaTeX``](https://www.latex-project.org/)-rendered labels using `key_tex`. Title includes the observable and method tag.

# Arguments
- `chk_arr::Array{Int,2}`: Overlap matrix (each entry `0`, `1`, or `2`).
- `err_arr::Array{Float64,2}`: Error ratio matrix.
- `N_lb_arr::Vector{Int}`: Labeled set percentage values for ``y``-axis ticks.
- `N_tr_arr::Vector{Int}`: Training set percentage values for ``x``-axis ticks.
- `key::Symbol`: Observable identifier (e.g., `:TrM1`, `:TrM4`).
- `pred_tag::Symbol`: Prediction method (e.g., `:Y_P1`, `:Y_P2`).
- `overall_name::String`: Identifier used in the output filename.
- `figs_dir::String`: Directory path to save the resulting figure.

# Keyword Arguments
- `key_tex::String=""`: [``\\LaTeX``](https://www.latex-project.org/)-formatted string for ``y``-axis label (e.g., ``\\mathcal{R}_{\\mathrm{LB}}``).
- `save_file::Bool=false`: Whether to save the figure as a PDF file under `figs_dir`.

# Side Effects
- Displays the generated heatmap figure using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).
- If `save_file=true`, saves a PDF under the filename: `figs_dir/heatmap_kurt_<overall_name>.pdf`.
"""
function render_overlap_and_error_heatmaps(
    chk_arr::Array{Int, 2},
    err_arr::Array{Float64, 2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
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
    axs[1].set_title(key_tex)
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

    basename = "heatmap_$(key)_$(pred_tag)_$(overall_name)"
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
        bc_arr::Array{Float64,2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        annotate::Bool = true,
        cmap_name::String = "viridis",
        levels::Vector{Float64} = [0.6, 0.8, 0.9],
        draw_contours::Bool = true
    ) -> Nothing

Render a single heatmap for the Bhattacharyya coefficient (``\\mathrm{BC}`` in ``[0,1]``).
- Colormap spans `0.0` (worst) to `1.0` (best).
- Optional numeric annotations per cell.
- Optional contour lines at given `levels` for quick visual thresholds.

# Arguments
- `bc_arr`      : 2D array of Bhattacharyya coefficients in ``[0,1]``.
- `N_lb_arr`    : ``y``-axis tick labels (`LBP` ``\\%``).
- `N_tr_arr`    : ``x``-axis tick labels (`TRP` ``\\%``).
- `key`         : Observable symbol (e.g., `:TrM1`).
- `pred_tag`    : Prediction method symbol (e.g., `:Y_P2`).
- `overall_name`: Identifier used in filename.
- `figs_dir`    : Output directory.

# Keywords
- `key_tex=""`     : [``\\LaTeX``](https://www.latex-project.org/) label (e.g., ``\\chi``).
- `save_file=false`: If true, saves a cropped PDF in `figs_dir`.
- `annotate=true`  : If true, prints ``\\mathrm{BC}`` values in each cell.
- `cmap_name="viridis"`: [`Matplotlib` colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) name.
- `levels=[0.6,0.8,0.9]`: Contour levels for ``\\mathrm{BC}``.
- `draw_contours=true`  : Whether to draw contour overlays.

# Side Effects
- Displays the heatmap; optionally writes `<figs_dir>/heatmap_bc_<key>_<pred_tag>_<overall_name>.pdf`.

"""
function render_bhattacharyya_heatmap(
    bc_arr::Array{Float64,2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    annotate::Bool = true,
    cmap_name::String = "viridis",
    levels::Vector{Float64} = [0.6, 0.8, 0.9],
    draw_contours::Bool = true
) :: Nothing

    fig, ax = PyPlot.subplots(1, 1, figsize=(5.2, 5.0), dpi=500)

    # Clamp to [0,1] defensively (small numeric drift)
    bc_safe = clamp.(bc_arr, 0.0, 1.0)

    # Heatmap
    cmap = PyPlot.get_cmap(cmap_name)
    im = ax.imshow(bc_safe; cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")

    # Labels / ticks
    # Title: left = observable (``\\LaTeX`` if provided), right = method tag
    title_left  = isempty(key_tex) ? string(key) : key_tex
    title_right = string(pred_tag)
    ax.set_title("$(title_left)  |  $(title_right)")
    ax.set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    ax.set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")

    ax.set_xticks(collect(0:length(N_tr_arr)-1))
    ax.set_yticks(collect(0:length(N_lb_arr)-1))
    ax.set_xticklabels(N_tr_arr)
    ax.set_yticklabels(N_lb_arr)

    # Contours at BC thresholds (optional)
    if draw_contours && !isempty(levels)
        try
            cs = ax.contour(bc_safe; levels=levels, colors="white",
                            linewidths=0.6, origin="lower")
            ax.clabel(cs; inline=true, fontsize=7, fmt="%.2f")
        catch err
            JobLoggerTools.warn_benji("Contour drawing failed: $err")
        end
    end

    # Numeric annotations (optional)
    if annotate
        pe = PyCall.pyimport("matplotlib.patheffects")
        # outline = [pe.withStroke(linewidth=0.4, foreground="black")]
        nrow, ncol = size(bc_safe)
        for i in 1:nrow, j in 1:ncol
            val = round(bc_safe[i,j]; digits=2)
            # Choose contrasting text color for readability
            txtcolor = (bc_safe[i,j] < 0.95) ? "white" : "black"
            # ax.text(j-1, i-1, string(val);
            #         ha="center", va="center",
            #         color=txtcolor, fontsize=8,
            #         path_effects=outline)
            ax.text(j-1, i-1, string(val);
                    ha="center", va="center",
                    color=txtcolor, fontsize=8)
        end
    end

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=22, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    # cbar.set_label("Bhattacharyya Coefficient")

    fig.tight_layout()
    display(fig)

    # Save (optional) with pdfcrop integration
    basename = "heatmap_bc_$(key)_$(pred_tag)_$(overall_name)"
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
        jsd_arr::Array{Float64,2},
        N_lb_arr::Vector{Int},
        N_tr_arr::Vector{Int},
        key::Symbol,
        pred_tag::Symbol,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false,
        annotate::Bool = true,
        cmap_name::String = "viridis",
        levels::Vector{Float64} = [0.2, 0.4, 0.6, 0.8],
        draw_contours::Bool = true
    ) -> Nothing

Render a single heatmap for the Jensen-Shannon divergence (``\\mathrm{JSD}`` in ``[0,1]``, base-2).
- Lower is better (`0` = identical, `1` = worst).
- Optional per-cell annotations and contour overlays.

# Arguments
- `jsd_arr`     : 2D array of ``\\mathrm{JSD}`` values in ``[0,1]``.
- `N_lb_arr`    : ``y``-axis tick labels (`LBP` ``\\%``).
- `N_tr_arr`    : ``x``-axis tick labels (`TRP` ``\\%``).
- `key`         : Observable symbol (e.g., `:TrM1`).
- `pred_tag`    : Prediction method symbol (e.g., `:Y_P1`).
- `overall_name`: Identifier used in filename.
- `figs_dir`    : Output directory.

# Keywords
- `key_tex=""`     : [``\\LaTeX``](https://www.latex-project.org/) label (e.g., ``\\chi``).
- `save_file=false`: If true, saves a cropped PDF in `figs_dir`.
- `annotate=true`  : If true, prints ``\\mathrm{JSD}`` values in each cell.
- `cmap_name="viridis"`: [`Matplotlib` colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html) name.
- `levels=[0.2,0.4,0.6,0.8]`: Contour levels for ``\\mathrm{JSD}``.
- `draw_contours=true`  : Whether to draw contour overlays.

# Side Effects
- Displays the heatmap; optionally writes `<figs_dir>/heatmap_jsd_<key>_<pred_tag>_<overall_name>.pdf`.

"""
function render_jsd_heatmap(
    jsd_arr::Array{Float64,2},
    N_lb_arr::Vector{Int},
    N_tr_arr::Vector{Int},
    key::Symbol,
    pred_tag::Symbol,
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool = false,
    annotate::Bool = true,
    cmap_name::String = "viridis",
    levels::Vector{Float64} = [0.2, 0.4, 0.6, 0.8],
    draw_contours::Bool = true
) :: Nothing

    fig, ax = PyPlot.subplots(1, 1, figsize=(5.2, 5.0), dpi=500)

    # Clamp to [0,1] defensively
    jsd_safe = clamp.(jsd_arr, 0.0, 1.0)

    # Heatmap
    cmap = PyPlot.get_cmap(cmap_name)
    im = ax.imshow(jsd_safe; cmap=cmap, origin="lower", vmin=0.0, vmax=1.0, aspect="auto")

    # Labels / ticks
    title_left  = isempty(key_tex) ? string(key) : key_tex
    title_right = string(pred_tag)
    ax.set_title("$(title_left)  |  $(title_right)  (JSD, lower is better)")
    ax.set_xlabel("\$\\mathcal{R}_{\\textrm{TR}}~[\\%]\$")
    ax.set_ylabel("\$\\mathcal{R}_{\\textrm{LB}}~[\\%]\$")

    ax.set_xticks(collect(0:length(N_tr_arr)-1))
    ax.set_yticks(collect(0:length(N_lb_arr)-1))
    ax.set_xticklabels(N_tr_arr)
    ax.set_yticklabels(N_lb_arr)

    # Contours (optional)
    if draw_contours && !isempty(levels)
        try
            cs = ax.contour(jsd_safe; levels=levels, colors="white",
                            linewidths=0.6, origin="lower")
            ax.clabel(cs; inline=true, fontsize=7, fmt="%.2f")
        catch err
            JobLoggerTools.warn_benji("Contour drawing failed: $err")
        end
    end

    # Numeric annotations (optional)
    if annotate
        pe = PyCall.pyimport("matplotlib.patheffects")
        outline = [pe.withStroke(linewidth=0.4, foreground="black")]
        nrow, ncol = size(jsd_safe)
        for i in 1:nrow, j in 1:ncol
            v = jsd_safe[i,j]
            val = round(v; digits=2)
            # Darker cells at low JSD -> use white text for contrast
            txtcolor = (v < 0.5) ? "white" : "black"
            ax.text(j-1, i-1, string(val);
                    ha="center", va="center",
                    color=txtcolor, fontsize=8,
                    path_effects=outline)
        end
    end

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, aspect=22, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    # cbar.set_label("Jensen–Shannon Divergence (base-2)")

    fig.tight_layout()
    display(fig)

    # Save (optional) with pdfcrop integration
    basename = "heatmap_jsd_$(key)_$(pred_tag)_$(overall_name)"
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

end  # module Heatmaps