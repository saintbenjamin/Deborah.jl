# ============================================================================
# src/Rebekah/PXvsBSPlotter.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PXvsBSPlotter

import ..PyPlot

import ..Sarah.JobLoggerTools

"""
    plot_PX_BS_vs_trains(
        key::String,
        suffix_orig::String,
        suffix1::String,
        suffix2::String,
        label_pct::Int,
        new_dict::Dict{String, Array{Float64,2}},
        trains_ext_int::Vector{Int},
        labels_int::Vector{Int},
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Plot `P1`, `P2`, and original results versus training percentage for a fixed label size.

This function compares three estimation methods for a given observable `key` at a fixed labeled set percentage `label_pct`.  
It extracts the average values and error bars from `new_dict` using three suffixes:
- `suffix_orig`: baseline (e.g., `"Y_BS"`)
- `suffix1`: first method (e.g., `"Y_P1"`)
- `suffix2`: second method (e.g., `"Y_P2"`)

All three curves are plotted against `trains_ext_int` (``x``-axis), with distinct markers, colors, and error bars.  
The ``y``-values are extracted for the corresponding row of `label_pct` in `labels_int`.

# Arguments
- `key::String`: Observable name (e.g., `"TrM1"`, `"Deborah"`). If `"Deborah"`, the dict key omits prefix.
- `suffix_orig::String`: Suffix for the baseline method (e.g., `"Y_BS"`).
- `suffix1::String`: Suffix for the first estimation method (e.g., `"Y_P1"`).
- `suffix2::String`: Suffix for the second estimation method (e.g., `"Y_P2"`).
- `label_pct::Int`: Label set percentage to fix the row index for ``y``-values.
- `new_dict::Dict{String, Array{Float64,2}}`: Dictionary containing precomputed average and error matrices.
- `trains_ext_int::Vector{Int}`: Training set percentages (``x``-axis).
- `labels_int::Vector{Int}`: Label set percentages (used to resolve row index).
- `overall_name::String`: Identifier string used in the saved filename.
- `figs_dir::String`: Output directory to save the figure.

# Keyword Arguments
- `key_tex::String = ""`: [``\\LaTeX``](https://www.latex-project.org/) string for ``y``-axis label.
- `save_file::Bool = false`: Whether to save the figure as PDF. If `true`, tries to crop it using [`pdfcrop`](https://ctan.org/pkg/pdfcrop).

# Behavior
- X-axis: `trains_ext_int` with left/right offsets for clarity.
- Y-axis: Observable value with error bars.
- Marker shapes: `"s"` (square) for `suffix_orig`, `"^"` for `suffix1`, `"o"` for `suffix2`.
- Raises an error if `label_pct` not found or if any ``y``-values are `NaN`/`Inf`.

# Side Effects
- Displays a `PyPlot` figure comparing all three methods.
- If `save_file=true`, saves a PDF under `figs_dir/plot_<key>_<overall_name>_LBP_<label_pct>.pdf`.
  If [`pdfcrop`](https://ctan.org/pkg/pdfcrop) is available, the output is cropped automatically.

"""
function plot_PX_BS_vs_trains(
    key::String,
    suffix_orig::String,
    suffix1::String,
    suffix2::String,
    label_pct::Int,
    new_dict::Dict{String, Array{Float64,2}},
    trains_ext_int::Vector{Int},
    labels_int::Vector{Int},
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool=false
)
    # --- Safe label index resolution ---
    label_idx = findfirst(==(label_pct), labels_int)
    label_idx === nothing && JobLoggerTools.error_benji("Label percentage $label_pct not found in labels_int")

    fig, ax = PyPlot.subplots(figsize=(6, 3), dpi=500)

    # --- Load data ---
    if key == "Deborah"
        y_p1x     = new_dict["Y_$suffix1:avg"][label_idx, :]
        y_p1x_err = new_dict["Y_$suffix1:err"][label_idx, :]
        y_p2x     = new_dict["Y_$suffix2:avg"][label_idx, :]
        y_p2x_err = new_dict["Y_$suffix2:err"][label_idx, :]
        y_bs      = new_dict["Y:$suffix_orig:avg"][label_idx, :]
        y_bs_err  = new_dict["Y:$suffix_orig:err"][label_idx, :]
    else
        y_p1x     = new_dict["$key:Y_$suffix1:avg"][label_idx, :]
        y_p1x_err = new_dict["$key:Y_$suffix1:err"][label_idx, :]
        y_p2x     = new_dict["$key:Y_$suffix2:avg"][label_idx, :]
        y_p2x_err = new_dict["$key:Y_$suffix2:err"][label_idx, :]
        y_bs      = new_dict["$key:Y:$suffix_orig:avg"][label_idx, :]
        y_bs_err  = new_dict["$key:Y:$suffix_orig:err"][label_idx, :]
    end

    combined_y = vcat(y_bs..., y_p1x..., y_p2x...)
    if any(!isfinite, combined_y) || isempty(combined_y)
        JobLoggerTools.error_benji("Invalid data (NaN/Inf or empty) detected in y values.")
    end

    # --- X shift for clarity ---
    sorted = sort(unique(trains_ext_int))
    offset = (length(sorted) ≥ 2) ? minimum(diff(sorted)) * 0.15 : 0.5

    x_bs  = trains_ext_int
    x_p1x = trains_ext_int .- offset
    x_p2x = trains_ext_int .+ offset

	ax.errorbar(x_bs, y_bs, yerr=y_bs_err,
	    fmt="s", label="Orig.", color="blue", alpha=1.0,
	    capsize=6,
	    markerfacecolor="none", markeredgecolor="blue")
	
	ax.errorbar(x_p1x, y_p1x, yerr=y_p1x_err,
	    fmt="^", label=suffix1, color="orange", alpha=1.0,
	    capsize=6,
	    markerfacecolor="none", markeredgecolor="orange")
	
	ax.errorbar(x_p2x, y_p2x, yerr=y_p2x_err,
	    fmt="o", label=suffix2, color="red", alpha=1.0,
	    capsize=6,
	    markerfacecolor="none", markeredgecolor="red")

    ax.set_xlabel("\$\\mathcal{R}_{\\mathrm{TR}}~\\mathrm{(training~set)}\$")
    ax.set_ylabel(key_tex)
    ax.set_title("\$\\mathcal{R}_{\\mathrm{LB}} = $(label_pct)\\%\$")
	ax.legend(
	    loc="upper left", 
	    bbox_to_anchor=(1.02, 0.7),
	    borderaxespad=0.0,
	    frameon=false,
    	labelspacing=1.5 
	)
    ax.grid(true)

    fig.tight_layout()
    display(fig)

    basename = "plot_$(key)_$(overall_name)_LBP_$(label_pct)"
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
    plot_PX_BS_vs_labels(
        key::String,
        suffix_orig::String,
        suffix1::String,
        suffix2::String,
        train_pct::Int,
        new_dict::Dict{String, Array{Float64,2}},
        labels_int::Vector{Int},
        trains_ext_int::Vector{Int},
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Plot `P1`, `P2`, and original results versus labeled set percentage for a fixed training set size.

This function compares three estimation methods for a given observable `key` at a fixed training set percentage `train_pct`.  
It extracts the average values and error bars from `new_dict` using three suffixes:
- `suffix_orig`: baseline (e.g., `"Y_BS"`)
- `suffix1`: first method (e.g., `"Y_P1"`)
- `suffix2`: second method (e.g., `"Y_P2"`)

All three curves are plotted against `labels_int` (``x``-axis), with distinct markers, colors, and error bars.  
The ``y``-values are extracted for the corresponding column of `train_pct` in `trains_ext_int`.

# Arguments
- `key::String`: Observable name (e.g., `"TrM1"`, `"Deborah"`). If `"Deborah"`, the dict key omits prefix.
- `suffix_orig::String`: Suffix for the baseline method (e.g., `"Y_BS"`).
- `suffix1::String`: Suffix for the first estimation method (e.g., `"Y_P1"`).
- `suffix2::String`: Suffix for the second estimation method (e.g., `"Y_P2"`).
- `train_pct::Int`: Training set percentage to fix the column index for ``y``-values.
- `new_dict::Dict{String, Array{Float64,2}}`: Dictionary containing precomputed average and error matrices.
- `labels_int::Vector{Int}`: Label set percentages (``x``-axis values).
- `trains_ext_int::Vector{Int}`: Training percentages (used to resolve column index).
- `overall_name::String`: Identifier string used in the saved filename.
- `figs_dir::String`: Output directory to save the figure.

# Keyword Arguments
- `key_tex::String = ""`: [``\\LaTeX``](https://www.latex-project.org/) string for ``y``-axis label.
- `save_file::Bool = false`: Whether to save the figure as PDF. If `true`, attempts to crop using [`pdfcrop`](https://ctan.org/pkg/pdfcrop).

# Behavior
- ``X``-axis: `labels_int`, with slight left/right shifts for `suffix1` and `suffix2` for clarity.
- ``Y``-axis: Observable value with error bars.
- Marker shapes: `"s"` (square) for `suffix_orig`, `"^"` for `suffix1`, `"o"` for `suffix2`.
- Raises an error if `train_pct` not found.
"""
function plot_PX_BS_vs_labels(
    key::String,
    suffix_orig::String,
    suffix1::String,
    suffix2::String,
    train_pct::Int,
    new_dict::Dict{String, Array{Float64,2}},
    labels_int::Vector{Int},
    trains_ext_int::Vector{Int},
    overall_name::String,
    figs_dir::String;
    key_tex::String = "",
    save_file::Bool=false
)
    # --- Find safe index ---
    idx = findfirst(==(train_pct), trains_ext_int)
    idx === nothing && JobLoggerTools.error_benji("Training percentage $train_pct not found in train index list.")
    train_idx = idx

    fig, ax = PyPlot.subplots(figsize=(6, 3), dpi=500)

    if key == "Deborah"
        y_p1x     = new_dict["Y_$suffix1:avg"][:, train_idx]
        y_p1x_err = new_dict["Y_$suffix1:err"][:, train_idx]
        y_p2x     = new_dict["Y_$suffix2:avg"][:, train_idx]
        y_p2x_err = new_dict["Y_$suffix2:err"][:, train_idx]
        y_bs      = new_dict["Y:$suffix_orig:avg"][:, train_idx]
        y_bs_err  = new_dict["Y:$suffix_orig:err"][:, train_idx]
    else
        y_p1x     = new_dict["$key:Y_$suffix1:avg"][:, train_idx]
        y_p1x_err = new_dict["$key:Y_$suffix1:err"][:, train_idx]
        y_p2x     = new_dict["$key:Y_$suffix2:avg"][:, train_idx]
        y_p2x_err = new_dict["$key:Y_$suffix2:err"][:, train_idx]
        y_bs      = new_dict["$key:Y:$suffix_orig:avg"][:, train_idx]
        y_bs_err  = new_dict["$key:Y:$suffix_orig:err"][:, train_idx]
    end

    combined_y = vcat(y_bs..., y_p1x..., y_p2x...)
    if any(!isfinite, combined_y) || isempty(combined_y)
        JobLoggerTools.error_benji("Invalid data (NaN/Inf or empty) detected in y values.")
    end
	
    sorted_labels = sort(unique(labels_int))
    offset = (length(sorted_labels) ≥ 2) ? minimum(diff(sorted_labels)) * 0.15 : 0.5

    x_bs  = labels_int
    x_p1x = labels_int .- offset
    x_p2x = labels_int .+ offset
	
	ax.errorbar(x_bs, y_bs, yerr=y_bs_err,
	    fmt="s", label="Orig.", color="blue", alpha=1.0,
	    capsize=6,
	    markerfacecolor="none", markeredgecolor="blue")
	
	ax.errorbar(x_p1x, y_p1x, yerr=y_p1x_err,
	    fmt="^", label=suffix1, color="orange", alpha=1.0,
	    capsize=6,
	    markerfacecolor="none", markeredgecolor="orange")
	
	ax.errorbar(x_p2x, y_p2x, yerr=y_p2x_err,
	    fmt="o", label=suffix2, color="red", alpha=1.0,
	    capsize=6,
	    markerfacecolor="none", markeredgecolor="red")

    ax.set_xlabel("\$\\mathcal{R}_{\\mathrm{LB}}~\\mathrm{(labeled~set)}\$")
	ax.set_ylabel(key_tex)
    ax.set_title("\$\\mathcal{R}_{\\mathrm{TR}} = $(train_pct)\\%\$")
	ax.legend(
	    loc="upper left", 
	    bbox_to_anchor=(1.02, 0.7),
	    borderaxespad=0.0,
	    frameon=false,
    	labelspacing=1.5 
	)
    ax.grid(true)

    fig.tight_layout()
    display(fig)

    basename = "plot_$(key)_$(overall_name)_TRP_$(train_pct)"
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

end  # PXvsBSPlotter