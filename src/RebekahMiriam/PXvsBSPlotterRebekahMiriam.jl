# ============================================================================
# src/RebekahMiriam/PXvsBSPlotterRebekahMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module PXvsBSPlotterRebekahMiriam

import ..PyPlot

import ..Sarah.JobLoggerTools

"""
    build_flat_plot_dict(
        key::Symbol,
        orig_tag::Symbol,
        pred_tag1::Symbol,
        pred_tag2::Symbol,
        keyword::String,
        new_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}}
    ) -> Dict{String, Array{Float64,2}}

Construct a flattened dictionary of average and error matrices for plotting, given a specific
`keyword` and observable `key`.

This utility reformats selected entries from a 4-key dictionary `new_dict`—keyed by
`(observable, kind, tag, keyword)`—into a flat `Dict{String, Array}` that mimics the naming
convention used in [`Deborah.Rebekah.PXvsBSPlotter`](@ref) (e.g., `"kurt:P_P2:avg"`).

It includes data from two prediction methods (`pred_tag1`, `pred_tag2`) as well as the original
reference tag (`orig_tag`) for the specified observable `key` and criterion `keyword`.

Key naming convention in the returned dictionary:
- Predictions: `"\$key:Y_<pred_tag>:avg"` and `"\$key:Y_<pred_tag>:err"`
- Original:    `"\$key:Y:<orig_tag>:avg"` and `"\$key:Y:<orig_tag>:err"`
  (Note the underscore for predictions vs colon for the original tag, matching [`Deborah.Rebekah.PXvsBSPlotter`](@ref) usage.)

# Arguments
- `key`: Observable symbol (e.g., `:kurt`).
- `orig_tag`: Original/reference tag to include (previously hard-coded as `:RWBS`).
- `pred_tag1`: First prediction tag to include (e.g., `:RWP1`).
- `pred_tag2`: Second prediction tag to include (e.g., `:RWP2`).
- `keyword`: Selector used in `new_dict` (e.g., `"susp"`, `"skew"`, `"kurt"` as interpolation criterion; or a ``\\kappa`` token like `"13580"` in measurement flows).
- `new_dict`: Source dictionary keyed by `(key, kind, tag, keyword)` and valued with `labels` ``\\times`` `trains` matrices.

# Returns
- A flattened dictionary with keys like `"kurt:Y_P2:avg"` or `"kurt:Y:RWBS:err"` mapped to matrices,
  suitable for direct use in plotting.
"""
function build_flat_plot_dict(
    key::Symbol,
    orig_tag::Symbol,
    pred_tag1::Symbol,
    pred_tag2::Symbol,
    keyword::String,
    new_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}}
)::Dict{String, Array{Float64,2}}

    flat = Dict{String, Array{Float64,2}}()
    for pred_tag in (pred_tag1, pred_tag2)
        flat["$key:Y_$(pred_tag):avg"] = new_dict[(key, :avg, pred_tag, keyword)]
        flat["$key:Y_$(pred_tag):err"] = new_dict[(key, :err, pred_tag, keyword)]
    end
    flat["$key:Y:$(orig_tag):avg"] = new_dict[(key, :avg, orig_tag, keyword)]
    flat["$key:Y:$(orig_tag):err"] = new_dict[(key, :err, orig_tag, keyword)]
    return flat
end

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
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Plot `P1`, `P2`, and original values versus training set percentage for a fixed label ratio,  
in a [`Deborah.Miriam`](@ref) analysis.

This function is specific to the [`Deborah.RebekahMiriam`](@ref).  
It visualizes the behavior of a given observable (`key`) across varying training set sizes (`trains_ext_int`),  
while fixing the labeled set size to a specific value `label_pct`.

The plotted values are extracted from `new_dict`, containing average and error arrays for:
- baseline method (`suffix_orig`)
- first estimation method (`suffix1`)
- second estimation method (`suffix2`)

Each curve is plotted with its own marker style and color,  
and slightly shifted on the ``x``-axis to improve readability.  
The ``y``-axis reflects the observable evaluated at a specific interpolation point  
``\\kappa_t`` determined by `keyword`.

# Arguments
- `key::String`: Name of the observable to be plotted (e.g., `"skew"`, `"kurt"`).
- `suffix_orig::String`: Suffix used in the key for baseline values (e.g., `"RWBS"`).
- `suffix1::String`: Suffix for the first interpolation method (e.g., `"RWP1"`).
- `suffix2::String`: Suffix for the second interpolation method (e.g., `"RWP2"`).
- `label_pct::Int`: The label set percentage (used to select the row index from the 2D arrays).
- `new_dict::Dict{String, Array{Float64,2}}`: Dictionary containing all relevant observable averages and error bars.
- `trains_ext_int::Vector{Int}`: Training set percentages to use for the ``x``-axis.
- `labels_int::Vector{Int}`: Labeled set percentages, used to resolve the index for `label_pct`.
- `keyword::String`: The origin cumulant that was used to determine the interpolation point ``\\kappa_t``.  
  For example, `"kurt"` means that ``\\kappa_t`` was selected based on kurtosis behavior,  
  and the observable in `key` was evaluated at that point.
- `overall_name::String`: Suffix used for output filenames.
- `figs_dir::String`: Output directory to save the figure.

# Keyword Arguments
- `key_tex::String = ""`: Optional [``\\LaTeX``](https://www.latex-project.org/) string used as ``y``-axis label.
- `save_file::Bool = false`: If `true`, the figure is saved as a PDF and cropped using [`pdfcrop`](https://ctan.org/pkg/pdfcrop) (if available).

# Side Effects
- Displays a [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure comparing three estimation methods at the specified `label_pct`.
- If `save_file=true`, saves a PDF named `plot_<key>_<keyword>_<overall_name>_LBP_<label_pct>.pdf` into `figs_dir`.
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
    keyword::String,
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

    # --- Axes labels ---
    ax.set_xlabel("\$\\mathcal{R}_{\\mathrm{TR}}~\\mathrm{(training~set)}\$")
    ax.set_ylabel(key_tex)
    ax.set_title("\$\\mathcal{R}_{\\mathrm{LB}} = $(label_pct)\\%\$" * " (\$\\kappa_t\$ with " * keyword * ")")

    # --- X shift for clarity ---
    sorted = sort(unique(trains_ext_int))
    offset = (length(sorted) ≥ 2) ? minimum(diff(sorted)) * 0.15 : 0.5

    x_bs  = trains_ext_int
    x_p1x = trains_ext_int .- offset
    x_p2x = trains_ext_int .+ offset

	ax.errorbar(x_bs, y_bs, yerr=y_bs_err,
	    fmt="s", label="Original", color="blue", alpha=1.0,
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

    basename = "plot_$(key)_$(keyword)_$(overall_name)_LBP_$(label_pct)"
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
    plot_PX_BS_vs_trains_for_measurements(
        key::String,
        suffix_orig::String,
        suffix1::String,
        suffix2::String,
        label_pct::Int,
        new_dict::Dict{String, Array{Float64,2}},
        trains_ext_int::Vector{Int},
        labels_int::Vector{Int},
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Plot `P1`, `P2`, and original values versus training-set percentage at a fixed label ratio,
for the single-ensemble-measurement-based workflow.

This variant mirrors [`plot_PX_BS_vs_trains`](@ref) but interprets `keyword` as a
kappa token string (e.g., `"13580"`). The plot title explicitly shows
``\\kappa`` `= 0.<keyword>` (e.g., ``\\kappa =`` `"0.13580"`), indicating that all values are
evaluated at that specific measurement point rather than an interpolation-derived ``\\kappa_t``.

The function visualizes a selected observable `key` across `trains_ext_int`, while
holding the labeled-set size at `label_pct`. Three curves are shown:
- baseline/original (`suffix_orig`),
- first estimation method (`suffix1`),
- second estimation method (`suffix2`).

Data are loaded from `new_dict`, which contains average and error matrices indexed
by `(label_index, train_index)` using flattened keys such as:
- `"\$key:Y_\$suffix:avg"` / `"\$key:Y_\$suffix:err"` for predictions,
- `"\$key:Y:\$suffix_orig:avg"` / `"\$key:Y:\$suffix_orig:err"` for the original.
The special case `key == "Deborah"` omits the `"\$key:"` prefix to match [`Deborah.DeborahCore`](@ref) naming.

# Arguments
- `key::String`: Observable name to plot (e.g., `"trM1"`, `"Q2"`).
- `suffix_orig::String`: Baseline/original tag suffix (e.g., `"Y_BS"`).
- `suffix1::String`: First method suffix (e.g., `"Y_P1"`).
- `suffix2::String`: Second method suffix (e.g., `"Y_P2"`).
- `label_pct::Int`: Labeled-set percentage; selects the row from the matrices.
- `new_dict::Dict{String, Array{Float64,2}}`: Flattened dictionary with avg/err arrays.
- `trains_ext_int::Vector{Int}`: Training-set percentages for the ``x``-axis.
- `labels_int::Vector{Int}`: Available labeled-set percentages; used to resolve `label_pct`.
- `keyword::String`: ``\\kappa`` token used for figure titling and output naming (e.g., `"13580"`).
- `overall_name::String`: Suffix used in the output filename.
- `figs_dir::String`: Directory for saving the figure.

# Keyword Arguments
- `key_tex::String = ""`: Optional [``\\LaTeX``](https://www.latex-project.org/) ``y``-axis label for the observable.
- `save_file::Bool = false`: If `true`, saves
  `plot_<key>_<keyword>_<overall_name>_LBP_<label_pct>.pdf` and crops with [`pdfcrop`](https://ctan.org/pkg/pdfcrop) if available.

# Side Effects
- Displays a [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure comparing three methods at the specified `label_pct`.
- Optionally writes the figure to disk when `save_file=true`.

# Notes
- This function assumes the measurement pipeline where results are indexed by discrete ``\\kappa`` values.
  Ensure that `new_dict` was built accordingly (e.g., via [`build_flat_plot_dict`](@ref)-style utilities).
- Curves are horizontally offset slightly to improve readability; error bars are drawn for each series.
"""
function plot_PX_BS_vs_trains_for_measurements(
    key::String,
    suffix_orig::String,
    suffix1::String,
    suffix2::String,
    label_pct::Int,
    new_dict::Dict{String, Array{Float64,2}},
    trains_ext_int::Vector{Int},
    labels_int::Vector{Int},
    keyword::String,
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

    # --- Axes labels ---
    ax.set_xlabel("\$\\mathcal{R}_{\\mathrm{TR}}~\\mathrm{(training~set)}\$")
    ax.set_ylabel(key_tex)
    ax.set_title("\$\\mathcal{R}_{\\mathrm{LB}} = $(label_pct)\\%\$" * " (\$\\kappa =\$0." * keyword * ")")

    # --- X shift for clarity ---
    sorted = sort(unique(trains_ext_int))
    offset = (length(sorted) ≥ 2) ? minimum(diff(sorted)) * 0.15 : 0.5

    x_bs  = trains_ext_int
    x_p1x = trains_ext_int .- offset
    x_p2x = trains_ext_int .+ offset

	ax.errorbar(x_bs, y_bs, yerr=y_bs_err,
	    fmt="s", label="Original", color="blue", alpha=1.0,
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

    basename = "plot_$(key)_$(keyword)_$(overall_name)_LBP_$(label_pct)"
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
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Plot `P1`, `P2`, and original values versus labeled set percentage for a fixed training set size,  
in a [`Deborah.Miriam`](@ref) analysis.

This function is specific to the [`Deborah.RebekahMiriam`](@ref).  
It visualizes the behavior of a given observable (`key`) across varying label set sizes (`labels_int`),  
while fixing the training set percentage to `train_pct`.

The plotted values are extracted from `new_dict`, containing average and error arrays for:
- baseline method (`suffix_orig`)
- first estimation method (`suffix1`)
- second estimation method (`suffix2`)

Each curve is plotted with its own marker style and color,  
and slightly shifted on the ``x``-axis to improve readability.  
The ``y``-axis reflects the observable evaluated at a specific interpolation point  
``\\kappa_t`` determined by `keyword`.

# Arguments
- `key::String`: Name of the observable to be plotted (e.g., `"skew"`, `"kurt"`).
- `suffix_orig::String`: Suffix used in the key for baseline values (e.g., `"RWBS"`).
- `suffix1::String`: Suffix for the first interpolation method (e.g., `"RWP1"`).
- `suffix2::String`: Suffix for the second interpolation method (e.g., `"RWP2"`).
- `train_pct::Int`: The training set percentage (used to select the column index from the 2D arrays).
- `new_dict::Dict{String, Array{Float64,2}}`: Dictionary containing all relevant observable averages and error bars.
- `labels_int::Vector{Int}`: Label set percentages to use for the ``x``-axis.
- `trains_ext_int::Vector{Int}`: Training set percentages, used to resolve the index for `train_pct`.
- `keyword::String`: The origin cumulant that was used to determine the interpolation point ``\\kappa_t``.  
  For example, `"kurt"` means that ``\\kappa_t`` was selected based on kurtosis behavior,  
  and the observable in `key` was evaluated at that point.
- `overall_name::String`: Suffix used for output filenames.
- `figs_dir::String`: Output directory to save the figure.

# Keyword Arguments
- `key_tex::String = ""`: Optional [``\\LaTeX``](https://www.latex-project.org/) string used as ``y``-axis label.
- `save_file::Bool = false`: If `true`, the figure is saved as a PDF and cropped using [`pdfcrop`](https://ctan.org/pkg/pdfcrop) (if available).

# Side Effects
- Displays a [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure comparing three estimation methods at the specified `train_pct`.
- If `save_file=true`, saves a PDF named `plot_<key>_<keyword>_<overall_name>_TRP_<train_pct>.pdf` into `figs_dir`.
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
    keyword::String,
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
	    fmt="s", label="Original", color="blue", alpha=1.0,
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
    ax.set_title("\$\\mathcal{R}_{\\mathrm{TR}} = $(train_pct)\\%\$" * " (\$\\kappa_t\$ with " * keyword * ")")
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

    basename = "plot_$(key)_$(keyword)_$(overall_name)_TRP_$(train_pct)"
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
    plot_PX_BS_vs_labels_for_measurements(
        key::String,
        suffix_orig::String,
        suffix1::String,
        suffix2::String,
        train_pct::Int,
        new_dict::Dict{String, Array{Float64,2}},
        labels_int::Vector{Int},
        trains_ext_int::Vector{Int},
        keyword::String,
        overall_name::String,
        figs_dir::String;
        key_tex::String = "",
        save_file::Bool = false
    ) -> Nothing

Plot `P1`, `P2`, and original values versus labeled-set percentage at a fixed training-set size,
for the single-ensemble-measurement-based workflow.

This variant mirrors [`plot_PX_BS_vs_labels`](@ref) but interprets `keyword` as a
kappa token string (e.g., `"13580"`). The plot title shows
``\\kappa = `` `0.<keyword>` (e.g., ``\\kappa = 0.13580``), indicating that all series are
evaluated at that specific measurement point rather than an interpolation-derived ``\\kappa_t``.

The function visualizes a selected observable `key` across `labels_int`, while fixing
the training-set percentage to `train_pct`. Three curves are drawn:
- baseline/original (`suffix_orig`),
- first estimation method (`suffix1`),
- second estimation method (`suffix2`).

Values are fetched from `new_dict`, which stores average and error matrices
indexed by `(label_index, train_index)` behind flattened keys:
- `"\$key:Y_\$suffix:avg"` / `"\$key:Y_\$suffix:err"` for predictions,
- `"\$key:Y:\$suffix_orig:avg"` / `"\$key:Y:\$suffix_orig:err"` for the original.
The special case `key == "Deborah"` omits the `"\$key:"` prefix to match legacy naming.

# Arguments
- `key::String`: Observable to plot (e.g., `"trM1"`, `"Q2"`).
- `suffix_orig::String`: Baseline/original tag suffix (e.g., `"Y_BS"`).
- `suffix1::String`: First method suffix (e.g., `"Y_P1"`).
- `suffix2::String`: Second method suffix (e.g., `"Y_P2"`).
- `train_pct::Int`: Training-set percentage; selects a column from the matrices.
- `new_dict::Dict{String, Array{Float64,2}}`: Flattened dictionary containing avg/err arrays.
- `labels_int::Vector{Int}`: Labeled-set percentages for the ``x``-axis.
- `trains_ext_int::Vector{Int}`: Training-set percentages; used to resolve `train_pct`.
- `keyword::String`: ``\\kappa`` token for titling and output naming (e.g., `"13580"`).
- `overall_name::String`: Suffix used in the output filename.
- `figs_dir::String`: Directory to save the figure.

# Keyword Arguments
- `key_tex::String = ""`: Optional [``\\LaTeX``](https://www.latex-project.org/) ``y``-axis label for the observable.
- `save_file::Bool = false`: If `true`, saves
  `plot_<key>_<keyword>_<overall_name>_TRP_<train_pct>.pdf` and crops with [`pdfcrop`](https://ctan.org/pkg/pdfcrop) if available.

# Side Effects
- Displays a [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure comparing three methods at the specified `train_pct`.
- Optionally writes a PDF to disk when `save_file=true`.

# Notes
- Assumes measurement summaries indexed by discrete ``\\kappa`` tokens; ensure `new_dict` was built accordingly.
- Curves are slightly offset along ``x`` for readability; error bars are shown for each series.
"""
function plot_PX_BS_vs_labels_for_measurements(
    key::String,
    suffix_orig::String,
    suffix1::String,
    suffix2::String,
    train_pct::Int,
    new_dict::Dict{String, Array{Float64,2}},
    labels_int::Vector{Int},
    trains_ext_int::Vector{Int},
    keyword::String,
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
	    fmt="s", label="Original", color="blue", alpha=1.0,
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
    ax.set_title("\$\\mathcal{R}_{\\mathrm{TR}} = $(train_pct)\\%\$" * " (\$\\kappa = \$0." * keyword * ")")
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

    basename = "plot_$(key)_$(keyword)_$(overall_name)_TRP_$(train_pct)"
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

end  # module PXvsBSPlotterRebekahMiriam