# ============================================================================
# src/Rahab/HistogramOrigML.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module HistogramOrigML

import ..PyPlot

"""
    plot_histogram_orig_vs_ml(
        trace_data::Dict{String, Vector{Vector{Float64}}},
        trace_idx::Int,
        nbins::Int,
        overall_name::String;
        subset::AbstractString="all",
        x_min::Union{Nothing, Real}=nothing,
        x_max::Union{Nothing, Real}=nothing,
        print_clipping::Bool=true,
        clipping_report_prefix::AbstractString="",
        outfile::AbstractString="histogram_bins.dat",
        include_out_of_range_counts_in_file::Bool=true,
        save_file::Bool=false,
        plot_dir::AbstractString=""
    ) -> Nothing

Plot histograms comparing original vs. ML-predicted trace data for a given observable index,
optionally clip the plotting range, optionally save the histogram *plot* as a cropped PDF
(with a heatmap-style filename convention), and always save binned counts to a `.dat` file.

This function compares two datasets (OG vs. ML) selected by `subset`, renders overlaid histograms
on a common bin grid, and writes the bin counts to `outfile`. If `x_min` and/or `x_max` are provided,
values outside the chosen range are discarded (clipped) before histogramming; discarded counts are
reported (optionally) and can be written to the `.dat` output. If `save_file=true`, the histogram
figure is saved to PDF (and cropped via `pdfcrop` when available) using a rule-based basename
that includes `subset` and `overall_name`.

# Arguments
- `trace_data`: Dictionary containing vectors of traces, keyed by:
    - `"Y_tr"` : Training set values,
    - `"Y_bc"` : Bias-correction set values,
    - `"Y_ul"` : Unlabeled set values (original),
    - `"YP_ul"`: Unlabeled set values (ML-predicted),
    - `"YP_bc"`: Bias-correction set values (ML-predicted) (required when `subset="bc"`).
- `trace_idx`: Index of the observable in each `trace_data[key]` entry.
  For example, if `trace_idx` corresponds to ``\\mathrm{Tr}\\,M^{-n}``, then `trace_idx=1,2,3,4`
  typically represent different inverse-trace powers or related observables in your pipeline.
- `nbins`: Number of histogram bins.
- `overall_name`: Suffix used to construct the output histogram *plot* filename when `save_file=true`,
  analogous to the heatmap naming convention used elsewhere. (Does not affect `outfile`.)

# Keyword Arguments
- `subset`: Select which subset to compare. Allowed values (case-insensitive):
    - `"all"`: (default) Use the original combined behavior:
        - OG = `vcat(Y_tr, Y_bc, Y_ul)`
        - ML = `vcat(Y_tr, Y_bc, YP_ul)`
    - `"tr"`: Training-only comparison:
        - OG = `Y_tr`
        - ML = `Y_tr`
      (This is expected to match identically by construction.)
    - `"bc"`: Bias-correction-only comparison:
        - OG = `Y_bc`
        - ML = `YP_bc`
      (Requires `trace_data["YP_bc"]` to exist.)
    - `"ul"`: Unlabeled-only comparison:
        - OG = `Y_ul`
        - ML = `YP_ul`
- `x_min`: If provided, forces the histogram minimum ``x``-range (values below are discarded).
- `x_max`: If provided, forces the histogram maximum ``x``-range (values above are discarded).
- `print_clipping`: If `true`, prints a short report of discarded points due to clipping.
- `clipping_report_prefix`: Optional prefix prepended to clipping report lines (useful in batch logs).
- `outfile`: Output filename for the tab-delimited bin-count table. This `.dat` file is always written.
- `include_out_of_range_counts_in_file`: If `true`, appends two extra columns (`OG_oob`, `ML_oob`)
  holding the number of discarded points for each dataset (same value repeated per row for convenience).
- `save_file`: If `true`, saves the histogram *plot* as a PDF using a rule-based filename and (if available)
  runs `pdfcrop` to produce a cropped PDF.
- `plot_dir`: Output directory for the histogram *plot* PDF when `save_file=true`.
  If empty, the current directory (`"."`) is used.

# Behavior
- First, selects the datasets to compare based on `subset`:
    - `subset="all"`:
        - OG = `vcat(Y_tr, Y_bc, Y_ul)`
        - ML = `vcat(Y_tr, Y_bc, YP_ul)`
    - `subset="tr"`:
        - OG = `Y_tr`
        - ML = `Y_tr`
    - `subset="bc"`:
        - OG = `Y_bc`
        - ML = `YP_bc`
    - `subset="ul"`:
        - OG = `Y_ul`
        - ML = `YP_ul`
- Determines a common plotting/binning range:
    - If `x_min`/`x_max` are both `nothing`, uses data-driven min/max from both datasets.
    - Otherwise uses the user-specified bound(s) and fills any missing side from the data-driven bound.
- If `x_min` and/or `x_max` are provided, values outside `[final_min, final_max]` are discarded before
  histogramming. The number of discarded points is computed separately for OG and ML, printed when
  `print_clipping=true`, and optionally written to the `.dat` file (see `include_out_of_range_counts_in_file`).
- Renders overlaid histograms using a shared `bin_edges` grid. Legend labels are adjusted by `subset`:
    - `subset="all"`: legend shows `OG` and `ML`.
    - Otherwise: legend shows `OG-<SUBSET>` and `ML-<SUBSET>` (e.g., `OG-UL`, `ML-UL`).
- If `save_file=true`, saves the histogram plot as a cropped PDF (when `pdfcrop` is available) using
  a heatmap-style basename convention:
    - If `trace_idx == 1`: `histogram_pbp_<subset>_<overall_name>.pdf`
    - Else:             `histogram_trdinv<trace_idx>_<subset>_<overall_name>.pdf`
  The file is written under `plot_dir` (or `"."` if `plot_dir` is empty).
- Always writes the binned histogram counts to `outfile` as a tab-delimited text file.

# Output
- Displays the histogram inline.
- If `save_file=true`, writes a histogram plot PDF into `plot_dir` with a rule-based filename.
- Writes `outfile` with columns:

    Bin  Min  Max  OG  ML  [OG_oob  ML_oob]

# Returns
- `Nothing` (side effects: plot display, optional PDF save, and `.dat` output written).
"""
function plot_histogram_orig_vs_ml(
    trace_data::Dict{String, Vector{Vector{Float64}}},
    trace_idx::Int,
    nbins::Int,
    overall_name::String;
    subset::AbstractString="all",
    x_min::Union{Nothing, Real}=nothing,
    x_max::Union{Nothing, Real}=nothing,
    print_clipping::Bool=true,
    clipping_report_prefix::AbstractString="",
    outfile::AbstractString="histogram_bins.dat",
    include_out_of_range_counts_in_file::Bool=true,
    save_file::Bool = false,
    plot_dir::AbstractString = ""
)

    fig1, ax1 = PyPlot.subplots(figsize=(6, 3), dpi=500)

    # --------------------------------------------------
    # Select subset behavior
    # --------------------------------------------------
    subset_key = lowercase(subset)

    if subset_key == "all"
        org_show = vcat(
            trace_data["Y_tr"][trace_idx],
            trace_data["Y_bc"][trace_idx],
            trace_data["Y_ul"][trace_idx]
        )

        mle_show = vcat(
            trace_data["Y_tr"][trace_idx],
            trace_data["Y_bc"][trace_idx],
            trace_data["YP_ul"][trace_idx]
        )

    elseif subset_key == "tr"
        org_show = trace_data["Y_tr"][trace_idx]
        mle_show = trace_data["Y_tr"][trace_idx]

    elseif subset_key == "bc"
        org_show = trace_data["Y_bc"][trace_idx]
        mle_show = trace_data["YP_bc"][trace_idx]

    elseif subset_key == "ul"
        org_show = trace_data["Y_ul"][trace_idx]
        mle_show = trace_data["YP_ul"][trace_idx]

    else
        error("Invalid subset='$(subset)'. Must be one of: all, tr, bc, ul")
    end

    # --------------------------------------------------
    # Determine bounds
    # --------------------------------------------------
    data_min = min(minimum(org_show), minimum(mle_show))
    data_max = max(maximum(org_show), maximum(mle_show))

    final_min = (x_min === nothing) ? data_min : float(x_min)
    final_max = (x_max === nothing) ? data_max : float(x_max)

    if !(final_min < final_max)
        error("Invalid histogram range: require x_min < x_max.")
    end

    clipping_enabled = (x_min !== nothing) || (x_max !== nothing)

    og_oob = 0
    ml_oob = 0
    org_used = org_show
    mle_used = mle_show

    if clipping_enabled
        og_oob = count(x -> (x < final_min) || (x > final_max), org_show)
        ml_oob = count(x -> (x < final_min) || (x > final_max), mle_show)

        org_used = filter(x -> (x >= final_min) && (x <= final_max), org_show)
        mle_used = filter(x -> (x >= final_min) && (x <= final_max), mle_show)

        if print_clipping
            println("$(clipping_report_prefix)Histogram clipping is ON: [$(final_min), $(final_max)]")
            println("$(clipping_report_prefix)  OG discarded $(og_oob)")
            println("$(clipping_report_prefix)  ML discarded $(ml_oob)")
        end
    end

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------
    bin_edges = range(final_min, final_max; length=nbins+1)

    # Build legend labels depending on subset
    subset_tag = uppercase(subset_key)

    if subset_tag == "ALL"
        label_og = "OG"
        label_ml = "ML"
    else
        label_og = "OG-" * subset_tag
        label_ml = "ML-" * subset_tag
    end

    counts_og, _, _ = PyPlot.hist(
        org_used,
        bins=bin_edges,
        alpha=0.5,
        label=label_og,
        edgecolor="black",
        linewidth=1.0
    )

    counts_ml, _, _ = PyPlot.hist(
        mle_used,
        bins=bin_edges,
        alpha=0.5,
        label=label_ml,
        edgecolor="black",
        linewidth=1.0
    )

    PyPlot.legend()
    ax1.grid(true)
    fig1.tight_layout()
    display(fig1)

    # --------------------------------------------------
    # Save histogram plot (optional)
    # --------------------------------------------------
    if save_file
        # Use a heatmap-like basename convention
        if trace_idx == 1
            basename = "histogram_pbp_$(subset_key)_$(overall_name)"
        else
            basename = "histogram_trdinv$(trace_idx)_$(subset_key)_$(overall_name)"
        end

        # Decide output directory (empty => current directory)
        out_dir = isempty(plot_dir) ? "." : plot_dir
        resfile = joinpath(out_dir, "$basename.pdf")
        cropped = joinpath(out_dir, "$basename-crop.pdf")
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
        display("saved plot $(resfile)")
    end

    # --------------------------------------------------
    # Save file
    # --------------------------------------------------
    open(outfile, "w") do io
        if include_out_of_range_counts_in_file
            println(io, "Bin\tMin\tMax\tOG\tML\tOG_oob\tML_oob")
        else
            println(io, "Bin\tMin\tMax\tOG\tML")
        end

        for i in 1:nbins
            rmin = round(bin_edges[i], digits=5)
            rmax = round(bin_edges[i+1], digits=5)
            og = counts_og[i]
            ml = counts_ml[i]

            if include_out_of_range_counts_in_file
                println(io, "$i\t$rmin\t$rmax\t$og\t$ml\t$og_oob\t$ml_oob")
            else
                println(io, "$i\t$rmin\t$rmax\t$og\t$ml")
            end
        end
    end

    display("saved $(outfile)")
    return nothing
end

end