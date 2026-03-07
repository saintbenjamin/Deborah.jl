# ============================================================================
# src/RebekahMiriam/ReweightingPlotRebekahMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ReweightingPlotRebekahMiriam

import ..PyPlot
import ..PlotlyJS

import ..Sarah.JobLoggerTools

"""
    find_label_train_indices(
        label::String,
        train::String,
        labels::Vector{String},
        trains::Vector{String}
    ) -> Tuple{Int, Int}

Resolve string-based label and train identifiers to their corresponding integer indices.

This utility function locates the positions of `label` and `train` strings within the corresponding `labels` and `trains` vectors.  
It returns the index pair `(i, j)` such that `labels[i] == label` and `trains[j] == train`.  
If either is not found, an error is raised.

# Arguments
- `label`: Label identifier to search for (e.g., `"25"`).
- `train`: Train identifier to search for (e.g., `"20"`).
- `labels`: Vector of all label strings.
- `trains`: Vector of all train strings.

# Returns
- A tuple `(i, j)` where `i` is the index of `label` in `labels`, and `j` is the index of `train` in `trains`.

# Throws
- An error if either `label` or `train` is not found in their respective vectors.
"""
function find_label_train_indices(
    label::String, 
    train::String, 
    labels::Vector{String}, 
    trains::Vector{String}
)::Tuple{Int, Int}
    i = findfirst(==(label), labels)
    j = findfirst(==(train), trains)
    isnothing(i) && JobLoggerTools.error_benji("Label \"$label\" not found in labels vector")
    isnothing(j) && JobLoggerTools.error_benji("Train \"$train\" not found in trains vector")
    return (i, j)
end

"""
    plot_reweighting_pyplot(
        rw_data_ext::Dict{String, Dict{String, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}},
        new_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64, 2}},
        ykey::Symbol,
        ykey_tex::String,
        label::String,
        train::String,
        labels::Vector{String},
        trains::Vector{String},
        interpolate::String
    ) -> Nothing

Render full reweighting curves and interpolation points for a selected observable using [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).

This function is part of the [`Deborah.RebekahMiriam`](@ref) module and visualizes multi-ensemble reweighting results  
for a specific observable (`ykey`) at a given `(label, train)` configuration and interpolation criterion.  
It loads both curve data (`rw_data_ext`) and point estimates (`new_dict`) for three reweighting methods:
- `RWBS` (original data)
- `RWP1` (`P1`)
- `RWP2` (`P2`)

The function plots smooth reweighting curves with uncertainty bands as well as discrete point estimates,  
including final interpolated values with associated error bars.

# Arguments
- `rw_data_ext`: Nested dictionary of reweighting scan data organized as  
  `rw_data_ext[label][train][tag][observable] → vector`.
- `new_dict`: Dictionary containing interpolated results and error estimates indexed by  
  `(observable, kind, tag, interpolate)`.
- `ykey`: Symbol of the observable to be plotted (e.g., `:skew`, `:kurt`).
- `ykey_tex`: [``\\LaTeX``](https://www.latex-project.org/)-formatted string used as the ``y``-axis label.
- `label`: Labeled set key (e.g., `"25"`).
- `train`: Training set key (e.g., `"20"`).
- `labels`: Vector of all label keys.
- `trains`: Vector of all train keys.
- `interpolate`: String indicating the interpolation criterion (e.g., `"kurt"`).

# Side Effects
- Displays a [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure combining scan curves, error bands, prediction points, and interpolated estimates.
"""
function plot_reweighting_pyplot(
    rw_data_ext::Dict{String, Dict{String, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}},
    new_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64, 2}},
    ykey::Symbol,
    ykey_tex::String,
    label::String,
    train::String,
    labels::Vector{String},
    trains::Vector{String},
    interpolate::String,
    overall_name::String,
    figs_dir::String;
    save_file::Bool=false
)

    (i, j) = find_label_train_indices(label, train, labels, trains)

    # ORG = RWBS
    kappa_t_orig      = new_dict[(:kappa_t, :avg, :RWBS, interpolate)][i, j]
    err_kappa_t_orig  = new_dict[(:kappa_t, :err, :RWBS, interpolate)][i, j]
    cumulant_orig     = new_dict[(ykey,     :avg, :RWBS, interpolate)][i, j]
    err_cumulant_orig = new_dict[(ykey,     :err, :RWBS, interpolate)][i, j]

    # P1 = RWP1
    kappa_t_P1        = new_dict[(:kappa_t, :avg, :RWP1, interpolate)][i, j]
    err_kappa_t_P1    = new_dict[(:kappa_t, :err, :RWP1, interpolate)][i, j]
    cumulant_P1       = new_dict[(ykey,     :avg, :RWP1, interpolate)][i, j]
    err_cumulant_P1   = new_dict[(ykey,     :err, :RWP1, interpolate)][i, j]

    # P2 = RWP2
    kappa_t_P2        = new_dict[(:kappa_t, :avg, :RWP2, interpolate)][i, j]
    err_kappa_t_P2    = new_dict[(:kappa_t, :err, :RWP2, interpolate)][i, j]
    cumulant_P2       = new_dict[(ykey,     :avg, :RWP2, interpolate)][i, j]
    err_cumulant_P2   = new_dict[(ykey,     :err, :RWP2, interpolate)][i, j]

    fig, ax = PyPlot.subplots(figsize=(5.6, 5.0), dpi=500)

    xkey = :kappa
    rwtag_ORG, ytag_ORG = :RWBS, :Y_BS
    rwtag_P1,  ytag_P1  = :RWP1, :Y_P1
    rwtag_P2,  ytag_P2  = :RWP2, :Y_P2

    if haskey(rw_data_ext[label][train], rwtag_P1)
        kappa = rw_data_ext[label][train][rwtag_P1][xkey]
        yval  = rw_data_ext[label][train][rwtag_P1][ykey]
        yerr  = rw_data_ext[label][train][rwtag_P1][Symbol(ykey, :_err)]
        ax.plot(kappa, yval, color="orange", label="P1")
        ax.fill_between(kappa, yval .- yerr, yval .+ yerr,
                        color="orange", alpha=0.3, label="P1 Band")
    end

    if haskey(rw_data_ext[label][train], rwtag_ORG)
        kappa = rw_data_ext[label][train][rwtag_ORG][xkey]
        yval  = rw_data_ext[label][train][rwtag_ORG][ykey]
        yerr  = rw_data_ext[label][train][rwtag_ORG][Symbol(ykey, :_err)]
        ax.plot(kappa, yval, color="C0", label="Original")
        ax.fill_between(kappa, yval .- yerr, yval .+ yerr,
                        color="C0", alpha=0.3, label="Original Band")
    end

    if haskey(rw_data_ext[label][train], rwtag_P2)
        kappa = rw_data_ext[label][train][rwtag_P2][xkey]
        yval  = rw_data_ext[label][train][rwtag_P2][ykey]
        yerr  = rw_data_ext[label][train][rwtag_P2][Symbol(ykey, :_err)]
        ax.plot(kappa, yval, color="C3", label="P2")
        ax.fill_between(kappa, yval .- yerr, yval .+ yerr,
                        color="C3", alpha=0.3, label="P2 Band")
    end

    if haskey(rw_data_ext[label][train], ytag_P1)
        kappa = rw_data_ext[label][train][ytag_P1][xkey]
        yval  = rw_data_ext[label][train][ytag_P1][ykey]
        yerr  = rw_data_ext[label][train][ytag_P1][Symbol(ykey, :_err)]
        ax.errorbar(kappa, yval, yerr=yerr,
                    fmt="^", markersize=10, color="orange", capsize=5, label="RWP1 points")
    end

    if haskey(rw_data_ext[label][train], ytag_ORG)
        kappa = rw_data_ext[label][train][ytag_ORG][xkey]
        yval  = rw_data_ext[label][train][ytag_ORG][ykey]
        yerr  = rw_data_ext[label][train][ytag_ORG][Symbol(ykey, :_err)]
        ax.errorbar(kappa, yval, yerr=yerr,
                    fmt="s", markersize=10, color="C0", capsize=5, label="RWBS points")
    end

    if haskey(rw_data_ext[label][train], ytag_P2)
        kappa = rw_data_ext[label][train][ytag_P2][xkey]
        yval  = rw_data_ext[label][train][ytag_P2][ykey]
        yerr  = rw_data_ext[label][train][ytag_P2][Symbol(ykey, :_err)]
        ax.errorbar(kappa, yval, yerr=yerr,
                    fmt="o", markersize=10, color="C3", capsize=5, label="RWP2 points")
    end

    if isfinite(kappa_t_P1) && isfinite(cumulant_P1)
        ax.errorbar([kappa_t_P1], [cumulant_P1],
                    xerr=[err_kappa_t_P1], yerr=[err_cumulant_P1],
                    fmt="^", mfc="none", color="orange", capsize=5, label="P1")
    end

    if isfinite(kappa_t_orig) && isfinite(cumulant_orig)
        ax.errorbar([kappa_t_orig], [cumulant_orig],
                    xerr=[err_kappa_t_orig], yerr=[err_cumulant_orig],
                    fmt="s", mfc="none", color="C0", capsize=5, label="ORG")
    end

    if isfinite(kappa_t_P2) && isfinite(cumulant_P2)
        ax.errorbar([kappa_t_P2], [cumulant_P2],
                    xerr=[err_kappa_t_P2], yerr=[err_cumulant_P2],
                    fmt="o", mfc="none", color="C3", capsize=5, label="P2")
    end

    ax.set_xlabel("\$\\kappa\$")
    ax.set_ylabel(ykey_tex)
    ax.legend()
    fig.tight_layout()
    display(fig)

    basename = "plot_$(ykey)_$(interpolate)_$(overall_name)"
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
    plot_reweighting_plotly(
        rw_data_ext::Dict{String, Dict{String, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}},
        new_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64, 2}},
        ykey::Symbol,
        ykey_tex::String,
        label::String,
        train::String,
        labels::Vector{String},
        trains::Vector{String},
        interpolate::String
    ) -> PlotlyJS.Plot

Render an interactive reweighting plot using [`PlotlyJS`](https://plotly.com/javascript/) for a given observable and interpolation method.

This function is part of the [`Deborah.RebekahMiriam`](@ref) module and provides a dynamic [`PlotlyJS`](https://plotly.com/javascript/)-based visualization  
of reweighting scan curves, prediction points, and interpolated values.  
It supports three reweighting schemes:
- `RWBS`
- `RWP1`
- `RWP2`

Each method is shown as a smooth curve with a filled error band,  
along with discrete prediction points and a final interpolated marker with full error bars.

# Arguments
- `rw_data_ext`: Nested dictionary containing reweighting scan curves and predictions.
- `new_dict`: Dictionary of interpolated kappa and observable values with errors.
- `ykey`: Observable symbol (e.g., `:kurt`).
- `ykey_tex`: [``\\LaTeX``](https://www.latex-project.org/)-formatted ``y``-axis label string.
- `label`: Label set identifier (e.g., `"25"`).
- `train`: Training set identifier (e.g., `"20"`).
- `labels`: Vector of all label keys.
- `trains`: Vector of all train keys.
- `interpolate`: Interpolation criterion (e.g., `"kurt"`).

# Returns
- A [`PlotlyJS.Plot`](https://plotly.com/javascript/) object containing the full reweighting visualization.
"""
function plot_reweighting_plotly(
    rw_data_ext::Dict{String, Dict{String, Dict{Symbol, Dict{Symbol, Vector{Float64}}}}},
    new_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64, 2}},
    ykey::Symbol,
    ykey_tex::String,
    label::String,
    train::String,
    labels::Vector{String},
    trains::Vector{String},
    interpolate::String
)
    (i, j) = find_label_train_indices(label, train, labels, trains)

    # ORG = RWBS
    kappa_t_orig      = new_dict[(:kappa_t, :avg, :RWBS, interpolate)][i, j]
    err_kappa_t_orig  = new_dict[(:kappa_t, :err, :RWBS, interpolate)][i, j]
    cumulant_orig     = new_dict[(ykey,     :avg, :RWBS, interpolate)][i, j]
    err_cumulant_orig = new_dict[(ykey,     :err, :RWBS, interpolate)][i, j]

    # P1 = RWP1
    kappa_t_P1        = new_dict[(:kappa_t, :avg, :RWP1, interpolate)][i, j]
    err_kappa_t_P1    = new_dict[(:kappa_t, :err, :RWP1, interpolate)][i, j]
    cumulant_P1       = new_dict[(ykey,     :avg, :RWP1, interpolate)][i, j]
    err_cumulant_P1   = new_dict[(ykey,     :err, :RWP1, interpolate)][i, j]

    # P2 = RWP2
    kappa_t_P2        = new_dict[(:kappa_t, :avg, :RWP2, interpolate)][i, j]
    err_kappa_t_P2    = new_dict[(:kappa_t, :err, :RWP2, interpolate)][i, j]
    cumulant_P2       = new_dict[(ykey,     :avg, :RWP2, interpolate)][i, j]
    err_cumulant_P2   = new_dict[(ykey,     :err, :RWP2, interpolate)][i, j]

    xkey = :kappa
    traces = PlotlyJS.AbstractTrace[]

    function band_trace(kappa, yval, yerr, name, color)
        lower = yval .- yerr
        upper = yval .+ yerr
        return [
            PlotlyJS.scatter(x=kappa, y=lower, mode="lines", line=PlotlyJS.attr(color=color, width=0), showlegend=false),
            PlotlyJS.scatter(x=kappa, y=upper, mode="lines", fill="tonexty", fillcolor=color, line=PlotlyJS.attr(width=0), name=name)
        ]
    end

    # P1
    if haskey(rw_data_ext[label][train], :RWP1)
        d = rw_data_ext[label][train][:RWP1]
        append!(traces, [PlotlyJS.scatter(x=d[xkey], y=d[ykey], mode="lines", name="P1", line=PlotlyJS.attr(color="orange"))])
        append!(traces, band_trace(d[xkey], d[ykey], d[Symbol(ykey, :_err)], "P1 Band", "rgba(255,165,0,0.3)"))
    end

    # ORG
    if haskey(rw_data_ext[label][train], :RWBS)
        d = rw_data_ext[label][train][:RWBS]
        append!(traces, [PlotlyJS.scatter(x=d[xkey], y=d[ykey], mode="lines", name="Original", line=PlotlyJS.attr(color="blue"))])
        append!(traces, band_trace(d[xkey], d[ykey], d[Symbol(ykey, :_err)], "Original Band", "rgba(0,0,255,0.3)"))
    end

    # P2
    if haskey(rw_data_ext[label][train], :RWP2)
        d = rw_data_ext[label][train][:RWP2]
        append!(traces, [PlotlyJS.scatter(x=d[xkey], y=d[ykey], mode="lines", name="P2", line=PlotlyJS.attr(color="red"))])
        append!(traces, band_trace(d[xkey], d[ykey], d[Symbol(ykey, :_err)], "P2 Band", "rgba(255,0,0,0.3)"))
    end

    function point_trace(κ, y, κerr, yerr, shape, edgecolor, name)
        PlotlyJS.scatter(
            x=[κ], y=[y],
            error_x=PlotlyJS.attr(array=[κerr], visible=true, color=edgecolor),
            error_y=PlotlyJS.attr(array=[yerr], visible=true, color=edgecolor),
            mode="markers",
            name=name,
            marker=PlotlyJS.attr(
                symbol=shape,
                size=10,
                color="rgba(0,0,0,0)",
                line=PlotlyJS.attr(width=2, color=edgecolor)
            )
        )
    end

    if isfinite(kappa_t_P1) && isfinite(cumulant_P1)
        push!(traces, point_trace(kappa_t_P1, cumulant_P1, err_kappa_t_P1, err_cumulant_P1, "triangle-up", "orange", "P1"))
    end
    if isfinite(kappa_t_orig) && isfinite(cumulant_orig)
        push!(traces, point_trace(kappa_t_orig, cumulant_orig, err_kappa_t_orig, err_cumulant_orig, "square", "blue", "ORG"))
    end
    if isfinite(kappa_t_P2) && isfinite(cumulant_P2)
        push!(traces, point_trace(kappa_t_P2, cumulant_P2, err_kappa_t_P2, err_cumulant_P2, "circle", "red", "P2"))
    end

    ytag_ORG = :Y_BS
    ytag_P1  = :Y_P1
    ytag_P2  = :Y_P2

    function errorbar_points(kappa_vec, y_vec, yerr_vec, shape, color, label)
        PlotlyJS.scatter(
            x=kappa_vec,
            y=y_vec,
            error_y=PlotlyJS.attr(array=yerr_vec, visible=true),
            mode="markers",
            name=label,
            marker=PlotlyJS.attr(symbol=shape, color=color, size=10)
        )
    end

    if haskey(rw_data_ext[label][train], ytag_P1)
        d = rw_data_ext[label][train][ytag_P1]
        push!(traces, errorbar_points(d[xkey], d[ykey], d[Symbol(ykey, :_err)], "triangle-up", "orange", "RWP1 points"))
    end

    if haskey(rw_data_ext[label][train], ytag_ORG)
        d = rw_data_ext[label][train][ytag_ORG]
        push!(traces, errorbar_points(d[xkey], d[ykey], d[Symbol(ykey, :_err)], "square", "blue", "RWBS points"))
    end

    if haskey(rw_data_ext[label][train], ytag_P2)
        d = rw_data_ext[label][train][ytag_P2]
        push!(traces, errorbar_points(d[xkey], d[ykey], d[Symbol(ykey, :_err)], "circle", "red", "RWP2 points"))
    end

    layout = PlotlyJS.Layout(
        title="\$\\kappa_t\$ determined with $(interpolate)",
        xaxis_title="\$\\kappa\$",
        yaxis_title = ykey_tex,
        width=1200,
        height=450,
        include_mathjax = "cdn"
    )

    return PlotlyJS.Plot(traces, layout)
end

end  # ReweightingPlotRebekahMiriam