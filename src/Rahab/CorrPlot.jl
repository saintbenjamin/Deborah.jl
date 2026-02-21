# ============================================================================
# src/Rahab/CorrPlot.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module CorrPlot

import ..Sarah.JobLoggerTools
import DelimitedFiles
import Statistics
import PyPlot

"""
    build_observables(
        keys::AbstractVector{<:AbstractString},
        key_doc::AbstractVector{<:AbstractString},
        raw_path::AbstractString;
        real_col::Integer=1,
        file_ext::AbstractString=".dat"
    ) -> Matrix{Float64}

Read per-key data files from `raw_path` named as `<key><file_ext>` and extract column `real_col`
(default: `1`). Stack them so that each row corresponds to one key, preserving the order of `keys`.
Returns only the matrix `mat::Matrix{Float64}`.

# Arguments
- `keys`: basenames like `["pbp", "trdinv2",...]`.
- `key_doc`: doc strings (must be same length/order as `keys`).
- `raw_path`: directory containing the files.
- `real_col`: ``1``-based column index to read (default: `1`).
- `file_ext`: file extension including dot (default: `".dat"`).

# Throws
- `AssertionError` if `length(keys) != length(key_doc)` or if sample lengths differ.
- `ArgumentError` if a file is missing or `real_col` is out of bounds.
"""
function build_observables(
    keys::AbstractVector{<:AbstractString},
    key_doc::AbstractVector{<:AbstractString},
    raw_path::AbstractString;
    real_col::Integer=1,
    file_ext::AbstractString=".dat"
)::Matrix{Float64}
    JobLoggerTools.assert_benji(
        length(keys) == length(key_doc),
        "keys and key_doc must have the same length."
    )

    series = Vector{Vector{Float64}}(undef, length(keys))
    nsamples_ref::Union{Nothing,Int} = nothing

    @inbounds for i in eachindex(keys)
        key = keys[i]
        filepath = joinpath(raw_path, string(key, file_ext))
        if !isfile(filepath)
            throw(ArgumentError("Data file not found: $filepath"))
        end

        data = DelimitedFiles.readdlm(filepath, Float64)
        ncols = size(data, 2)
        if real_col < 1 || real_col > ncols
            throw(ArgumentError("File $filepath has $ncols columns, but real_col=$real_col requested."))
        end

        col = Vector{Float64}(data[:, real_col])
        series[i] = col

        if nsamples_ref === nothing
            nsamples_ref = length(col)
        elseif length(col) != nsamples_ref
            JobLoggerTools.assert_benji(
                false,
                "All series must have the same number of samples. Mismatch at key='$key': got $(length(col)) vs $nsamples_ref."
            )
        end
    end

    # Row = key, Column = sample
    return vcat((s' for s in series)...)
end

"""
    corrmat_plot(
        mat::AbstractMatrix{<:Real},
        key_doc::Vector{String},
        ensemble::String,
        figs_dir::String;
        save_file::Bool=false
    ) -> Nothing

Render and (optionally) save a heatmap of the absolute correlation matrix
computed across the rows of `mat` (`Statistics.cor(mat, dims=2)`).
Each row is treated as a variable and each column as a sample.

# Arguments
- `mat`: Real-valued data matrix of shape `(n_vars` ``\\times`` `n_samples)`; rows are variables, columns are samples.
- `key_doc`: Tick labels for variables (length must equal `size(mat, 1)`).
  Labels may include [`mathtext`](https://matplotlib.org/stable/users/explain/text/mathtext.html)/[``\\LaTeX``](https://www.latex-project.org/)-like strings (e.g., `"\$\\mathrm{Tr} M^{-1}\$"`).
- `ensemble`: Identifier used to construct the output filename (e.g., `"L8T4b1.60k13570"` → `corr_L8T4b1.60k13570.pdf`).
- `figs_dir`: Directory where the PDF is written if `save_file=true`.

# Keyword Arguments
- `save_file::Bool=false`: When `true`, saves a PDF to `figs_dir`. If [`pdfcrop`](https://ctan.org/pkg/pdfcrop) is available
  on the system `PATH`, a cropped version is produced and overwrites the original.

# Behavior
- Computes `cormat = abs.(Statistics.cor(mat, dims=2))`.
- Draws a heatmap with `vmin=0.0`, `vmax=1.0`, and colormap `"autumn_r"`.
- Annotates each cell with its value (rounded to 3 digits) and chooses black/white text
  based on a fixed threshold (`0.4`) for contrast.
- Sets both axes ticks to `0:(n_vars-1)` with labels from `key_doc`, rotates ``x``-tick labels by 60°,
  sets square cells via `ax.set_aspect(1)`, calls `tight_layout()`, and displays the figure.
- If `save_file=true`, writes `<figs_dir>/corr_\$(ensemble).pdf` and, when available,
  runs [`pdfcrop`](https://ctan.org/pkg/pdfcrop) and replaces the original with the cropped output.

# Notes
- If labels include [``\\LaTeX``](https://www.latex-project.org/) commands unsupported by [`Matplotlib`'s mathtext](https://matplotlib.org/stable/users/explain/text/mathtext.html) (e.g., `\\textrm`),
  prefer [mathtext](https://matplotlib.org/stable/users/explain/text/mathtext.html)-friendly forms (e.g., `\\mathrm`) or enable `usetex=true` globally if you need full [``\\LaTeX``](https://www.latex-project.org/).

# Returns
- `Nothing`.
"""
function corrmat_plot(
    mat::AbstractMatrix{<:Real},
    key_doc::Vector{String},
    ensemble::String,
    figs_dir::String;
    save_file::Bool=false
)
    # debug

    # compute correlation matrix
    # cormat = abs.(Statistics.cor(mat, dims=2))
    cormat = Statistics.cor(mat, dims=2)
    # visualize
    # Create a heatmap from the matrix
    fig, ax = PyPlot.subplots()  # Create a figure and a set of subplots
    # im = ax.imshow(cormat,vmin=0.0,vmax=1.,cmap="autumn_r")  # 'hot' is just one of the many color maps
    im = ax.imshow(cormat,vmin=-1.,vmax=1.,cmap="RdBu_r")  # 'hot' is just one of the many color maps
    # Display the color bar
    PyPlot.colorbar(im)

    # Loop over data dimensions and create text annotations, displaying the value in each cell
    for (i, row) in enumerate(eachrow(cormat))
        for (j, value) in enumerate(row)
            # c = value <= 0.8 ? "black" : "w"
            # c = "black"
            # c = value <= 0.4 ? "black" : "w"
            c = abs(value) <= 0.5 ? "black" : "w"
            ax.text(j-1, i-1, round(value, digits=3), ha="center", va="center", color=c,fontsize="x-small")
        end
    end

    # Axis labels can be set if needed
    # ax.set_xlabel("X-axis label")
    # ax.set_ylabel("Y-axis label")

    ax.set_xticks(Array(range(1,length(mat[:,1]))).-1,labels=key_doc,rotation=60)
    ax.set_yticks(Array(range(1,length(mat[:,1]))).-1,labels=key_doc)
    # Optionally set the aspect of the plot to 'auto' so that cells are not forced to be square
    ax.set_aspect(1)
    # ax.set_title("\\texttt{$ensemble}")

    fig.tight_layout()
    display(fig)

    basename = "corr_$(ensemble)"
    resfile  = joinpath(figs_dir, "$basename.pdf")
    cropped  = joinpath(figs_dir, "$basename-crop.pdf")
    if save_file
        isdir(figs_dir) || mkpath(figs_dir)
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

end

end # module CorrPlot