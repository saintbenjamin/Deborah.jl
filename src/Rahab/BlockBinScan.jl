# ============================================================================
# src/Rahab/BlockBinScan.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BlockBinScan

import ..PyPlot
import ..ProgressMeter
import ..Random
import ..Statistics

import ..Sarah.Bootstrap
import ..Sarah.Jackknife
import ..Sarah.JobLoggerTools

"""
    block_bin_scan(
        X_info_ORG::AbstractVector{<:Real},
        bin_sizes::Vector{<:Integer},
        tot_bin::Int;
        resample::Symbol = :bootstrap,
        N_bs::Int = 1000,
        rng::Random.AbstractRNG = Random.default_rng(),
        method::String = "nonoverlapping"
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Scan over the provided sizes (`bin_sizes`) and compute ``\\mu \\pm \\sigma`` using
either **bootstrap** or **jackknife** resampling. Plots the curve with a
shaded ``\\pm \\sigma`` band.

# Arguments
- `X_info_ORG`: Vector of observable values (`length = total samples`).
- `bin_sizes`: Sizes to evaluate (e.g., `collect(1:max_bin)`).
- `tot_bin`: Printed only to show its divisors for reference.

# Keywords
- `resample`: `:bootstrap` or `:jackknife` (default `:bootstrap`).
- `N_bs`: Number of bootstrap resamples (used only when `resample == :bootstrap`).
- `rng`: RNG for bootstrap (ignored by jackknife).
- `method`: Bootstrap method (e.g., `"nonoverlapping"`, `"circular"`). Ignored by jackknife.

# Behavior
- Prints all divisors of `tot_bin` for reference.
- For each `b` ``\\in`` `bin_sizes`, computes `(mean, error)` via the selected resampling.
- Plots mean curve with a shaded ``\\pm \\sigma`` band.
  - ``x``-axis label: "block size" for bootstrap, "bin size" for jackknife.

# Returns
- `(means, errs) :: (Vector{Float64}, Vector{Float64})` aligned with `bin_sizes`.

# Notes
- Requires [`ProgressMeter.jl`](https://github.com/timholy/ProgressMeter.jl) and [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl).
"""
function block_bin_scan(
    ensemble::String,
    X_info_ORG::AbstractVector{<:Real},
    bin_sizes::Vector{<:Integer},
    tot_bin::Int;
    resample::Symbol = :bootstrap,
    N_bs::Int = 1000,
    rng::Random.AbstractRNG = Random.default_rng(),
    method::String = "nonoverlapping"
)::Tuple{Vector{Float64}, Vector{Float64}}

    # Log divisors for reference
    divisors = [d for d in 1:tot_bin if tot_bin % d == 0]
    JobLoggerTools.println_benji("Divisors of $tot_bin: $divisors")

    means = zeros(Float64, length(bin_sizes))
    errs  = zeros(Float64, length(bin_sizes))

    # Normalize and validate resample mode
    mode = resample
    JobLoggerTools.assert_benji(
        mode in (:bootstrap, :jackknife),
        "resample must be :bootstrap or :jackknife"
    )

    # Progress text
    ptxt = mode === :bootstrap ? "bootstrap sweep..." : "jackknife sweep..."

    ProgressMeter.@showprogress 1 ptxt for (i, b) in enumerate(bin_sizes)
        if mode === :bootstrap
            m_tmp, e_tmp = Bootstrap.bootstrap_average_error_from_raw(
                X_info_ORG, N_bs, b, rng; method=method
            )
        else
            m_tmp, e_tmp = Jackknife.jackknife_average_error_from_raw(
                X_info_ORG, b
            )
        end
        means[i] = m_tmp
        errs[i]  = e_tmp
    end

    # Plot
    fig, ax = PyPlot.subplots()
    ax.plot(bin_sizes, means, label = (mode === :bootstrap ? "Bootstrap mean" : "Jackknife mean"),
            color="blue", linewidth=1.5)
    ax.fill_between(bin_sizes, means .- errs, means .+ errs,
                    color="skyblue", alpha=0.25, label="± error")
    ax.set_xlabel(mode === :bootstrap ? "block size" : "bin size")
    ax.set_ylabel("mean ± error")
    ax.set_title(mode === :bootstrap ? "Bootstrap vs. block size ($ensemble)" : "Jackknife vs. bin size ($ensemble)")
    ax.grid(true)
    ax.legend()
    fig.tight_layout()
    display(fig)

    return means, errs
end

"""
    bootstrap_block_scan(
        ensemble::String,
        X::AbstractVector{<:Real},
        bins::Vector{<:Integer},
        N_bs::Int,
        tot_bin::Int,
        rng::Random.AbstractRNG;
        method::String = "nonoverlapping"
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Convenience wrapper around [`block_bin_scan`](@ref) that runs a bootstrap sweep over
`bins` and plots the mean curve with a shaded ``\\pm \\sigma`` band.

# Arguments
- `ensemble`: Label used in the plot title (e.g., ensemble ID).
- `X`: Vector of observable values (`length = total samples`).
- `bins`: Block sizes to evaluate (e.g., `collect(1:max_bin)`).
- `N_bs`: Number of bootstrap resamples.
- `tot_bin`: Printed only to show its divisors for reference.
- `rng`: RNG used by the bootstrap resampler.
- `method`: Bootstrap method (e.g., `"nonoverlapping"`, `"circular"`).

# Behavior
- Prints all divisors of `tot_bin` for reference.
- For each `b` ``\\in`` `bins`, computes `(mean, error)` using bootstrap.
- Produces a plot of mean vs. block size with a shaded ``\\pm \\sigma`` band.

# Returns
- `(means, errs)` aligned with `bins`.

# Notes
- Equivalent to calling:
  [`block_bin_scan(ensemble, X, bins, tot_bin; resample=:bootstrap, N_bs=N_bs, rng=rng, method=method)`](@ref block_bin_scan).
"""
bootstrap_block_scan(ensemble, X, bins, N_bs, tot_bin, rng; method="nonoverlapping") =
    block_bin_scan(ensemble, X, bins, tot_bin; resample=:bootstrap, N_bs=N_bs, rng=rng, method=method)

"""
    jackknife_bin_scan(
        ensemble::String,
        X::AbstractVector{<:Real},
        bins::Vector{<:Integer},
        tot_bin::Int
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Convenience wrapper around [`block_bin_scan`](@ref) that runs a jackknife sweep over
`bins` and plots the mean curve with a shaded ``\\pm \\sigma`` band.

# Arguments
- `ensemble`: Label used in the plot title (e.g., ensemble ID).
- `X`: Vector of observable values (`length = total samples`).
- `bins`: Bin sizes to evaluate (e.g., `collect(1:max_bin)`).
- `tot_bin`: Printed only to show its divisors for reference.

# Behavior
- Prints all divisors of `tot_bin` for reference.
- For each `b` ``\\in`` `bins`, computes `(mean, error)` using jackknife.
- Produces a plot of mean vs. bin size with a shaded ``\\pm \\sigma`` band.

# Returns
- `(means, errs)` aligned with `bins`.

# Notes
- Equivalent to calling:
  [`block_bin_scan(ensemble, X, bins, tot_bin; resample=:jackknife)`](@ref block_bin_scan).
"""
jackknife_bin_scan(ensemble, X, bins, tot_bin) =
    block_bin_scan(ensemble, X, bins, tot_bin; resample=:jackknife)

"""
    block_bin_scan_blocked_average(
        bin_sizes::Vector{<:Integer},
        means::Vector{Float64},
        errs::Vector{Float64},
        win::Int;
        resample::Symbol = :bootstrap,
        include_tail::Bool=false
    ) -> Nothing

Plot horizontal step segments of blocked averages (``\\mu \\pm \\sigma``) over
non-overlapping windows of length `win`, alongside the original curve.

- When `resample == :bootstrap`: labels/``x``-axis refer to block size.
- When `resample == :jackknife`: labels/``x``-axis refer to bin size.

# Arguments
- `bin_sizes`: Sizes corresponding to each point (e.g., `collect(1:max_bin)`).
- `means`: Mean per size (same length as `bin_sizes`).
- `errs`:  Error per size (same length as `bin_sizes`).
- `win`: Window length for non-overlapping grouping (e.g., `win=40` → `[1..40], [41..80], ...`).

# Keywords
- `resample`: `:bootstrap` or `:jackknife` (default `:bootstrap`).
- `include_tail`: Include the last partial window if present (default `false`).

# Behavior
- Splits indices into windows of length `win` (optionally includes the tail).
- For each window, computes
  `blocked_mean = mean(means[window])`, `blocked_err = mean(errs[window])`.
- Plots:
  - original ``\\mu \\pm \\sigma`` band as reference,
  - blocked mean as horizontal steps,
  - blocked ``\\pm`` error as shaded rectangles.

# Returns
- `Nothing` (displays the figure inline).
"""
function block_bin_scan_blocked_average(
    ensemble::String,
    bin_sizes::Vector{<:Integer},
    means::Vector{Float64},
    errs::Vector{Float64},
    win::Int;
    resample::Symbol = :bootstrap,
    include_tail::Bool=false
) :: Nothing
    JobLoggerTools.assert_benji(
        length(bin_sizes) == length(means) && length(means) == length(errs),
        "Input vectors must have equal length"
    )
    JobLoggerTools.assert_benji(
        win >= 1, 
        "win must be ≥ 1"
    )
    JobLoggerTools.assert_benji(
        resample in (:bootstrap, :jackknife),
        "resample must be :bootstrap or :jackknife"
    )

    N  = length(bin_sizes)
    nb = div(N, win)  # number of full windows
    ranges = [((k-1)*win + 1):(k*win) for k in 1:nb]
    if include_tail && N > nb*win
        push!(ranges, (nb*win + 1):N)
        nb += 1
    end

    # Block representatives + blocked stats
    blk_left  = [bin_sizes[first(r)] for r in ranges]
    blk_right = [bin_sizes[last(r)]  for r in ranges]
    blk_mu    = [Statistics.mean(means[r]) for r in ranges]
    blk_err   = [Statistics.mean(errs[r])  for r in ranges]

    # Labels switch by resample mode
    is_boot  = (resample === :bootstrap)
    curve_lbl = is_boot ? "bootstrap mean" : "jackknife mean"
    xlabel_  = is_boot ? "block size"     : "bin size"
    title_   = is_boot ? "Bootstrap vs block size :: blocked averages (window = $win, $ensemble)" : "Jackknife vs bin size :: blocked averages (window = $win, $ensemble)"

    # Plot
    fig, ax = PyPlot.subplots()

    # Original curve
    ax.plot(bin_sizes, means; label=curve_lbl, linewidth=1.0)
    ax.fill_between(bin_sizes, means .- errs, means .+ errs; alpha=0.20, label="± error")

    # Blocked averages as horizontal steps
    for k in eachindex(blk_mu)
        x1, x2 = blk_left[k], blk_right[k]
        μ, e = blk_mu[k], blk_err[k]

        ax.hlines(μ, x1, x2; linewidth=2.0, label=(k == 1 ? "blocked mean" : nothing))
        ax.fill_between([x1, x2], [μ - e, μ - e], [μ + e, μ + e]; alpha=0.25,
                        label=(k == 1 ? "blocked ± error" : nothing))
    end

    ax.set_xlabel(xlabel_)
    ax.set_ylabel("mean ± error")
    ax.set_title(title_)
    ax.grid(true)
    ax.legend()
    fig.tight_layout()
    display(fig)

    return nothing
end

"""
    bootstrap_block_scan_blocked_average(
        ensemble::String,
        bin_sizes::Vector{<:Integer},
        means::Vector{Float64},
        errs::Vector{Float64},
        win::Int;
        include_tail::Bool = false
    ) -> Nothing

Convenience wrapper around [`block_bin_scan_blocked_average`](@ref) that plots bootstrap
blocked averages over windows of length `win`.

# Arguments
- `ensemble`: Label used in the plot title (e.g., ensemble ID).
- `bin_sizes`: Block sizes (same length as `means` and `errs`).
- `means`: Mean per block size.
- `errs`: Error per block size.
- `win`: Window length for grouping into non-overlapping segments.
- `include_tail`: If `true`, includes the final partial window (default `false`).

# Behavior
- Groups indices into windows of length `win` (plus tail if enabled).
- For each group, computes average mean and error.
- Plots:
  - original bootstrap ``\\mu \\pm \\sigma`` band,
  - horizontal step segments of blocked means,
  - shaded rectangles for blocked ``\\pm`` error.

# Returns
- `Nothing` (displays the figure inline).

# Notes
- Equivalent to:
  [`block_bin_scan_blocked_average(ensemble, bin_sizes, means, errs, win; resample=:bootstrap, include_tail=include_tail)`](@ref block_bin_scan_blocked_average).
"""
bootstrap_block_scan_blocked_average(ensemble, bin_sizes, means, errs, win; include_tail=false) =
    block_bin_scan_blocked_average(ensemble, bin_sizes, means, errs, win; resample=:bootstrap, include_tail=include_tail)

"""
    jackknife_bin_scan_blocked_average(
        ensemble::String,
        bin_sizes::Vector{<:Integer},
        means::Vector{Float64},
        errs::Vector{Float64},
        win::Int;
        include_tail::Bool = false
    ) -> Nothing

Convenience wrapper around [`block_bin_scan_blocked_average`](@ref) that plots jackknife
blocked averages over windows of length `win`.

# Arguments
- `ensemble`: Label used in the plot title (e.g., ensemble ID).
- `bin_sizes`: Bin sizes (same length as `means` and `errs`).
- `means`: Mean per bin size.
- `errs`: Error per bin size.
- `win`: Window length for grouping into non-overlapping segments.
- `include_tail`: If `true`, includes the final partial window (default `false`).

# Behavior
- Groups indices into windows of length `win` (plus tail if enabled).
- For each group, computes average mean and error.
- Plots:
  - original jackknife ``\\mu \\pm \\sigma`` band,
  - horizontal step segments of blocked means,
  - shaded rectangles for blocked ``\\pm`` error.

# Returns
- `Nothing` (displays the figure inline).

# Notes
- Equivalent to:
  [`block_bin_scan_blocked_average(ensemble, bin_sizes, means, errs, win; resample=:jackknife, include_tail=include_tail)`](@ref block_bin_scan_blocked_average).
"""
jackknife_bin_scan_blocked_average(ensemble, bin_sizes, means, errs, win; include_tail=false) =
    block_bin_scan_blocked_average(ensemble, bin_sizes, means, errs, win; resample=:jackknife, include_tail=include_tail)

"""
    nsr_block_scan(
        bin_sizes::Vector{Int},
        means::Vector{Float64},
        errs::Vector{Float64};
        logscale::Bool=false
    ) -> Nothing

Plot the noise-to-signal ratio (NSR) defined as ``\\sigma / |\\mu|``
as a function of block size.

# Arguments
- `bin_sizes::Vector{Int}`: Block-size values corresponding to each point.
- `means::Vector{Float64}`: Bootstrap means for each block size.
- `errs::Vector{Float64}`: Bootstrap errors for each block size.
- `logscale::Bool`: If `true`, use a logarithmic ``y``-axis (default = `false`).

# Behavior
- Computes NSR values safely, avoiding division by zero
  (returns `NaN` when `error == 0`).
- Produces a PyPlot figure with NSR vs. block size.
- Optionally sets ``y``-axis to log scale.

# Returns
- `Nothing`. Displays the figure inline.
"""
function nsr_block_scan(
    ensemble::String,
    bin_sizes::Vector{Int},
    means::Vector{Float64},
    errs::Vector{Float64};
    logscale::Bool=false
) :: Nothing
    # Compute NSR safely (avoid division by zero)
    nsr = map((m, e) -> (e > 0 ? e/abs(m) : NaN), means, errs)

    # Plot NSR vs. block size
    fig, ax = PyPlot.subplots()
    ax.plot(bin_sizes, nsr, linewidth=1.5, label=raw"NSR = error / $|\mathrm{mean}|$")
    ax.set_xlabel("block size")
    ax.set_ylabel("NSR")
    ax.set_title("Noise-to-signal ratio vs. block size ($ensemble)")
    ax.grid(true)
    ax.legend()
    if logscale
        ax.set_yscale("log")
    end
    fig.tight_layout()
    display(fig)

    return nothing
end

"""
    nsr_block_scan_with_discarded_zero(
        X_info_ORG::AbstractVector{<:Real},
        tot_bin::Int,
        means::Vector{Float64},
        errs::Vector{Float64};
        logscale::Bool=false
    ) -> Tuple{Vector{Int}, Vector{Float64}, Vector{Int}}

Plot the noise-to-signal ratio (NSR) defined as ``\\sigma / |\\mu|``
versus block size and overlay, in red markers, the block sizes where
no samples are discarded (i.e., `N_total % b == 0`).

# Arguments
- `X_info_ORG`: Raw observable values; only its length is used (`N_total`).
- `tot_bin`: Upper bound on block size; we sweep `b = 1:min(tot_bin, N_total)`.
- `means`: Bootstrap means for each block size `b` (length must equal `min(tot_bin, N_total)`).
- `errs`:  Bootstrap errors for each block size `b` (same length as `means`).
- `logscale`: If `true`, use a logarithmic ``y``-axis (default: `false`).

# Behavior
- Computes `bin_sizes = 1:max_bin` with `max_bin = min(tot_bin, N_total)`.
- Computes `NSR[b] = error[b] / |mean[b]|` safely (`NaN` when `error[b] == 0`).
- Computes `discarded[b] = N_total % b`; overlays red markers at bins with `discarded[b] == 0`.
- Produces a [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure with the NSR curve and the overlay.

# Returns
- `(good_xs, good_nsrs, discarded)`:
  - `good_xs :: Vector{Int}` — bin sizes where `discarded == 0`,
  - `good_nsrs :: Vector{Float64}` — NSR values at those bins,
  - `discarded :: Vector{Int}` — per-bin discarded counts (`N_total % b`) for all `b`.
"""
function nsr_block_scan_with_discarded_zero(
    ensemble::String,
    X_info_ORG::AbstractVector{<:Real},
    tot_bin::Int,
    means::Vector{Float64},
    errs::Vector{Float64};
    logscale::Bool=false
) :: Tuple{Vector{Int}, Vector{Float64}, Vector{Int}}

    # Build bin sizes and basic checks
    N       = length(X_info_ORG)
    max_bin = min(tot_bin, N)
    bin_sizes = collect(1:max_bin)

    JobLoggerTools.assert_benji(
        length(means) == max_bin && length(errs) == max_bin,
        "means/errs must have length == min(tot_bin, length(X_info_ORG))"
    )

    # NSR(b) = error/|mean|
    nsr = map((m, e) -> (e > 0 ? e/abs(m) : NaN), means, errs)

    # Discarded count per block size: N % b
    discarded = [N - b * div(N, b) for b in bin_sizes]
    good_bins = findall(==(0), discarded)  # indices of bins with discarded == 0

    # Plot
    fig, ax = PyPlot.subplots()
    ax.plot(bin_sizes, nsr; linewidth=1.5, label=raw"NSR = error / $|\mathrm{mean}|$")
    if !isempty(good_bins)
        ax.plot(bin_sizes[good_bins], nsr[good_bins];
                marker="o", markersize=6, mfc="none", mec="red",
                linestyle="None", label="discarded = 0")
    end
    ax.set_xlabel("block size")
    ax.set_ylabel("NSR")
    ax.set_title("Noise-to-signal ratio vs. block size ($ensemble)")
    ax.grid(true)
    ax.legend()
    if logscale
        ax.set_yscale("log")
    end
    fig.tight_layout()
    display(fig)

    # Return overlay coordinates + full discarded vector
    good_xs    = bin_sizes[good_bins]
    good_nsrs  = nsr[good_bins]
    return good_xs, good_nsrs, discarded
end

"""
    nsr_block_scan_blocked_average(
        bin_sizes::Vector{Int},
        means::Vector{Float64},
        errs::Vector{Float64},
        win::Int;
        include_tail::Bool=false,
        logscale::Bool=false
    ) -> Nothing

Plot a blocked (non-overlapping) step approximation of the
noise-to-signal ratio (NSR), defined as ``\\sigma / |\\mu|``, together
with the raw NSR curve.

# Arguments
- `bin_sizes::Vector{Int}`: Block-size values corresponding to each point.
- `means::Vector{Float64}`: Bootstrap means for each block size.
- `errs::Vector{Float64}`: Bootstrap errors for each block size.
- `win::Int`: Window size for non-overlapping grouping (e.g., `win=10` makes `[1..10], [11..20], ...`).
- `include_tail::Bool`: If `true`, include the final partial window if present (default `false`).
- `logscale::Bool`: If `true`, use a logarithmic ``y``-axis (default `false`).

# Behavior
- Computes the raw NSR per point as ``\\sigma / |\\mu|`` (returns `NaN` when `error == 0`;
  note: if ``|\\mu| = 0`` the ratio is `Inf` by definition).
- Groups points into non-overlapping windows of length `win` and computes per-window
  averages of `mean` and `error`, then forms the blocked NSR as
  ```math
  \\text{blocked NSR} = \\frac{\\text{avg error}}{|\\text{avg mean}|} \\,.
  ```
- Produces a [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure showing the raw NSR curve and the blocked step NSR.

# Returns
- `Nothing`. Displays the figure inline.
"""
function nsr_block_scan_blocked_average(
    ensemble::String,
    bin_sizes::Vector{Int},
    means::Vector{Float64},
    errs::Vector{Float64},
    win::Int;
    include_tail::Bool=false,
    logscale::Bool=false
) :: Nothing

    N  = length(bin_sizes)
    JobLoggerTools.assert_benji(
        N == length(means) && length(means) == length(errs),
        "bin_sizes, means, errs must have equal length"
    )
    JobLoggerTools.assert_benji(
        win >= 1, 
        "win must be ≥ 1"
    )

    # Raw NSR per point (NaN if error==0; Inf if |mean|==0)
    nsr_full = map((m,e) -> (e > 0 ? e/abs(m) : NaN), means, errs)

    # Build non-overlapping windows
    nb = div(N, win)                            # number of full windows
    blk_ranges = [((k-1)*win + 1):(k*win) for k in 1:nb]

    # Block representatives and blocked stats
    blk_left  = [bin_sizes[first(r)] for r in blk_ranges]
    blk_right = [bin_sizes[last(r)]  for r in blk_ranges]
    blk_mu    = [Statistics.mean(means[r]) for r in blk_ranges]
    blk_err   = [Statistics.mean(errs[r])  for r in blk_ranges]
    blk_nsr   = map((m,e) -> (e > 0 ? e/abs(m) : NaN), blk_mu, blk_err)

    # Optionally include tail window
    if include_tail && N > nb*win
        tail_range = (nb*win + 1):N
        push!(blk_left,  bin_sizes[first(tail_range)])
        push!(blk_right, bin_sizes[last(tail_range)])
        μ_tail  = Statistics.mean(means[tail_range])
        err_tail = Statistics.mean(errs[tail_range])
        push!(blk_mu,  μ_tail)
        push!(blk_err, err_tail)
        push!(blk_nsr, (err_tail > 0 ? err_tail/abs(μ_tail) : NaN))
    end

    # Plot: raw NSR + blocked step NSR
    fig, ax = PyPlot.subplots()

    ax.plot(bin_sizes, nsr_full, linewidth=1.0, alpha=0.6, label="NSR (raw)")
    for k in eachindex(blk_nsr)
        x1, x2 = blk_left[k], blk_right[k]
        ax.hlines(blk_nsr[k], x1, x2, linewidth=2.0,
                  label=(k==1 ? "NSR (blocked)" : nothing))
    end

    ax.set_xlabel("block size")
    ax.set_ylabel("NSR")
    ax.set_title("Blocked noise-to-signal ratio (window = $win, $ensemble)")
    ax.grid(true)
    ax.legend()
    if logscale
        ax.set_yscale("log")
    end
    fig.tight_layout()
    display(fig)

    return nothing
end

"""
    nsr_block_scan_blocked_relchange(
        X_info_ORG::AbstractVector{<:Real},
        tot_bin::Int,
        means::Vector{Float64},
        errs::Vector{Float64},
        win::Int;
        logscale::Bool=false
    ) -> Tuple{Vector{Int}, Vector{Float64}, Vector{Int}}

Overlay the normalized change of blocked NSR (noise-to-signal ratio),
together with the raw NSR-based residual-like sequence, and return the
overlay coordinates and the per-bin `discarded` counts.

NSR is defined as ``\\sigma / |\\mu|``. Blocked NSR for a window is
`(average error) / |average mean|`.

# Arguments
- `X_info_ORG`: Raw observable values (used only to infer `N_total = length(X_info_ORG)`).
- `tot_bin`: Upper bound for block sizes to consider; the sweep uses `max_bin = min(tot_bin, N_total)`.
- `means`: Bootstrap means for each block size `b = 1:max_bin` (length must be `max_bin`).
- `errs`:  Bootstrap errors for each block size `b = 1:max_bin` (length must be `max_bin`).
- `win`: Window size for non-overlapping grouping (e.g., `win=10` produces `[1..10], [11..20], ...`).
- `logscale`: If `true`, uses a logarithmic y-axis for the residual-like plot (default `false`).

# Behavior
- Builds `bin_sizes = 1:max_bin` with `max_bin = min(tot_bin, N_total)`.
- Computes `discarded[b] = N_total % b` for each `b` and identifies bins divisible into full blocks (`discarded == 0`).
- Forms non-overlapping windows of length `win` over `bin_sizes` and computes blocked statistics:
  `blk_mu = mean(means[win_k])`, `blk_err = mean(errs[win_k])`,
  `blk_nsr = blk_err / |blk_avg|`.
- Computes the normalized change between adjacent blocked windows:
  `f_k = (NSR_k - NSR_{k-1}) / NSR_k` for `k = 2..nb`.
- Overlays only those `f_k` whose window contains at least one `b` with `discarded[b] == 0`.
- Plots the full residual-like sequence and highlights selected windows.

# Returns
- `(xs, ys, discarded)`:
  - `xs :: Vector{Int}` — right-edge block sizes of highlighted windows,
  - `ys :: Vector{Float64}` — corresponding normalized changes `f_k`,
  - `discarded :: Vector{Int}` — per-`b` counts: `discarded[b] = N_total % b`.

# Notes
- Assumes `means` and `errs` are already computed for all `b = 1:max_bin` in this session.
- Set `logscale=true` if the normalized change spans multiple orders of magnitude.
"""
function nsr_block_scan_blocked_relchange(
    ensemble::String,
    X_info_ORG::AbstractVector{<:Real},
    tot_bin::Int,
    means::Vector{Float64},
    errs::Vector{Float64},
    win::Int;
    logscale::Bool=false,
    save_file::Bool=false
) :: Tuple{Vector{Int}, Vector{Float64}, Vector{Int}}

    # 0) Build bin sizes and basic checks
    N       = length(X_info_ORG)                 # total raw sample count
    max_bin = min(tot_bin, N)
    bin_sizes = collect(1:max_bin)

    JobLoggerTools.assert_benji(
        length(means) == max_bin && length(errs) == max_bin,
        "means/errs must have length == min(tot_bin, length(X_info_ORG))"
    )
    JobLoggerTools.assert_benji(
        win >= 1, 
        "win must be ≥ 1"
    )

    # 1) Per-bin discarded counts: number of samples dropped at block size b
    #    (equivalently: N % b)
    discarded = [N - b * div(N, b) for b in bin_sizes]
    good_bins = findall(==(0), discarded)  # indices where discarded == 0

    # 2) Non-overlapping windows over bin_sizes
    Nbins = length(bin_sizes)
    nb    = div(Nbins, win)                        # number of full windows
    blk_ranges = [((k-1)*win + 1):(k*win) for k in 1:nb]

    # Window representatives and blocked stats
    blk_left  = [bin_sizes[first(r)] for r in blk_ranges]
    blk_right = [bin_sizes[last(r)]  for r in blk_ranges]
    blk_mu    = [Statistics.mean(means[r]) for r in blk_ranges]
    blk_err   = [Statistics.mean(errs[r])  for r in blk_ranges]
    blk_nsr   = map((m, e) -> (e > 0 ? e/abs(m) : NaN), blk_mu, blk_err)

    # 3) Normalized change between adjacent blocked windows:
    #    f_k = (NSR_k - NSR_{k-1}) / NSR_k, k = 2..nb
    nsr_rel_change_curr = vcat(
        NaN,
        (blk_nsr[2:end] .- blk_nsr[1:end-1]) ./ blk_nsr[2:end]
    )
    # x-positions (right edge of each window; align with f_k definition)
    x_at_right = vcat(NaN, blk_right[2:end])

    # 4) Keep windows that contain any bin size with discarded == 0 (k ≥ 2)
    keep_blocks = [any(discarded[r] .== 0) for r in blk_ranges]
    ks = filter(k -> k >= 2, findall(keep_blocks))

    # Coordinates for highlighted points
    xs = [blk_right[k] for k in ks]
    ys = [nsr_rel_change_curr[k] for k in ks]

    # 5) Plot: residual-like sequence + overlay for selected windows
    fig_rc_nsr, ax_rc_nsr = PyPlot.subplots(1,1, figsize=(5.2,5.0), dpi=500)
    ax_rc_nsr.axhline(0, linewidth=1.0, linestyle="--")

    ax_rc_nsr.plot(
        x_at_right, nsr_rel_change_curr;
        marker="o", markersize=4, mfc="none", mec="blue",
        linestyle="-", linewidth=1.0,
        # label="\$(\\mathrm{NSR}_i - \\mathrm{NSR}_{i-1}) / \\mathrm{NSR}_i\$"
    )
    ax_rc_nsr.plot(
        xs, ys;
        marker="o", markersize=6, mfc="none", mec="red",
        linestyle="None",
        label="exactly divisible by block size"
    )

    ax_rc_nsr.set_xlabel("\$B\$ (block size)")
    ax_rc_nsr.set_ylabel("\$r_B\$")
    # ax_rc_nsr.set_title("Residual-like change between adjacent NSR windows (window = $win, $ensemble)")
    ax_rc_nsr.grid(true)
    ax_rc_nsr.legend()
    if logscale
        ax_rc_nsr.set_yscale("log")
    end
    fig_rc_nsr.tight_layout()
    display(fig_rc_nsr)

    # Save (optional) with pdfcrop integration
    basename = "block_size_resid_$(ensemble)_win_size_$(win)"
    resfile  = joinpath(".", "$basename.pdf")
    cropped  = joinpath(".", "$basename-crop.pdf")
    if save_file
        PyPlot.savefig(resfile)
        if Sys.which("pdfcrop") !== nothing
            run(`pdfcrop $resfile`)
            mv(cropped, resfile; force=true)
        end
    end

    # Optional textual summary
    JobLoggerTools.println_benji("block sizes with discarded == 0: $(bin_sizes[good_bins])")
    JobLoggerTools.println_benji("window indices (k) that contain any discarded == 0: $(ks)")
    JobLoggerTools.println_benji("x(right-edge) for those windows: $(xs)")

    return xs, ys, discarded
end

"""
    plot_discarded_vs_blocks(
        X_info_ORG::AbstractVector{<:Real},
        tot_bin::Int;
        style::String = "line"
    ) -> Vector{Int}

Plot the number of discarded raw samples vs. block size.

# Arguments
- `X_info_ORG`: Raw data vector (`length = total samples`).
- `tot_bin`: Maximum reference bin size (typically ``\\le`` length `X_info_ORG`).

# Keywords
- `style`: `"line"` (default) → connect with line only,
           `"point"` → plot markers only,
           `"both"` → line ``+`` markers together.

# Behavior
- Computes discarded samples per block size as `N % b`.
- Plots discarded counts (primary ``y``-axis).
- Plots discarded fraction (``\\%``) (secondary ``y``-axis).
- ``x``-axis label: "block size".

# Returns
- `discarded :: Vector{Int}`: discarded counts for each block size.
"""
function plot_discarded_vs_blocks(
    X_info_ORG::AbstractVector{<:Real},
    tot_bin::Int;
    style::String = "line"
) :: Vector{Int}
    N = length(X_info_ORG)          # total raw samples
    max_bin = min(tot_bin, N)
    bin_sizes = collect(1:max_bin)

    # Discarded counts
    discarded = [N - b * div(N, b) for b in bin_sizes]

    # Plot
    fig, ax = PyPlot.subplots()

    if style == "point"
        ax.plot(bin_sizes, discarded;
                marker="o", markersize=4, mfc="none", mec="blue",
                linestyle="None", label="discarded count")
    elseif style == "both"
        ax.plot(bin_sizes, discarded;
                marker="o", markersize=4, mfc="none", mec="blue",
                linewidth=1.0, label="discarded count")
    else
        ax.plot(bin_sizes, discarded; linewidth=1.5, label="discarded count")
    end

    ax.set_xlabel("block size")
    ax.set_ylabel("discarded raw data (count)")
    ax.set_title("Discarded samples vs. block size")
    ax.grid(true)

    # Secondary y-axis: fraction (%)
    frac = discarded ./ N .* 100
    ax2 = ax.twinx()
    if style == "point"
        ax2.plot(bin_sizes, frac;
                 marker="o", markersize=3, mfc="none", mec="red",
                 linestyle="None", label="discarded (%)")
    elseif style == "both"
        ax2.plot(bin_sizes, frac;
                 marker="o", markersize=3, mfc="none", mec="red",
                 linestyle="--", linewidth=1.0, label="discarded (%)")
    else
        ax2.plot(bin_sizes, frac; linestyle="--", linewidth=1.0, label="discarded (%)")
    end
    ax2.set_ylabel("discarded (percent)")

    fig.tight_layout()
    display(fig)

    return discarded
end

end  # module BlockBinScan