# ============================================================================
# src/Rahab/ObservableHistory.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ObservableHistory

import ..PyPlot
import ..Statistics
import ..Printf: @sprintf

import ..Sarah.JobLoggerTools

"""
    observable_history(
        ensemble::AbstractString,
        conf_idx::AbstractVector{<:Integer},
        X_info_ORG::AbstractVector{<:Real},
        observable::AbstractString
    ) -> Nothing

Plot the history of a given observable against configuration indices for a
specific ensemble.

# Arguments
- `ensemble`: Ensemble name, displayed in the plot title.
- `conf_idx`: Vector of configuration indices (``x``-axis).
- `X_info_ORG`: Vector of observable values (``y``-axis).
- `observable`: Name of the observable, used in the plot title and legend.

The function displays the figure inline (suitable for [`Jupyter`](https://jupyter.org/)).
"""
function observable_history(
    ensemble::String,
    conf_idx::AbstractVector{<:Integer},
    X_info_ORG::AbstractVector{<:Real},
    observable::AbstractString
)::Nothing
    
    # fig, ax = PyPlot.subplots(figsize=(6, 3), dpi=500)
    fig, ax = PyPlot.subplots()
    
    ax.plot(conf_idx, X_info_ORG; label="$observable", color="blue", linewidth=1.5)
    
    ax.set_xlabel("configuration index")
    ax.set_ylabel("value")
    
    ax.set_title("history of $observable ($ensemble)")
    
    ax.grid(true)
    
    ax.legend()
    
    fig.tight_layout()
    
    display(fig)
    
    return nothing
end

"""
    autocorr(
        x::AbstractVector{<:Real},
        maxlag::Integer
    ) -> Vector{Float64}

Compute the (normalized) autocorrelation function ``\\rho(\\Delta)`` of a 1D
observable up to lag `maxlag`. The result is normalized so that ``\\rho(0) =
1``.

# Arguments
- `x`: Observable time series (1D vector).
- `maxlag`: Maximum lag Δ to evaluate (inclusive).

# Returns
- `Vector{Float64}`: Autocorrelation values at lags `Δ = 0:maxlag`, with
  ``\\rho(0) = 1.0``.
"""
function autocorr(
    x::AbstractVector{<:Real}, 
    maxlag::Integer
)::Vector{Float64}
    n = length(x)
    μ = Statistics.mean(x)
    v = Statistics.var(x)
    ρ = zeros(Float64, maxlag + 1)
    @inbounds for Δ in 0:maxlag
        num = sum((x[1:n-Δ] .- μ) .* (x[Δ+1:n] .- μ))
        ρ[Δ+1] = num / ((n - Δ) * v)
    end
    return ρ
end

"""
    tau_int_from_rho(
        ρ::AbstractVector{<:Real};
        window::Symbol=:first_nonpositive
    ) -> Tuple{Float64,Int}

Estimate ``\\tau_{\\text{int}}`` from a precomputed autocorrelation array
``\\rho(\\Delta)``, `ρ[1]=ρ(0)=1`.

# Arguments
- `ρ`: Vector of autocorrelations at lags `Δ=0,1,...` (`ρ[1]=1`).
- `window`: Truncation rule for the summation over `Δ` ``\\ge 1``.
    - `:first_nonpositive` — sum until the first ``\\Delta`` with
      ``\\rho(\\Delta)`` ``\\le 0`` (simple positive-sequence window).
    - `:fixed` — sum all entries provided in ``\\rho`` (i.e., up to `Δ =
      length(ρ)-1)`.

# Returns
- `(τ_int, Δ_cut)` where `Δ_cut` is the last lag included in the sum.
"""
function tau_int_from_rho(
    ρ::AbstractVector{<:Real};
    window::Symbol=:first_nonpositive
)::Tuple{Float64,Int}
    JobLoggerTools.assert_benji(
        !isempty(ρ),
        "ρ must be non-empty and include ρ(0) at index 1"
    )
    posρ = ρ[2:end]  # drop ρ(0)
    Δ_cut = length(posρ)
    if window === :first_nonpositive
        first_nonpos = findfirst(≤(0), posρ)
        if first_nonpos !== nothing
            Δ_cut = first_nonpos - 1
        end
    elseif window === :fixed
        Δ_cut = length(posρ)
    else
        JobLoggerTools.error_benji("Unsupported window=:$(window)")
    end
    Δ_cut = max(0, Δ_cut)
    τ = 0.5 + (Δ_cut == 0 ? 0.0 : sum(@view posρ[1:Δ_cut]))
    return (τ, Δ_cut)
end

"""
    plot_autocorr_tauint(
        ensemble::AbstractString,
        x::AbstractVector{<:Real},
        observable::AbstractString;
        maxlag::Integer=200,
        window::Symbol=:first_nonpositive
    ) -> Nothing

Plot the autocorrelation ``\\rho(\\Delta)`` of an observable and overlay the
cumulative 

```math
\\displaystyle{\\tau_{\\text{int}} = \\dfrac{1}{2} +
\\sum_{k=1}^{\\Delta} \\rho(k)}
``` 

to visualize convergence of the integrated
autocorrelation time.

The figure shows:
- ``\\rho(\\Delta)`` for `Δ = 0:maxlag` (line with markers)
- A vertical line at `Δ_cut` (last lag included in the ``\\tau_{\\text{int}}(\\Delta)`` sum)
- A shaded band over `Δ` ``\\in`` `[1, Δ_cut]` indicating the summed region
- On the right ``y``-axis, the cumulative ``\\tau_{\\text{int}}(\\Delta)`` curve

# Arguments
- `ensemble`: Ensemble name for title.
- `x`: Observable time series (1D vector).
- `observable`: Label used in title/legend.

# Keyword Arguments
- `maxlag`: Maximum lag for autocorrelation (clipped to `length(x)-1`).
- `window`: Truncation rule for ``\\tau_{\\text{int}}(\\Delta)`` (`:first_nonpositive` or `:fixed`).

# Notes
- A typical effective spacing between independent samples is about ``2 \\, \\tau_{\\text{int}}``.
- Inspect the ``\\tau_{\\text{int}}(\\Delta)`` curve plateau for stability.

The function displays the figure inline.
"""
function plot_autocorr_tauint(
    ensemble::AbstractString,
    x::AbstractVector{<:Real},
    observable::AbstractString;
    maxlag::Integer=1000,
    window::Symbol=:first_nonpositive
)::Nothing
    JobLoggerTools.assert_benji(
        length(x) ≥ 2, 
        "time series too short"
    )
    maxlag = min(maxlag, length(x)-1)

    # --- autocorr ρ(Δ), normalized so ρ(0)=1
    n = length(x)
    μ = Statistics.mean(x)
    v = Statistics.var(x)
    ρ = zeros(Float64, maxlag + 1)
    @inbounds for Δ in 0:maxlag
        num = sum((x[1:n-Δ] .- μ) .* (x[Δ+1:n] .- μ))
        ρ[Δ+1] = num / ((n - Δ) * v)
    end

    # --- τ_int from ρ with window
    posρ = ρ[2:end]                      # drop ρ(0)
    Δ_cut = length(posρ)
    if window === :first_nonpositive
        first_nonpos = findfirst(≤(0), posρ)
        if first_nonpos !== nothing
            Δ_cut = first_nonpos - 1
        end
    elseif window !== :fixed
        JobLoggerTools.error_benji("Unsupported window=:$(window)")
    end
    Δ_cut = max(0, Δ_cut)
    τ = 0.5 + (Δ_cut == 0 ? 0.0 : sum(@view posρ[1:Δ_cut]))

    # cumulative τ_int(Δ)
    csum = cumsum(posρ)
    τ_curve = 0.5 .+ csum                 # length = maxlag

    # --- plot
    fig, ax = PyPlot.subplots()
    lags = 0:maxlag
    ax.plot(lags, ρ; marker="o", linewidth=1.2, markersize=3, label="\$\\rho (\\Delta)\$")
    ax.axhline(0.0; linestyle="--", linewidth=1.0)
    if Δ_cut > 0
        ax.axvspan(1 - 0.5, Δ_cut + 0.5; alpha=0.15, color="C1", label="summed window")
        ax.axvline(Δ_cut; linestyle=":", linewidth=1.2, color="C1")
    end
    ax.set_xlabel("lag \$\\Delta\$")
    ax.set_ylabel("\$\\rho (\\Delta)\$")
    ax.set_title("autocorrelation and \$\\tau_{\\mathrm{int}}\$ ($observable, $ensemble)")
    ax.grid(true)

    axr = ax.twinx()
    if maxlag ≥ 1
        axr.plot(1:maxlag, τ_curve; linewidth=1.2, linestyle="-", color="C2", label="\$\\tau_{\\mathrm{int}}(\\Delta)\$")
    end
    axr.set_ylabel("\$\\tau_{\\mathrm{int}}(\\Delta)\$")

    # merged legend
    lines_l = ax.get_lines();  labels_l = [l.get_label() for l in lines_l]
    lines_r = axr.get_lines(); labels_r = [l.get_label() for l in lines_r]
    ax.legend(vcat(lines_l, lines_r), vcat(labels_l, labels_r); loc="best", fontsize="small")

    fig.tight_layout()
    display(fig)

    JobLoggerTools.println_benji(
        "[autocorr] ensemble=$(ensemble), obs=$(observable): " *
        "tau_int≈$(round(τ, digits=3)), " *
        "two_tau≈$(round(2τ, digits=3)), " *
        "Δ_cut=$(Δ_cut)"
    )

    return nothing
end

end  # module ObservableHistory