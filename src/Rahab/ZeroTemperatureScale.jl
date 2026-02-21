# ============================================================================
# src/Rahab/ZeroTemperatureScale.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ZeroTemperatureScale

import PyPlot
import Printf: @sprintf
import ..Sarah.AvgErrFormatter
import ..Sarah.JobLoggerTools

"""
    g0sq(
        beta; 
        Nc=3
    ) -> Float64

Compute the bare gauge coupling squared ``g_0^2`` from ``\\beta``.

# Definition
```math
\\beta = \\frac{2N_c}{g_0^2}
\\;\\;\\;\\Rightarrow\\;\\;\\;
g_0^2 = \\frac{2N_c}{\\beta}
```

# Arguments
- `beta::Real` : Gauge coupling parameter ``\\beta``.
- `Nc::Int=3`  : Number of colors (default = ``3``).

# Returns
- `Float64` : The bare coupling squared ``g_0^2``.
"""
g0sq(beta; Nc::Int=3) = (2Nc) / beta


"""
    kappa_c(
        beta; 
        Nc=3
    ) -> Float64

Interpolation formula for the critical hopping parameter ``\\kappa_c`` at ``N_{\\text{f}} = 4``.

# Polynomial form (provided by the user):
```math
\\kappa_c(g_0^2) \\;=\\; 0.125
  + 0.003681192 \\, g_0^2
  + 0.000117 \\,(g_0^2)^2
  + 0.000048 \\,(g_0^2)^3
  - 0.000013 \\,(g_0^2)^4
```
with ``g_0^2 = \\dfrac{2N_c}{\\beta}``.

# Arguments
- `beta::Real` : Gauge coupling parameter ``\\beta``.
- `Nc::Int=3`  : Number of colors (default = ``3``).

# Returns
- `Float64` : The critical hopping parameter ``\\kappa_c``.
"""
function kappa_c(
    beta; 
    Nc::Int=3
)
    g2 = g0sq(beta; Nc)
    g4 = g2^2
    g6 = g4*g2
    g8 = g4^2
    return 0.125 + 0.003681192*g2 + 0.000117*g4 + 0.000048*g6 - 0.000013*g8
end

"""
    mqa_sub(
        kappa, 
        beta; 
        Nc=3
    ) -> Float64

Compute the subtracted bare quark mass for Wilson fermions.

# Definition
```math
m_q a \\;=\\; \\frac{1}{2} \\left( \\frac{1}{\\kappa} - \\frac{1}{\\kappa_c(\\beta)} \\right)
```
where ``\\kappa_c(\\beta)`` is obtained from the interpolation polynomial at given ``\\beta``.

# Arguments
- `kappa::Real` : Hopping parameter ``\\kappa``.
- `beta::Real`  : Gauge coupling parameter ``\\beta``.
- `Nc::Int=3`   : Number of colors (default = `3`).

# Returns
- `Float64` : The subtracted bare quark mass (dimensionless).
"""
mqa_sub(kappa, beta; Nc::Int=3) = 0.5 * (1/kappa - 1/kappa_c(beta; Nc))

"""
    wls_linear(
        x::AbstractVector, 
        y::AbstractVector, 
        σy::AbstractVector
    ) -> (c0, c1, cov, χ², dof)

Perform a weighted least-squares (WLS) linear regression of the form:
```math
y = c_0 + c_1 \\, x
```
using uncertainties ``\\sigma_y`` as weights.

# Arguments
- `x::AbstractVector`   : Independent variable values.
- `y::AbstractVector`   : Dependent variable values.
- `σy::AbstractVector`  : Uncertainties for each ``y`` value.

# Returns
- `c0::Float64` : Intercept.
- `c1::Float64` : Slope.
- `cov::Matrix` : ``2 \\times 2`` covariance matrix of the fit parameters.
- `χ²::Float64` : Chi-squared value of the fit.
- `dof::Int`    : Degrees of freedom (`max(N-2,1)`).
"""
function wls_linear(
    x::AbstractVector, 
    y::AbstractVector, 
    σy::AbstractVector
)
    JobLoggerTools.assert_benji(
        length(x) == length(y) && length(y) == length(σy),
        "vector length mismatch"
    )
    w = 1.0 ./ (σy.^2)
    X = [ones(length(x)) x]
    # Normal equations with weights: (X' W X) θ = X' W y
    WX = X .* w
    XtWX = X' * WX
    XtWy = X' * (w .* y)
    θ = XtWX \ XtWy
    c0, c1 = θ
    # residuals and χ²
    r = y .- (c0 .+ c1 .* x)
    χ2 = sum(@. (r/σy)^2)
    dof = max(length(x) - 2, 1)
    # covariance: σ_fit^2 * (X' W X)^{-1}; with known σy, σ_fit^2 ≈ χ²/dof
    cov = inv(XtWX) * (χ2/dof)
    return (c0, c1, cov, χ2, dof)
end


"""
    wls_quadratic(
        x::AbstractVector, 
        y::AbstractVector, 
        σy::AbstractVector
    ) -> (d0, d1, cov, χ², dof)

Perform a weighted least-squares (WLS) quadratic regression of the form:
```math
y = d_0 \\, x + d_1 \\, x^2
```
(no constant term), using uncertainties ``\\sigma_y`` as weights.

# Arguments
- `x::AbstractVector`   : Independent variable values.
- `y::AbstractVector`   : Dependent variable values.
- `σy::AbstractVector`  : Uncertainties for each ``y`` value.

# Returns
- `d0::Float64` : Linear coefficient.
- `d1::Float64` : Quadratic coefficient.
- `cov::Matrix` : ``2 \\times 2`` covariance matrix of the fit parameters.
- `χ²::Float64` : Chi-squared value of the fit.
- `dof::Int`    : Degrees of freedom (`max(N-2,1)`).
"""
function wls_quadratic(
    x::AbstractVector, 
    y::AbstractVector, 
    σy::AbstractVector
)
    JobLoggerTools.assert_benji(
        length(x) == length(y) && length(y) == length(σy),
        "vector length mismatch"
    )
    w = 1.0 ./ (σy.^2)
    X = [x  x.^2]                    # columns: x, x^2
    WX = X .* w
    XtWX = X' * WX
    XtWy = X' * (w .* y)
    θ = XtWX \ XtWy
    d0, d1 = θ
    r = y .- (d0 .* x .+ d1 .* (x.^2))
    χ2 = sum(@. (r/σy)^2)
    dof = max(length(x) - 2, 1)
    cov = inv(XtWX) * (χ2/dof)
    return (d0, d1, cov, χ2, dof)
end

"""
    read_spectroscopy_table(
        path::AbstractString
    ) -> NamedTuple

Read a spectroscopy table stored in a plain text file where data are arranged
row-by-row and column-by-column in fixed order (cf. `hspec.dat`).  

The expected header is:

    # beta kappa L T sqrt(t0)/a err  mPSa err  mNa err  mqa err

Lines starting with `#` are ignored. Scientific notation with `E`/`e`
is accepted. Malformed lines are skipped.

# Arguments
- `path::AbstractString` : Path to the data file.

# Returns
- `NamedTuple` with fields:
  - `beta::Vector{Float64}`, 
  - `kappa::Vector{Float64}`,
  - `L::Vector{Int}`, 
  - `Nt::Vector{Int}`,
  - `t0a::Vector{Float64}`, 
  - `t0e::Vector{Float64}`,
  - `mpsa::Vector{Float64}`, 
  - `mpse::Vector{Float64}`,
  - `mna::Vector{Float64}`,  
  - `mne::Vector{Float64}`,
  - `mqa::Vector{Float64}`,  
  - `mqae::Vector{Float64}`.
"""
function read_spectroscopy_table(
    path::AbstractString
) :: NamedTuple
    raw = readlines(path)
    rows = Vector{NTuple{12,Float64}}()
    for ln in raw
        s = strip(ln)
        isempty(s) && continue
        startswith(s, "#") && continue
        parts = split(s)
        length(parts) < 12 && continue
        try
            β     = parse(Float64, parts[1])
            κ     = parse(Float64, parts[2])
            L     = parse(Float64, parts[3]) |> x->round(Int,x)
            Nt    = parse(Float64, parts[4]) |> x->round(Int,x)
            t0a   = parse(Float64, replace(parts[5],  "E"=>"e"))
            t0e   = parse(Float64, replace(parts[6],  "E"=>"e"))
            mpsa  = parse(Float64, replace(parts[7],  "E"=>"e"))
            mpse  = parse(Float64, replace(parts[8],  "E"=>"e"))
            mna   = parse(Float64, replace(parts[9],  "E"=>"e"))
            mne   = parse(Float64, replace(parts[10], "E"=>"e"))
            mqa   = parse(Float64, replace(parts[11], "E"=>"e"))
            mqae  = parse(Float64, replace(parts[12], "E"=>"e"))
            push!(rows, (β, κ, L, Nt, t0a, t0e, mpsa, mpse, mna, mne, mqa, mqae))
        catch
            # skip malformed line
        end
    end
    β   = [r[1]  for r in rows]
    κ   = [r[2]  for r in rows]
    L   = [Int(r[3])  for r in rows]
    Nt  = [Int(r[4])  for r in rows]
    t0a = [r[5]  for r in rows]
    t0e = [r[6]  for r in rows]
    mpsa= [r[7]  for r in rows]
    mpse= [r[8]  for r in rows]
    mna = [r[9]  for r in rows]
    mne = [r[10] for r in rows]
    mqa = [r[11] for r in rows]
    mqae= [r[12] for r in rows]
    return (beta=β, kappa=κ, L=L, Nt=Nt, t0a=t0a, t0e=t0e, mpsa=mpsa, mpse=mpse,
            mna=mna, mne=mne, mqa=mqa, mqae=mqae)
end

"""
    plot_mps2_vs_mqa_sub!(
        x::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
        σy::AbstractVector{<:Real},
        d0::Real,
        d1::Real;
        label_data::AbstractString = "raw T=0",
        label_fit::AbstractString = "fit f(x)",
        save_file::Union{Bool,AbstractString} = false
    ) -> Tuple{PyPlot.Figure, PyPlot.Axes}

Plot ``(m_{\\text{PS}} \\, a)^2`` vs ``m_q \\, a`` with error bars and quadratic fit
```math
(m_{\\text{PS}} \\, a)^2 = d_0 \\, x + d_1 \\, x^2
```

# Arguments
- `x`    : Vector of ``m_q \\, a`` values.
- `y`    : Vector of ``(m_{\\text{PS}} \\, a)^2`` values.
- `σy`   : Vector of uncertainties for ``y``.
- `d0,d1`: Fit coefficients for the quadratic model.

# Keywords
- `label_data` : Legend label for data points.
- `label_fit`  : Legend label for the fit curve.
- `save_file`  : `false` (no save), `true` (default name), or `String` (filename).

# Returns
- `(fig, ax)` : [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure and axis objects.
"""
function plot_mps2_vs_mqa_sub!(
    x, 
    y, 
    σy, 
    d0, 
    d1; 
    label_data="raw T=0", 
    label_fit="fit f(x)", 
    save_file=false
)
    fig, ax = PyPlot.subplots()

    # Fit curve first (for behind markers)
    xx = range(minimum(x), maximum(x), length=200)
    yy = d0 .* xx .+ d1 .* (xx .^ 2)
    ax.plot(xx, yy; color="orange", linewidth=2.5, alpha=0.9, label=label_fit)

    # Error bars only; points drawn with scatter below
    ax.errorbar(x, y, yerr=σy, fmt="none",
        ecolor=(0.0,0.0,1.0,0.8), elinewidth=1.5, capsize=3, capthick=1.5)

    # Hollow markers (edge only)
    ax.scatter(x, y;
        s=72, marker="o",
        facecolors="none",
        edgecolors=(0.0,0.0,1.0,0.8),
        linewidths=1.2,
        label=label_data)

    ax.set_xlabel("\$m_q a\$")
    ax.set_ylabel("\$(m_{\\mathrm{PS}} a)^2\$")
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    display(fig)

    if save_file === true
        PyPlot.savefig("mps2_vs_mqa_sub.pdf")
    elseif save_file isa AbstractString
        PyPlot.savefig(String(save_file))
    end
    return fig, ax
end

"""
    plot_t0a_vs_mqa_sub!(
        x::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
        σy::AbstractVector{<:Real},
        c0::Real,
        c1::Real;
        label_data::AbstractString = "raw T=0",
        label_fit::AbstractString = "fit f(x)",
        save_file::Union{Bool,AbstractString} = false
    ) -> Tuple{PyPlot.Figure, PyPlot.Axes}

Plot ``\\sqrt{t_0}/a`` vs ``m_q \\, a`` with error bars and linear fit
```math
\\frac{\\sqrt{t_0}}{a} = c_0 + c_1 \\, x \\,.
```

# Arguments
- `x`    : Vector of ``m_q \\, a`` values.
- `y`    : Vector of ``\\sqrt{t_0}/a`` values.
- `σy`   : Vector of uncertainties for ``y``.
- `c0,c1`: Fit coefficients for the linear model.

# Keywords
- `label_data` : Legend label for data points.
- `label_fit`  : Legend label for the fit line.
- `save_file`  : `false` (no save), `true` (default name), or `String` (filename).

# Returns
- `(fig, ax)` : [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) figure and axis objects.
"""
function plot_t0a_vs_mqa_sub!(x, y, σy, c0, c1; label_data="raw T=0", label_fit="fit f(x)", save_file=false)
    fig, ax = PyPlot.subplots()

    # Fit line first
    xx = range(minimum(x), maximum(x), length=200)
    yy = c0 .+ c1 .* xx
    ax.plot(xx, yy; color="orange", linewidth=2.5, alpha=0.9, label=label_fit)

    # Error bars only
    ax.errorbar(x, y, yerr=σy, fmt="none",
        ecolor=(0.0,0.0,1.0,0.8), elinewidth=1.5, capsize=3, capthick=1.5)

    # Hollow markers (edge only)
    ax.scatter(x, y;
        s=72, marker="o",
        facecolors="none",
        edgecolors=(0.0,0.0,1.0,0.8),
        linewidths=1.2,
        label=label_data)

    ax.set_xlabel("\$m_q a\$")
    ax.set_ylabel("\$\\sqrt{t_0}/a\$")
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    display(fig)

    if save_file === true
        PyPlot.savefig("t0a_vs_mqa_sub.pdf")
    elseif save_file isa AbstractString
        PyPlot.savefig(String(save_file))
    end
    return fig, ax
end

"""
    const ħc_MeVfm

Reduced Planck constant times the speed of light in ``[\\text{MeV} \\cdot \\text{fm}]``.

Used for converting between inverse lengths ``[1/\\text{fm}]`` and energies ``[\\text{MeV}]``.
"""
const ħc_MeVfm = 197.3269804

"""
    id_conversions(t0a; invt0) -> (ainv_GeV, a_fm)

Convert ``\\sqrt{t_0}/a`` at a given ``\\beta`` into lattice spacing quantities using the ``\\text{GeV}`` convention:

# Definitions (user convention)

- ``a^{-1}`` in ``[\\text{GeV}]``
```math
\\texttt{ainv\\_GeV} = \\texttt{t0a} \\times \\texttt{invt0}
```
- ``a`` in ``[\\text{fm}]`` (``1000 \\; \\text{MeV} = 1 \\; \\text{GeV}``)
```math
\\texttt{a\\_fm} = \\frac{\\hbar c \\, [\\text{MeV} \\cdot \\text{fm}]}{\\texttt{ainv\\_GeV} \\times 1000}
```

# Arguments
- `t0a::Real`  : Value of ``\\dfrac{\\sqrt{t_0}}{a}`` (dimensionless).
- `invt0::Float64` : ``\\dfrac{1}{\\sqrt{t_0}_{\\text{phys}}}`` in ``[\\text{GeV}]``.

# Returns
- `(ainv_GeV::Float64, a_fm::Float64)` :
    - `ainv_GeV` : ``a^{-1}`` in ``[\\text{GeV}]``.
    - `a_fm`     : ``a`` in ``[\\text{fm}]``.

Notes
- If you instead keep `invt0` in ``[\\text{fm}^{-1}]``, do not use this function; either convert
  to ``\\text{GeV}`` first (`invt0_GeV = invt0_perfm * ħc_MeVfm / 1000`) or write a dedicated
  per-fm variant to avoid mixed units.
"""
function id_conversions(t0a; invt0::Float64)
    ainv_GeV = t0a * invt0
    a_fm = ħc_MeVfm / (ainv_GeV * 1000.0)
    return ainv_GeV, a_fm
end

"""
    mps_GeV(
        mpsa, 
        ainv_GeV
    ) -> Float64

Convert a dimensionless pseudoscalar mass ``m_{\\text{PS}} \\, a`` to physical units ``[\\text{GeV}]``
using lattice inverse spacing in ``[\\text{GeV}]``:

```math
    m_{PS}[\\text{GeV}] = \\texttt{mPSa} \\times (a^{-1}[\\text{GeV}])
```

# Arguments
- `mpsa::Real`     : Dimensionless mass ``m_{\\text{PS}} \\, a``.
- `ainv_GeV::Real` : Lattice inverse spacing ``a^{-1}`` in ``[\\text{GeV}]``.

# Returns
- `Float64` : ``m_{\\text{PS}}`` in ``[\\text{GeV}]``.

# Warning
- Do not pass `1/a_fm` here. If you only have `a_fm` in ``\\text{fm}``, either invert and
  multiply by ``\\dfrac{\\hbar c}{1000}`` outside, or write a separate helper for the ``[\\text{fm}^{-1}]`` convention.
"""
mps_GeV(mpsa, ainv_GeV) = mpsa * ainv_GeV


"""
    run_zero_analysis(
        path::String,
        beta::String;
        Nc::Int = 3,
        invt0_GeV::Float64 = 1.347,
        target_kappas::Vector{Float64}
    ) -> NamedTuple{(:beta, :Nc, :invt0_GeV, :fit_mps2, :fit_t0a, :results)}

Run a zero-temperature ``\\beta``-slice analysis from an `hspec`-style table.

Pipeline:
1. Filter rows at the specified ``\\beta``.
2. Build ``x \\equiv m_q \\, a`` via the ``N_\\text{f}=4`` ``\\kappa_c(g_0^2)`` interpolation and compute:
    - ``(m_{\\text{PS}} \\, a)^2``  and  ``\\sqrt{t_0}/a``.
3. Perform weighted least-squares fits:
    - ``(m_{\\text{PS}} \\, a)^2 = d_0 \\, x + d_1 \\, x^2``,
    - ``\\sqrt{t_0}/a = c_0 + c_1 \\, x``.
4. Print fit summaries with compact avg±err formatting.
5. Predict ID-style quantities at `target_kappas`:
    - `t0a`, ``a^{-1}\\;\\text{[GeV]}``, ``a\\;\\text{[fm]}``, ``m_{\\text{PS}}\\;\\text{[GeV]}``.
6. Plot ``(m_{\\text{PS}} \\, a)^2`` vs ``m_q \\, a``  and  ``\\sqrt{t_0}/a`` vs ``m_q \\, a`` (PDF saving handled inside each plotter).

Unit convention:
- `invt0_GeV` is ``1/\\sqrt{t_0}_{\\text{phys}}`` in ``[\\text{GeV}]``.
- Then  ``a^{-1}\\;\\text{[GeV]}`` = ``\\sqrt{t_0}/a \\times 1/\\sqrt{t_0}_{\\text{phys}}\\;\\text{[GeV]}``, and ``m_{\\text{PS}}\\;\\text{[GeV]}`` = ``(m_{\\text{PS}} \\, a) \\times a^{-1}\\;\\text{[GeV]}``.

# Arguments
- `path::String` : Input file path (e.g., `"hspec.dat"`), whitespace-delimited with the header
    `# beta kappa L T sqrt(t0)/a err  mPSa err  mNa err  mqa err`.
- `beta::String` : ``\\beta`` value selector (e.g., `"1.60"`).

# Keywords
- `Nc::Int=3`                  : Number of colors.
- `invt0_GeV::Float64=1.347`   : ``1/\\sqrt{t_0}_{\\text{phys}}`` in ``[\\text{GeV}]`` (``\\text{GeV}`` convention).
- `target_kappas::Vector{Float64}` : ``\\kappa`` list at which to make predictions (required).

# Returns
- `NamedTuple` with fields:
    - `beta::Float64`
    - `Nc::Int`
    - `invt0_GeV::Float64`
    - `fit_mps2` : `(d0, d1, d0err, d1err, chi2_dof)`
    - `fit_t0a`  : `(c0, c1, c0err, c1err, chi2_dof)`
    - `results::Vector{NamedTuple}` where each item has
        `(kappa, mqa_sub, t0a, ainv_GeV, a_fm, mPS_GeV)`

Notes
- Printing uses [`AvgErrFormatter.avgerr_e2d_from_float`](@ref Deborah.Sarah.AvgErrFormatter.avgerr_e2d_from_float) for compact avg±err strings.
- Plots are produced via [`plot_mps2_vs_mqa_sub!`](@ref) and [`plot_t0a_vs_mqa_sub!`](@ref).
"""
function run_zero_analysis(
    path::String,
    beta::String;
    Nc::Int=3,
    invt0_GeV::Float64=1.347,
    target_kappas::Vector{Float64}
)
    # parse beta if passed as string
    β = beta isa AbstractString ? parse(Float64, beta) : float(beta)

    # ---- load and filter ----
    tab = read_spectroscopy_table(path)
    sel = findall(b -> isapprox(b, β; atol=1e-8), tab.beta)
    isempty(sel) && JobLoggerTools.error_benji("No rows for beta=$(β) found in $path")

    κ    = tab.kappa[sel]
    mpsa = tab.mpsa[sel]
    mpse = tab.mpse[sel]
    t0a  = tab.t0a[sel]
    t0e  = tab.t0e[sel]

    # x = mqa_sub(κ; β)
    x = [mqa_sub(k, β; Nc) for k in κ]

    # --- Fit (mPS a)^2 = d0 x + d1 x^2 with σ_y = 2 mPSa * δmPSa
    y_mps2   = mpsa .^ 2
    σy_mps2  = 2 .* mpsa .* mpse
    d0, d1, cov_d, χ2_d, dof_d = wls_quadratic(x, y_mps2, σy_mps2)
    d0err = sqrt(cov_d[1,1]); d1err = sqrt(cov_d[2,2])

    # --- Fit sqrt(t0)/a = c0 + c1 x with σ_y = δ(sqrt(t0)/a)
    c0, c1, cov_c, χ2_c, dof_c = wls_linear(x, t0a, t0e)
    c0err = sqrt(cov_c[1,1]); c1err = sqrt(cov_c[2,2])

    # ==== PRINT BLOCK (fits + predictions) ====================================
    JobLoggerTools.println_benji("=== Fits at β=$(beta) (Nc=$(Nc)) ===")
    c0_ae = AvgErrFormatter.avgerr_e2d_from_float(c0, c0err)
    c1_ae = AvgErrFormatter.avgerr_e2d_from_float(c1, c1err)
    JobLoggerTools.println_benji(
        "sqrt(t0)/a = c0 + c1 x    : c0 = $(c0_ae), c1 = $(c1_ae), chi2/dof = $(round(χ2_c/dof_c, digits=3))"
    )
    d0_ae = AvgErrFormatter.avgerr_e2d_from_float(d0, d0err)
    d1_ae = AvgErrFormatter.avgerr_e2d_from_float(d1, d1err)
    JobLoggerTools.println_benji(
        "(mPS a)^2 = d0 x + d1 x^2 : d0 = $(d0_ae), d1 = $(d1_ae), chi2/dof = $(round(χ2_d/dof_d, digits=3))"
    )

    JobLoggerTools.println_benji("\n--- Predictions at target kappas ---")
    JobLoggerTools.println_benji(
        @sprintf("%8s  %10s  %10s  %12s  %10s  %10s",
                "kappa", "mqa_sub", "t0a", "a^{-1}[GeV]", "a[fm]", "mPS[GeV]")
    )
    results = Vector{NamedTuple}()
    for k in target_kappas
        xk     = mqa_sub(k, β; Nc)
        t0a_k  = c0 + c1*xk
        mpsa_k = sqrt(max(0.0, d0*xk + d1*xk^2))
        ainv_GeV, a_fm = id_conversions(t0a_k; invt0=invt0_GeV)   # a^{-1}[GeV], a[fm]
        mps_GeV_k      = mps_GeV(mpsa_k, ainv_GeV)                # m_PS[GeV] = (mPS*a)*a^{-1}[GeV]
        JobLoggerTools.println_benji(
            @sprintf("%8.5f  %10.6f  %10.5f  %12.5f  %10.5f  %10.5f",
                    k, xk, t0a_k, ainv_GeV, a_fm, mps_GeV_k)
        )
        push!(results, (kappa=k, mqa_sub=xk, t0a=t0a_k, ainv_GeV=ainv_GeV, a_fm=a_fm, mPS_GeV=mps_GeV_k))
    end
    # ==========================================================================

    # --- Plots
    plot_mps2_vs_mqa_sub!(x, y_mps2, σy_mps2, d0, d1;
        label_data="raw \$T=0\$", label_fit="fit \$f(x)\$")

    plot_t0a_vs_mqa_sub!(x, t0a, t0e, c0, c1;
        label_data="raw \$T=0\$", label_fit="fit \$f(x)\$")

    return (
        beta = β,
        Nc = Nc,
        invt0_GeV = invt0_GeV,
        fit_mps2 = (d0=d0, d1=d1, d0err=d0err, d1err=d1err, chi2_dof=χ2_d/dof_d),
        fit_t0a  = (c0=c0, c1=c1, c0err=c0err, c1err=c1err, chi2_dof=χ2_c/dof_c),
        results = results
    )
end

end # module ZeroTemperatureScale