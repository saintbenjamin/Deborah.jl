# ============================================================================
# src/Rebekah/Comparison.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Comparison

import ..Sarah.JobLoggerTools

"""
    check_overlap(
        a::Float64, 
        ea::Float64, 
        b::Float64, 
        eb::Float64
    ) -> Int

Check whether two values ``\\mu_a ± \\sigma_a`` and ``\\mu_b ± \\sigma_b`` statistically overlap.

- Returns `2` if both intervals mutually include each other.
- Returns `1` if only one direction overlaps.
- Returns `0` if there is no overlap at all.

# Arguments
- `a::Float64`  : Central value of the first measurement.
- `ea::Float64` : Uncertainty (error bar) of the first value.
- `b::Float64`  : Central value of the second measurement.
- `eb::Float64` : Uncertainty (error bar) of the second value.

# Returns
- `Int` : Overlap status  
    - `0`: no overlap  
    - `1`: one-sided overlap  
    - `2`: mutual overlap
"""
function check_overlap(
    a::Float64, 
    ea::Float64, 
    b::Float64, 
    eb::Float64
)::Int
    cond1 = (a >= b - eb) && (a <= b + eb)
    cond2 = (b >= a - ea) && (b <= a + ea)
    return cond1 && cond2 ? 2 : (cond1 || cond2 ? 1 : 0)
end

"""
    check_overlap_type_b(
        μa::Float64, 
        σa::Float64, 
        μb::Float64; 
        σ_floor::Float64=1e-12
    ) -> Float64

Return a normalized separation between two central values using the uncertainty of `a`:

```math
d \\;\\equiv\\; \\frac{|\\mu_a - \\mu_b|}{\\max(\\sigma_a, \\sigma_{\\text{floor}})}.
```

# Interpretation

* `d = 0`  : identical central values.
* `d = 1`  : `μb` is one-σ away from `μa` (measured **relative to `σa`**).
* Larger `d` means farther separation in units of `σa`.

# Notes

* This is intentionally **asymmetric**: it measures distance relative to `σa` only.
* A small floor `σ_floor` avoids division by (near) zero.

# See also

* [`bhattacharyya_coeff_normals`](@ref) — overlap proxy using both variances.
* [`check_overlap`](@ref) — interval-overlap classifier (`0`/`1`/`2`).
"""
function check_overlap_type_b(
    μa::Float64, 
    σa::Float64, 
    μb::Float64; 
    σ_floor::Float64=1e-12
)::Float64
    denom = max(σa, σ_floor)
    return abs(μa - μb) / denom
end

"""
    err_ratio(
        mles_err::Float64, 
        orig_err::Float64
    ) -> Float64

Compute the relative error ratio between machine-learned and original estimates.

This function returns the ratio of ``\\sigma_{\\text{MLES}} / \\sigma_{\\text{ORIG}}``, where ``\\sigma_{\\text{MLES}}`` is the 
estimated error from a machine-learned method, and ``\\sigma_{\\text{ORIG}}`` is the baseline error.

# Arguments
- `mles_err::Float64`: Machine-learned error estimate.
- `orig_err::Float64`: Original (baseline) error estimate.

# Returns
- A `Float64` representing the relative error ratio.
"""
function err_ratio(
    mles_err::Float64, 
    orig_err::Float64
)::Float64
    return mles_err / orig_err
end

"""
    build_overlap_and_error_dicts(
        new_dict::Dict{String, Array{Float64,2}},
        keys::Vector{Symbol},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector
    ) -> Tuple{Dict{Tuple{String,String}, Array{Int,2}}, Dict{Tuple{String,String}, Array{Float64,2}}}

Construct dictionaries containing overlap flags and error ratios between prediction and reference observables.

This function iterates over all combinations of observable keys and prediction tags to compare against a fixed reference.
For each pair, it computes:
- An overlap matrix (`0`, `1`, `2`) via [`check_overlap`](@ref)
- An error ratio matrix via [`err_ratio`](@ref)
The results are returned as two dictionaries keyed by (`pred`, `orig`) string pairs.

# Arguments
- `new_dict`: Dictionary of 2D arrays containing average and error values for all observables.
- `keys`: Observable types (e.g., `:Deborah`, `:TrM1`, etc.).
- `pred_tags`: Prediction tags (e.g., `:Y_P1`, `:Y_P2`).
- `orig_tag`: Tag to use for the reference data (usually `:Y_BS`).
- `labels`: Labeled set index vector for the vertical axis (`LBP`).
- `trains_ext`: Training set index vector for the horizontal axis (`TRP`).

# Returns
- A tuple of two dictionaries:
    - `chk_dict`: Maps `(pred, orig)` → overlap matrix `(Int)`
    - `err_dict`: Maps `(pred, orig)` → error ratio matrix `(Float64)`
"""
function build_overlap_and_error_dicts(
    new_dict::Dict{String, Array{Float64,2}},
    keys::Vector{Symbol},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector
)::Tuple{Dict{Tuple{String,String}, Array{Int,2}}, Dict{Tuple{String,String}, Array{Float64,2}}}

    chk_dict = Dict{Tuple{String,String}, Array{Int,2}}()
    err_dict = Dict{Tuple{String,String}, Array{Float64,2}}()

    for key in keys
        for tag in pred_tags
            if key == :Deborah
                pred = string(tag)
                orig = "Y:" * string(orig_tag)
            else
                pred = String(key) * ":" * String(tag)
                orig = String(key) * ":Y:" * String(orig_tag)
            end


            chk_dict[(pred, orig)] = [
                Comparison.check_overlap(
                    new_dict[pred * ":avg"][ilb, itr],
                    new_dict[pred * ":err"][ilb, itr],
                    new_dict[orig * ":avg"][ilb, itr],
                    new_dict[orig * ":err"][ilb, itr]
                )
                for ilb in eachindex(labels), itr in eachindex(trains_ext)
            ]

            err_dict[(pred, orig)] = [
                Comparison.err_ratio(
                    new_dict[pred * ":err"][ilb, itr],
                    new_dict[orig * ":err"][ilb, itr]
                )
                for ilb in eachindex(labels), itr in eachindex(trains_ext)
            ]
        end
    end

    return chk_dict, err_dict
end

"""
    bhattacharyya_coeff_normals(
        μa::Float64, 
        σa::Float64,
        μb::Float64, 
        σb::Float64;
        σ_floor::Float64=1e-12
    ) -> Float64


Compute the Bhattacharyya coefficient (``\\mathrm{BC}``) between two normal distributions  
`` \\mathcal{N}(\\mu_a, \\sigma_a^2) `` and `` \\mathcal{N}(\\mu_b, \\sigma_b^2) ``.

# Range
- `` \\mathrm{BC} \\in [0, 1] ``
- `` \\mathrm{BC} = 1 ``: complete overlap (identical distributions)  
- `` \\mathrm{BC} = 0 ``: no overlap in the Bhattacharyya sense  

# Notes
- Inputs `σa`, `σb` are interpreted as ``1 \\sigma`` standard deviations.  
- To avoid degeneracy when ``\\sigma \\approx 0``, both `σa` and `σb` are floored:  
```math
  \\sigma \\leftarrow \\max(\\sigma, \\sigma_{\\text{floor}})
```

# Formula
For two normals,
```math
\\mathrm{BC}(\\mu_a, \\sigma_a; \\mu_b, \\sigma_b) \\;=\\;
\\sqrt{ \\frac{2\\,\\sigma_a \\sigma_b}{\\sigma_a^2 + \\sigma_b^2} }
\\;
\\exp\\!\\left(
  - \\frac{(\\mu_a - \\mu_b)^2}{4(\\sigma_a^2 + \\sigma_b^2)}
\\right).
```

# Notes
- This function interprets input uncertainties as ``1 \\sigma`` standard deviations.
- To avoid degeneracy when ``\\sigma \\approx 0``, a small floor `σ_floor` is applied.
"""
function bhattacharyya_coeff_normals(
    μa::Float64, 
    σa::Float64,
    μb::Float64, 
    σb::Float64;
    σ_floor::Float64=1e-12
)::Float64
    σ1 = max(σa, σ_floor)
    σ2 = max(σb, σ_floor)
    s2 = σ1^2 + σ2^2
    pref = sqrt((2 * σ1 * σ2) / s2)
    expo = exp(- (μa - μb)^2 / (4 * s2))
    bc = pref * expo
    return bc < 0 ? 0.0 : (bc > 1 ? 1.0 : bc)
end

"""
    hellinger_from_bc(bc::Float64) -> Float64

Convert Bhattacharyya coefficient into Hellinger distance.

## Range
* ``H \\in [0,1]``
* ``H = 0``: best (identical distributions)
* ``H = 1``: worst (maximal separation)

## Formula
Given a Bhattacharyya coefficient (``\\mathrm{BC}``),
```math
H = \\sqrt{\\,1 - \\mathrm{BC}\\,}.
```
Here, (``\\mathrm{BC}``) is clamped into (``[0,1]``) before evaluation.
"""
@inline function hellinger_from_bc(bc::Float64)::Float64
    bc_clamped = bc < 0 ? 0.0 : (bc > 1 ? 1.0 : bc)
    return sqrt(1 - bc_clamped)
end

"""
    build_bhattacharyya_dicts(
        new_dict::Dict{String, Array{Float64,2}},
        keys::Vector{Symbol},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64 = 1e-12,
        also_hellinger::Bool = false
    ) -> Tuple{Dict{Tuple{String,String}, Array{Float64,2}},
             Union{Nothing, Dict{Tuple{String,String}, Array{Float64,2}}}}

Construct Bhattacharyya-coefficient (``\\mathrm{BC}``) matrices — and optionally
Hellinger-distance matrices -- for each `(observable key, prediction tag)` pair
against a fixed reference tag, following the same naming scheme as
[`build_overlap_and_error_dicts`](@ref).

Key construction matches your logic:
- If `key == :Deborah`
    - `pred = string(tag)`
    - `orig = "Y:" * string(orig_tag)`
- Else
    - `pred = string(key) * ":" * string(tag)`
    - `orig = string(key) * ":Y:" * string(orig_tag)`

For each grid point `(ilb, itr)`:
-` μ_pred = new_dict[pred * ":avg"][ilb, itr]`
- `σ_pred = new_dict[pred * ":err"][ilb, itr]`
- `μ_orig = new_dict[orig * ":avg"][ilb, itr]`
- `σ_orig = new_dict[orig * ":err"][ilb, itr]`
and [`BC = bhattacharyya_coeff_normals(μ_pred, σ_pred, μ_orig, σ_orig)`](@ref bhattacharyya_coeff_normals).

# Arguments
- `new_dict`   : Must contain the 2D arrays for `":avg"` and `":err"` of both `pred`/`orig`.
- `keys`       : Observable keys (e.g., `[:Deborah, :cond, :susp, :skew, :kurt]`).
- `pred_tags`  : Prediction tags (e.g., `[:Y_P1, :Y_P2]`).
- `orig_tag`   : Reference tag (e.g., `:Y_BS`).
- `labels`     : Row axis (`LBP`-like index).
- `trains_ext` : Column axis (`TRP`-like index).
- `σ_floor`    : Small ``\\sigma`` floor to stabilize degenerate cases.
- `also_hellinger` : If true, also returns Hellinger matrices.

# Returns
- `bc_dict` : `Dict[(pred, orig)]` ``\\Rightarrow`` 2D `Float64` matrix of ``\\mathrm{BC}`` in `[0,1]`.
- `H_dict`  :` Dict[...]` of Hellinger in `[0,1]` (if `also_hellinger` is `true`), else `nothing`.
"""
function build_bhattacharyya_dicts(
    new_dict::Dict{String, Array{Float64,2}},
    keys::Vector{Symbol},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64 = 1e-12,
    also_hellinger::Bool = false
) :: Tuple{Dict{Tuple{String,String}, Array{Float64,2}},
           Union{Nothing, Dict{Tuple{String,String}, Array{Float64,2}}}}

    bc_dict = Dict{Tuple{String,String}, Array{Float64,2}}()
    H_dict  = also_hellinger ? Dict{Tuple{String,String}, Array{Float64,2}}() : nothing

    for key in keys
        for tag in pred_tags
            pred = (key == :Deborah) ? string(tag) :
                   (String(key) * ":" * String(tag))
            orig = (key == :Deborah) ? ("Y:" * string(orig_tag)) :
                   (String(key) * ":Y:" * String(orig_tag))

            μ_pred  = new_dict[pred * ":avg"]
            σ_pred  = new_dict[pred * ":err"]   # interpreted as 1σ
            μ_orig  = new_dict[orig * ":avg"]
            σ_orig  = new_dict[orig * ":err"]   # interpreted as 1σ

            nrow, ncol = size(μ_pred)
            JobLoggerTools.assert_benji(
                size(σ_pred) == (nrow, ncol),
                "Shape mismatch for $(pred):err"
            )
            JobLoggerTools.assert_benji(
                size(μ_orig) == (nrow, ncol),
                "Shape mismatch for $(orig):avg"
            )
            JobLoggerTools.assert_benji(
                size(σ_orig) == (nrow, ncol),
                "Shape mismatch for $(orig):err"
            )

            bc_mat = Array{Float64}(undef, nrow, ncol)
            H_mat  = also_hellinger ? Array{Float64}(undef, nrow, ncol) : nothing

            @inbounds for ilb in eachindex(labels)
                for itr in eachindex(trains_ext)
                    bc = Comparison.bhattacharyya_coeff_normals(
                        μ_orig[ilb, itr], σ_orig[ilb, itr],
                        μ_pred[ilb, itr], σ_pred[ilb, itr],
                        σ_floor=σ_floor
                    )
                    bc_mat[ilb, itr] = bc
                    if also_hellinger
                        H_mat[ilb, itr] = Comparison.hellinger_from_bc(bc)
                    end
                end
            end

            bc_dict[(pred, orig)] = bc_mat
            if also_hellinger
                JobLoggerTools.assert_benji(
                    H_dict !== nothing, 
                    "H_dict must not be nothing"
                )
                H_dict[(pred, orig)] = H_mat
            end
        end
    end

    return bc_dict, H_dict
end

"""
    jsd_normals(
        μa::Float64, 
        σa::Float64,
        μb::Float64, 
        σb::Float64;
        σ_floor::Float64 = 1e-12,
        k::Float64 = 8.0,
        n::Int = 2001
    ) -> Float64

Compute the Jensen-Shannon divergence (``\\mathrm{JSD}``), base-2, between two univariate normal distributions  
`` \\mathcal{N}(\\mu_a,\\sigma_a^2) `` and `` \\mathcal{N}(\\mu_b,\\sigma_b^2) `` numerically.

# Range
- `` \\mathrm{JSD}_2 \\in [0,1] ``
- `` \\mathrm{JSD}_2 = 0 \\Leftrightarrow `` identical distributions
- `` \\mathrm{JSD}_2 = 1 \\Leftrightarrow `` maximally different (in the ``\\mathrm{JSD}`` sense, base-2)

# Definition (base-2)
For densities ``p,q`` and the mixture ``m=\\tfrac12(p+q)``,
```math
\\mathrm{JSD}_2(P\\|Q) \\;=\\;
\\frac12 \\int_{\\mathbb{R}} p(x)\\,\\log_2\\!\\frac{p(x)}{m(x)}\\,dx
\\;+\\;
\\frac12 \\int_{\\mathbb{R}} q(x)\\,\\log_2\\!\\frac{q(x)}{m(x)}\\,dx.
```

# Specialization to normals

Let ``p = \\mathcal{N}(\\mu_a,\\sigma_a^2)``, ``q=\\mathcal{N}(\\mu_b,\\sigma_b^2)``.
There is no simple closed-form for ``\\mathrm{JSD}`` between two general Gaussians, so this routine computes it by numerical quadrature over a finite window that covers both tails.

## Integration window

We integrate on

```math
[L, R] \\;=\\;
\\Big[\\min(\\mu_a - k\\,\\sigma_a^{\\ast},\\, \\mu_b - k\\,\\sigma_b^{\\ast}),\\;
      \\max(\\mu_a + k\\,\\sigma_a^{\\ast},\\, \\mu_b + k\\,\\sigma_b^{\\ast})\\Big],
```

where ``\\sigma^{\\ast} = \\max(\\sigma,\\sigma_{\\text{floor}})``.
The hyperparameter ``k`` controls tail coverage (default ``k=8``).

## Uniform grid approximation

Using `n` evenly spaced points ``x_1,\\dots,x_n`` on ``[L,R]`` with spacing ``\\Delta x``,

```math
\\widehat{\\mathrm{JSD}}_2
\\;=\\; \\Delta x \\sum_{i=1}^{n}
\\frac12\\Big(p(x_i)\\,\\log_2\\!\\frac{p(x_i)}{m(x_i)}
          + q(x_i)\\,\\log_2\\!\\frac{q(x_i)}{m(x_i)}\\Big),
\\qquad m(x_i)=\\tfrac12\\big(p(x_i)+q(x_i)\\big).
```

Here ``p(x)`` and ``q(x)`` are the normal p.d.f.'s:

```math
\\phi(x;\\mu,\\sigma)
=\\frac{1}{\\sqrt{2\\pi}\\,\\sigma}\\exp\\!\\Big(-\\tfrac12\\big(\\tfrac{x-\\mu}{\\sigma}\\big)^2\\Big),
\\quad
\\sigma \\leftarrow \\max(\\sigma, \\sigma_{\\text{floor}}).
```

# Numerical stability & safeguards

- Sigma floor: ``\\sigma \\leftarrow \\max(\\sigma,\\sigma_{\\text{floor}})`` prevents degeneracy as ``\\sigma\\to 0``.
- Window fallback: if ``R\\le L`` due to extreme inputs, a tiny symmetric window around ``\\tfrac{\\mu_a+\\mu_b}{2}`` is used.
- Clamping: the final result is clamped into ``[0,1]`` to absorb floating-point round-off.

# Parameters

* `μa::Float64, σa::Float64, μb::Float64, σb::Float64` — normal means and ``1\\sigma`` standard deviations.
* `σ_floor::Float64=1e-12` — lower bound for standard deviations.
* `k::Float64=8.0` — tail coverage in units of ``\\sigma``.
* `n::Int=2001` — number of grid points (increase for accuracy, at higher cost).

# Returns

* `Float64` — ``\\widehat{\\mathrm{JSD}}_2 \\in [0,1]``, base-2.

# Complexity

* Time: ``\\mathcal{O}(n)`` evaluations of pdf and logs.
* Memory: ``\\mathcal{O}(1)`` besides the grid iterator.

Jensen-Shannon divergence (``\\mathrm{JSD}``) between two univariate normal distributions
``N(\\mu_a, \\sigma_a^2)`` and ``N(\\mu_b, \\sigma_b^2)``, computed numerically with base-2 logarithms.

- Returns a value in ``[0, 1]``, where ``0`` means identical, ``1`` means maximally different.
- Uses a finite window ``[L, R]`` with `L = min(μa - kσa, μb - kσb)`, `R = max(μa + kσa, μb + kσb)`.
- Integrates on a uniform grid of `n` points. Larger `n` or `k` increases accuracy.

Notes
- `σ_floor` avoids degeneracy when ``\\sigma \\approx 0`` (interprets inputs as ``1\\sigma`` standard deviation).
- No dependencies on external packages.
"""
function jsd_normals(
    μa::Float64, 
    σa::Float64,
    μb::Float64, 
    σb::Float64;
    σ_floor::Float64 = 1e-12,
    k::Float64 = 8.0,
    n::Int = 2001
)::Float64
    # Safe sigmas
    σ1 = max(σa, σ_floor)
    σ2 = max(σb, σ_floor)

    # Normal PDF (no external deps)
    invsqrt2π = 1.0 / sqrt(2π)
    @inline function npdf(x::Float64, μ::Float64, σ::Float64)
        z = (x - μ) / σ
        return invsqrt2π / σ * exp(-0.5 * z*z)
    end

    # Finite integration window covering both tails
    L = min(μa - k*σ1, μb - k*σ2)
    R = max(μa + k*σ1, μb + k*σ2)
    # JobLoggerTools.assert_benji(
    #     R > L,
    #     "Integration bounds collapsed; check inputs."
    # )

    if !all(isfinite, (L,R,μa,σ1,μb,σ2))
        return 1.0 
    end

    if !(R > L)
        μm = (μa + μb) / 2
        δ  = max(8*σ_floor, eps(μm)*16, 1e-9)
        L  = μm - δ
        R  = μm + δ
    end

    # Uniform grid
    n = max(n, 101)                # keep it reasonable
    xs = range(L, R; length=n)
    dx = step(xs)

    acc = 0.0
    @inbounds for x in xs
        p = npdf(x, μa, σ1)
        q = npdf(x, μb, σ2)
        m = 0.5*(p + q)
        # For normals p,q>0 on R, so no zero-safe needed; still guard tiny m
        if m > 0
            acc += 0.5 * (p * (log(p/m)/log(2)) + q * (log(q/m)/log(2)))
        end
    end

    jsd = acc * dx
    # Clamp to [0,1] for numerical robustness
    return jsd < 0 ? 0.0 : (jsd > 1 ? 1.0 : jsd)
end

"""
    build_jsd_dicts(
        new_dict::Dict{String, Array{Float64,2}},
        keys::Vector{Symbol},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64 = 1e-12,
        k::Float64 = 8.0,
        n::Int = 1201
    ) -> Dict{Tuple{String,String}, Array{Float64,2}}

Construct Jensen-Shannon divergence (``\\mathrm{JSD}``, base-2) matrices for each
`(observable key, prediction tag)` pair against a fixed reference tag,
following the same naming scheme as [`build_overlap_and_error_dicts`](@ref).

Key construction matches your logic:
- If `key == :Deborah`
    - `pred = string(tag)`
    - `orig = "Y:" * string(orig_tag)`
- Else
    - `pred = string(key) * ":" * string(tag)`
    - `orig = string(key) * ":Y:" * string(orig_tag)`

For each grid point `(ilb, itr)`:
- `μ_pred = new_dict[pred * ":avg"][ilb, itr]`
- `σ_pred = new_dict[pred * ":err"][ilb, itr]`
- `μ_orig = new_dict[orig * ":avg"][ilb, itr]`
- `σ_orig = new_dict[orig * ":err"][ilb, itr]`
and [`JSD = jsd_normals(μ_pred, σ_pred, μ_orig, σ_orig; ...)`](@ref jsd_normals).

# Arguments
- `new_dict` : Must contain the 2D arrays for `":avg"` and `":err"` of both `pred`/`orig`.
- `keys`     : Observable keys (e.g., `[:Deborah, :cond, :susp, :skew, :kurt]`).
- `pred_tags`: Prediction tags (e.g., `[:Y_P1, :Y_P2]`).
- `orig_tag` : Reference tag (e.g., `:Y_BS`).
- `labels`   : Row axis (`LBP`-like index).
- `trains_ext`: Column axis (`TRP`-like index).
- `σ_floor`, `k`, `n`: Parameters forwarded to [`jsd_normals`](@ref).

# Returns
- `jsd_dict` : `Dict[(pred, orig)]` ``\\Rightarrow`` 2D `Float64` matrix of ``\\mathrm{JSD}`` in ``[0,1]`` (`0` best).
"""
function build_jsd_dicts(
    new_dict::Dict{String, Array{Float64,2}},
    keys::Vector{Symbol},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64 = 1e-12,
    k::Float64 = 8.0,
    n::Int = 1201
) :: Dict{Tuple{String,String}, Array{Float64,2}}

    jsd_dict = Dict{Tuple{String,String}, Array{Float64,2}}()

    for key in keys
        for tag in pred_tags
            pred = (key == :Deborah) ? string(tag) :
                   (String(key) * ":" * String(tag))
            orig = (key == :Deborah) ? ("Y:" * string(orig_tag)) :
                   (String(key) * ":Y:" * String(orig_tag))

            μ_pred  = new_dict[pred * ":avg"]
            σ_pred  = new_dict[pred * ":err"]   # interpreted as 1σ
            μ_orig  = new_dict[orig * ":avg"]
            σ_orig  = new_dict[orig * ":err"]   # interpreted as 1σ

            nrow, ncol = size(μ_pred)
            JobLoggerTools.assert_benji(
                size(σ_pred) == (nrow, ncol),
                "Shape mismatch for $(pred):err"
            )
            JobLoggerTools.assert_benji(
                size(μ_orig) == (nrow, ncol),
                "Shape mismatch for $(orig):avg"
            )
            JobLoggerTools.assert_benji(
                size(σ_orig) == (nrow, ncol),
                "Shape mismatch for $(orig):err"
            )

            jsd_mat = Array{Float64}(undef, nrow, ncol)

            @inbounds for ilb in eachindex(labels)
                for itr in eachindex(trains_ext)
                    jsd_mat[ilb, itr] = Comparison.jsd_normals(
                        μ_pred[ilb, itr], σ_pred[ilb, itr],
                        μ_orig[ilb, itr], σ_orig[ilb, itr];
                        σ_floor=σ_floor, k=k, n=n
                    )
                end
            end

            jsd_dict[(pred, orig)] = jsd_mat
        end
    end

    return jsd_dict
end

end