# ============================================================================
# src/RebekahMiriam/ComparisonRebekahMiriam.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ComparisonRebekahMiriam

import ..Rebekah.Comparison
import ..Sarah.JobLoggerTools

"""
    build_overlap_error_and_ovl_dicts(
        ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
        keys::Vector{Symbol},
        keywords::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64=1e-12
    ) -> Tuple{
        Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
        Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}},
        Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
    }

Construct overlap, error-ratio, and type-B distance dictionaries for all
observables and keywords in a multi-criterion setup.

This function is specific to the [`Deborah.Miriam`](@ref) analysis framework. It traverses
combinations of observable keys (`keys`), keyword criteria (`keywords`), and
prediction methods (`pred_tags`) to compare against a reference (`orig_tag`).
Each data array is extracted from `ext_dict`, which stores 2D matrices indexed
by 4-tuples of the form

    (observable_key, kind, tag, keyword)

where `kind` is either `:avg` or `:err`. Only entries that exist in `ext_dict`
are processed.

# Arguments
- `ext_dict`: Dictionary of 2D matrices for all observable combinations, keyed
  by 4-tuples.
- `keys`: Observable types (e.g., `:TrM1`, `:TrM2`, ...).
- `keywords`: Interpolation/selection criteria (e.g., `"kurt"`, `"skew"`, ...).
- `pred_tags`: Prediction method tags to evaluate (e.g., `:RWP1`, `:RWP2`).
- `orig_tag`: Tag used for reference data (typically `:RWBS`).
- `labels`: Vector indexing the `LBP` axis.
- `trains_ext`: Vector indexing the `TRP` axis.
- `σ_floor`: Small positive floor for uncertainty to avoid divide-by-zero in
  type-B distance.

# Returns
- `chk_dict[(key, pred_tag, keyword)] :: Array{Int,2}` → overlap codes
  (`0`/`1`/`2`) via [`Deborah.Rebekah.Comparison.check_overlap`](@ref).
- `err_dict[(key, pred_tag, keyword)] :: Array{Float64,2}` → error ratios
  `pred_err / orig_err` via [`Deborah.Rebekah.Comparison.err_ratio`](@ref).
- `ovl_dict[(key, pred_tag, keyword)] :: Array{Float64,2}` → type-B distances
  ``d \\equiv \\dfrac{|\\mu_{\\text{orig}} - \\mu_{\\text{pred}}|}{\\max(\\sigma_{\\text{orig}}, \\sigma_{\\text{floor}})}`` computed by [`Deborah.Rebekah.Comparison.check_overlap_type_b`](@ref).

# Notes
- `ovl_dict` is intentionally **asymmetric** (measured in units of the
  reference/original ``\\sigma``).
- The overlap code (`chk_dict`) is a coarse classifier; `ovl_dict` provides a
  graded “how far in `σ_orig`” measure that complements it.

# See also
- [`Deborah.Rebekah.Comparison.check_overlap`](@ref) — interval overlap
  classifier (`0`/`1`/`2`).
- [`Deborah.Rebekah.Comparison.check_overlap_type_b`](@ref) — asymmetric
  σ-normalized separation.
- [`Deborah.Rebekah.Comparison.bhattacharyya_coeff_normals`](@ref) — symmetric
  overlap proxy using both variances.
"""
function build_overlap_error_and_ovl_dicts(
    ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
    keys::Vector{Symbol},
    keywords::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64=1e-12
)::Tuple{
    Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
    Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}},
    Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
}
    chk_dict = Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}}()
    err_dict = Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}()
    ovl_dict = Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}()

    for key in keys, keyword in keywords, pred_tag in pred_tags
        k_pred_avg = (key, :avg, pred_tag, keyword)
        k_pred_err = (key, :err, pred_tag, keyword)
        k_orig_avg = (key, :avg, orig_tag, keyword)
        k_orig_err = (key, :err, orig_tag, keyword)

        if haskey(ext_dict, k_pred_avg) && haskey(ext_dict, k_pred_err) &&
           haskey(ext_dict, k_orig_avg) && haskey(ext_dict, k_orig_err)

            pred_avg = ext_dict[k_pred_avg]
            pred_err = ext_dict[k_pred_err]
            orig_avg = ext_dict[k_orig_avg]
            orig_err = ext_dict[k_orig_err]

            # overlap code (0/1/2)
            chk_mat = [Comparison.check_overlap(
                           pred_avg[ilb, itr], pred_err[ilb, itr],
                           orig_avg[ilb, itr], orig_err[ilb, itr]
                       )
                       for ilb in eachindex(labels), itr in eachindex(trains_ext)]

            # error ratio (pred / orig)
            err_mat = [Comparison.err_ratio(
                           pred_err[ilb, itr], orig_err[ilb, itr]
                       )
                       for ilb in eachindex(labels), itr in eachindex(trains_ext)]

            # type-B distance measured in σ_orig
            ovl_mat = [Comparison.check_overlap_type_b(
                           orig_avg[ilb, itr], orig_err[ilb, itr],  # (μa, σa)
                           pred_avg[ilb, itr];                      # μb
                           σ_floor=σ_floor
                       )
                       for ilb in eachindex(labels), itr in eachindex(trains_ext)]

            chk_dict[(key, pred_tag, keyword)] = chk_mat
            err_dict[(key, pred_tag, keyword)] = err_mat
            ovl_dict[(key, pred_tag, keyword)] = ovl_mat
        end
    end

    return chk_dict, err_dict, ovl_dict
end

"""
    build_overlap_and_error_dicts(
        ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
        keys::Vector{Symbol},
        keywords::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector
    ) -> Tuple{
        Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
        Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
    }

Construct overlap and error-ratio dictionaries for all observables and keywords in a
multi-criterion setup.

This is a **backward-compatible wrapper** that returns only the original two outputs
(overlap codes and error ratios). Internally it delegates to
[`build_overlap_error_and_ovl_dicts`](@ref) and discards the additional `ovl_dict`.
Use the 3-return variant if you also need the ``\\sigma``-normalized type-B distances.

# Arguments
(Identical to [`build_overlap_error_and_ovl_dicts`] except no `σ_floor` keyword.)

# Returns
- `chk_dict[(key, pred_tag, keyword)] :: Array{Int,2}`
  → overlap quality codes for each `(label, train)`.
- `err_dict[(key, pred_tag, keyword)] :: Array{Float64,2}`
  → error ratios for each `(label, train)`.

# Notes
- Existing call sites like
  `chk_dict, err_dict = build_overlap_and_error_dicts(...)`
  remain valid and unchanged.
- Prefer [`build_overlap_error_and_ovl_dicts`](@ref) for new code when you want the third output
  (`ovl_dict`) without altering older call sites.
"""
function build_overlap_and_error_dicts(
    ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
    keys::Vector{Symbol},
    keywords::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector
)::Tuple{
    Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
    Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
}
    chk_dict, err_dict, _ = build_overlap_error_and_ovl_dicts(
        ext_dict, keys, keywords, pred_tags, orig_tag, labels, trains_ext;
        σ_floor=1e-12
    )
    return chk_dict, err_dict
end

"""
    build_overlap_error_and_ovl_dicts_for_measurements(
        ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
        keys::Vector{Symbol},
        kappa_list::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64=1e-12
    ) -> Tuple{
        Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
        Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}},
        Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
    }

Build overlap, error-ratio, and type-B distance dictionaries from measurement summaries
for a single ensemble.

This variant uses `kappa_list` as the 4th-key dimension of `ext_dict`, i.e.,
data are indexed by 4-tuples

    (observable_key, kind, tag, kappa_str)

with `kind` ``\\in`` `{:avg, :err}`.

# Arguments
- `ext_dict`: Summary dictionary returned by
  [`Deborah.RebekahMiriam.SummaryLoaderRebekahMiriam.load_miriam_summary_for_measurement`](@ref).
- `keys`: List of observable keys (e.g., `[:trM1, :trM2, :trM3, :trM4]` or capitalized variants).
- `kappa_list`: ``\\kappa`` values as strings (dictionary dimension).
- `pred_tags`: Prediction tags (e.g., `[:T_P1, :T_P2]`).
- `orig_tag`: Original/reference tag (e.g., `:T_BS`).
- `labels`: `LBP` ratios index axis.
- `trains_ext`: `TRP` ratios index axis.
- `σ_floor`: Small positive floor for uncertainty to avoid divide-by-zero in type-B distance.

# Returns
- `chk_dict[(key, pred_tag, kappa_str)] :: Array{Int,2}`
  → overlap codes (`0`/`1`/`2`) per `(label, train)`.
- `err_dict[(key, pred_tag, kappa_str)] :: Array{Float64,2}`
  → error ratios `pred_err / orig_err` per `(label, train)`.
- `ovl_dict[(key, pred_tag, kappa_str)] :: Array{Float64,2}`
  → type-B distances ``d \\equiv \\dfrac{|\\mu_{\\text{orig}} - \\mu_{\\text{pred}}|}{\\max(\\sigma_{\\text{orig}}, \\sigma_{\\text{floor}})}`` per `(label, train)`.

# Notes
- Asymmetric ``\\sigma`` scaling (units of the reference/original) by design.
- Complements the coarse overlap code with a continuous ``\\sigma``-distance measure.

# See also
- [`build_overlap_error_and_ovl_dicts`](@ref) — keyword-based 4th dimension.
- [`build_overlap_and_error_dicts_for_measurements`](@ref) — 2-return wrapper for backward compatibility.
"""
function build_overlap_error_and_ovl_dicts_for_measurements(
    ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
    keys::Vector{Symbol},
    kappa_list::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64=1e-12
)::Tuple{
    Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
    Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}},
    Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
}
    chk_dict = Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}}()
    err_dict = Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}()
    ovl_dict = Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}()

    for key in keys, kappa_str in kappa_list, pred_tag in pred_tags
        k_pred_avg = (key, :avg, pred_tag, kappa_str)
        k_pred_err = (key, :err, pred_tag, kappa_str)
        k_orig_avg = (key, :avg, orig_tag, kappa_str)
        k_orig_err = (key, :err, orig_tag, kappa_str)

        if haskey(ext_dict, k_pred_avg) && haskey(ext_dict, k_pred_err) &&
           haskey(ext_dict, k_orig_avg) && haskey(ext_dict, k_orig_err)

            pred_avg = ext_dict[k_pred_avg]
            pred_err = ext_dict[k_pred_err]
            orig_avg = ext_dict[k_orig_avg]
            orig_err = ext_dict[k_orig_err]

            chk_mat = [Comparison.check_overlap(
                           pred_avg[ilb, itr], pred_err[ilb, itr],
                           orig_avg[ilb, itr], orig_err[ilb, itr]
                       )
                       for ilb in eachindex(labels), itr in eachindex(trains_ext)]

            err_mat = [Comparison.err_ratio(
                           pred_err[ilb, itr], orig_err[ilb, itr]
                       )
                       for ilb in eachindex(labels), itr in eachindex(trains_ext)]

            ovl_mat = [Comparison.check_overlap_type_b(
                           orig_avg[ilb, itr], orig_err[ilb, itr],
                           pred_avg[ilb, itr]; σ_floor=σ_floor
                       )
                       for ilb in eachindex(labels), itr in eachindex(trains_ext)]

            chk_dict[(key, pred_tag, kappa_str)] = chk_mat
            err_dict[(key, pred_tag, kappa_str)] = err_mat
            ovl_dict[(key, pred_tag, kappa_str)] = ovl_mat
        end
    end
    return chk_dict, err_dict, ovl_dict
end

"""
    build_overlap_and_error_dicts_for_measurements(
        ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
        keys::Vector{Symbol},
        kappa_list::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector
    ) -> Tuple{
        Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
        Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
    }

Build overlap-check and error-ratio dictionaries from measurement summaries for a
single ensemble.

This is a **backward-compatible wrapper** that returns only the original two outputs.
Internally it calls [`build_overlap_error_and_ovl_dicts_for_measurements`](@ref) and
discards the additional `ovl_dict`. Use the 3-return variant if you also need
the ``\\sigma``-normalized type-B distances.

# Arguments
(Identical to [`build_overlap_error_and_ovl_dicts_for_measurements`](@ref) except no `σ_floor` keyword.)

# Returns
- `chk_dict[(key, pred_tag, kappa_str)] :: Array{Int,2}`
  → overlap quality codes for each `(label, train)`.
- `err_dict[(key, pred_tag, kappa_str)] :: Array{Float64,2}`
  → error ratios for each `(label, train)`.

# Notes
- Existing call sites like
  `chk_dict, err_dict = build_overlap_and_error_dicts_for_measurements(...)`
  continue to work unchanged.
- Prefer the 3-return variant
  [`build_overlap_error_and_ovl_dicts_for_measurements`](@ref) for new code that consumes
  `ovl_dict`.
"""
function build_overlap_and_error_dicts_for_measurements(
    ext_dict::Dict{Tuple{Symbol, Symbol, Symbol, String}, Array{Float64,2}},
    keys::Vector{Symbol},
    kappa_list::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector
)::Tuple{
    Dict{Tuple{Symbol, Symbol, String}, Array{Int,2}},
    Dict{Tuple{Symbol, Symbol, String}, Array{Float64,2}}
}
    chk_dict, err_dict, _ = build_overlap_error_and_ovl_dicts_for_measurements(
        ext_dict, keys, kappa_list, pred_tags, orig_tag, labels, trains_ext;
        σ_floor=1e-12
    )
    return chk_dict, err_dict
end

"""
    build_bhattacharyya_dicts(
        ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
        keys::Vector{Symbol},
        keywords::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64 = 1e-12,
        also_hellinger::Bool = false
    ) -> Tuple{
        Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}},
        Union{Nothing, Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}}
    }

Construct Bhattacharyya-coefficient (``\\mathrm{BC}``) matrices — and optionally Hellinger-distance matrices — for all `(key, keyword, pred_tag)` triples against a fixed `orig_tag`.  
Keying and grid traversal mirror [`build_overlap_and_error_dicts`](@ref).

# What it builds
For each `(key, keyword, pred_tag)`:
- `bc_dict[(key, pred_tag, keyword)] :: Array{Float64,2}` — Bhattacharyya coefficient matrix over the `(labels` ``\\times`` `trains_ext)` grid.
- `H_dict[(key, pred_tag, keyword)] :: Array{Float64,2}` (optional) — Hellinger distance matrix over the same grid.

# Inputs
- `ext_dict::Dict{(Symbol,Symbol,Symbol,String) => Array{Float64,2}}` holding 2D arrays for keys of the form `(key, kind, tag, keyword)`, where `kind` ``\\in`` `{:avg, :err}`, `tag` ``\\in`` `pred_tags` ``\\cup`` `{orig_tag}`.
  - `(key, :avg, pred_tag, keyword)` → ``\\mu_{\\text{pred}}``
  - `(key, :err, pred_tag, keyword)` → ``\\sigma_{\\text{pred}}``
  - `(key, :avg,  orig_tag, keyword)` → ``\\mu_{\\text{orig}}``
  - `(key, :err,  orig_tag, keyword)` → ``\\sigma_{\\text{orig}}``
- `keys, keywords, pred_tags, orig_tag, labels, trains_ext`
- `σ_floor::Float64=1e-12` — floor for standard deviations to ensure numerical stability.
- `also_hellinger::Bool=false` — if `true`, also compute Hellinger matrices.

# Output structure
- `bc_dict :: Dict{(key::Symbol, pred_tag::Symbol, keyword::String) => Array{Float64,2}}`
- `H_dict :: Union{Nothing, Dict{(key,pred_tag,keyword)=>Array{Float64,2}}}`

Rows correspond to `labels` indices; columns correspond to `trains_ext` indices.

# See also
- [`Deborah.Rebekah.Comparison.bhattacharyya_coeff_normals`](@ref)
- [`Deborah.Rebekah.Comparison.hellinger_from_bc`](@ref)
"""
function build_bhattacharyya_dicts(
    ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
    keys::Vector{Symbol},
    keywords::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64 = 1e-12,
    also_hellinger::Bool = false
) :: Tuple{
    Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}},
    Union{Nothing, Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}}
}
    bc_dict = Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}()
    H_dict  = also_hellinger ? Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}() : nothing

    for key in keys, keyword in keywords, pred_tag in pred_tags
        k_pred_avg = (key, :avg, pred_tag, keyword)
        k_pred_err = (key, :err, pred_tag, keyword)
        k_orig_avg = (key, :avg, orig_tag, keyword)
        k_orig_err = (key, :err, orig_tag, keyword)

        if haskey(ext_dict, k_pred_avg) && haskey(ext_dict, k_pred_err) &&
           haskey(ext_dict, k_orig_avg) && haskey(ext_dict, k_orig_err)

            μ_pred = ext_dict[k_pred_avg]
            σ_pred = ext_dict[k_pred_err]
            μ_orig = ext_dict[k_orig_avg]
            σ_orig = ext_dict[k_orig_err]

            nrow, ncol = size(μ_pred)
            JobLoggerTools.assert_benji(
                size(σ_pred) == (nrow, ncol),
                "Shape mismatch for $(k_pred_err)"
            )
            JobLoggerTools.assert_benji(
                size(μ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_avg)"
            )
            JobLoggerTools.assert_benji(
                size(σ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_err)"
            )

            bc_mat = Array{Float64}(undef, nrow, ncol)
            H_mat  = also_hellinger ? Array{Float64}(undef, nrow, ncol) : nothing

            @inbounds for ilb in eachindex(labels)
                for itr in eachindex(trains_ext)
                    bc = Comparison.bhattacharyya_coeff_normals(
                        μ_orig[ilb, itr], σ_orig[ilb, itr],
                        μ_pred[ilb, itr], σ_pred[ilb, itr];
                        σ_floor=σ_floor
                    )
                    bc_mat[ilb, itr] = bc
                    if also_hellinger
                        H_mat[ilb, itr] = Comparison.hellinger_from_bc(bc)
                    end
                end
            end

            bc_dict[(key, pred_tag, keyword)] = bc_mat
            if also_hellinger
                JobLoggerTools.assert_benji(
                    H_dict !== nothing, 
                    "H_dict must not be nothing"
                )
                H_dict[(key, pred_tag, keyword)] = H_mat
            end
        end
    end

    return bc_dict, H_dict
end

"""
    build_bhattacharyya_dicts_for_measurements(
        ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
        keys::Vector{Symbol},
        kappa_list::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64 = 1e-12,
        also_hellinger::Bool = false
    ) -> Tuple{
        Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}},
        Union{Nothing, Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}}
    }

Build ``\\mathrm{BC}`` (and optionally Hellinger) matrices for measurement summaries for single ensemble keyed by `(key, kind, tag, kappa_str)`.  
Keying/loop structure is analogous to [`build_bhattacharyya_dicts`](@ref), but uses `kappa_str` in place of `keyword`.

# What it builds
- `bc_dict[(key, pred_tag, kappa_str)] :: Array{Float64,2}`
- `H_dict[(key, pred_tag, kappa_str)] :: Array{Float64,2}` *(optional)*

# Inputs / Outputs / Formulas
Same as [`build_bhattacharyya_dicts`](@ref), with internal keys:
- `(key, :avg, pred_tag, kappa_str)`, `(key, :err, pred_tag, kappa_str)`,
- `(key, :avg,  orig_tag, kappa_str)`, `(key, :err,  orig_tag, kappa_str)`.

`σ_floor` clipping and the `BC`/`H` formulas are identical.

# See also
- [`Deborah.Rebekah.Comparison.bhattacharyya_coeff_normals`](@ref)
- [`Deborah.Rebekah.Comparison.hellinger_from_bc`](@ref)
"""
function build_bhattacharyya_dicts_for_measurements(
    ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
    keys::Vector{Symbol},
    kappa_list::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64 = 1e-12,
    also_hellinger::Bool = false
) :: Tuple{
    Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}},
    Union{Nothing, Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}}
}
    bc_dict = Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}()
    H_dict  = also_hellinger ? Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}() : nothing

    for key in keys, kappa_str in kappa_list, pred_tag in pred_tags
        k_pred_avg = (key, :avg, pred_tag, kappa_str)
        k_pred_err = (key, :err, pred_tag, kappa_str)
        k_orig_avg = (key, :avg, orig_tag, kappa_str)
        k_orig_err = (key, :err, orig_tag, kappa_str)

        if haskey(ext_dict, k_pred_avg) && haskey(ext_dict, k_pred_err) &&
           haskey(ext_dict, k_orig_avg) && haskey(ext_dict, k_orig_err)

            μ_pred = ext_dict[k_pred_avg]
            σ_pred = ext_dict[k_pred_err]
            μ_orig = ext_dict[k_orig_avg]
            σ_orig = ext_dict[k_orig_err]

            nrow, ncol = size(μ_pred)
            JobLoggerTools.assert_benji(
                size(σ_pred) == (nrow, ncol),
                "Shape mismatch for $(k_pred_err):err"
            )
            JobLoggerTools.assert_benji(
                size(μ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_avg):avg"
            )
            JobLoggerTools.assert_benji(
                size(σ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_err):err"
            )


            bc_mat = Array{Float64}(undef, nrow, ncol)
            H_mat  = also_hellinger ? Array{Float64}(undef, nrow, ncol) : nothing

            @inbounds for ilb in eachindex(labels)
                for itr in eachindex(trains_ext)
                    bc = Comparison.bhattacharyya_coeff_normals(
                        μ_pred[ilb, itr], σ_pred[ilb, itr],
                        μ_orig[ilb, itr], σ_orig[ilb, itr];
                        σ_floor=σ_floor
                    )
                    bc_mat[ilb, itr] = bc
                    if also_hellinger
                        H_mat[ilb, itr] = Comparison.hellinger_from_bc(bc)
                    end
                end
            end

            bc_dict[(key, pred_tag, kappa_str)] = bc_mat
            if also_hellinger
                JobLoggerTools.assert_benji(
                    H_dict !== nothing, 
                    "H_dict must not be nothing"
                )
                H_dict[(key, pred_tag, kappa_str)] = H_mat
            end
        end
    end

    return bc_dict, H_dict
end

"""
    build_jsd_dicts(
        ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
        keys::Vector{Symbol},
        keywords::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64 = 1e-12,
        k::Float64 = 8.0,
        n::Int = 1201
    ) -> Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}

Construct Jensen-Shannon divergence (``\\mathrm{JSD}``, base-2; range ``[0,1]``) matrices for all `(key, keyword, pred_tag)` triples against a fixed `orig_tag`.  
Keying/layout mirror [`build_bhattacharyya_dicts`](@ref).

# What it builds
- `jsd_dict[(key, pred_tag, keyword)] :: Array{Float64,2}` — ``\\mathrm{JSD}`` (base-2) over the `labels` ``\\times`` `trains_ext` grid.

# Inputs
- `ext_dict` with the same 4-tuple key scheme `(key, :avg/err, tag, keyword)` for both `pred_tag` and `orig_tag`.
- `σ_floor::Float64=1e-12` — floor for standard deviations.
- `k::Float64=8.0`, `n::Int=1201` — numerical integration window/resolution parameters.

# See also
- [`Deborah.Rebekah.Comparison.jsd_normals`](@ref)
"""
function build_jsd_dicts(
    ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
    keys::Vector{Symbol},
    keywords::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64 = 1e-12,
    k::Float64 = 8.0,
    n::Int = 1201
) :: Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}

    jsd_dict = Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}()

    for key in keys, keyword in keywords, pred_tag in pred_tags
        k_pred_avg = (key, :avg, pred_tag, keyword)
        k_pred_err = (key, :err, pred_tag, keyword)
        k_orig_avg = (key, :avg, orig_tag, keyword)
        k_orig_err = (key, :err, orig_tag, keyword)

        if haskey(ext_dict, k_pred_avg) && haskey(ext_dict, k_pred_err) &&
           haskey(ext_dict, k_orig_avg) && haskey(ext_dict, k_orig_err)

            μ_pred = ext_dict[k_pred_avg]
            σ_pred = ext_dict[k_pred_err]
            μ_orig = ext_dict[k_orig_avg]
            σ_orig = ext_dict[k_orig_err]

            nrow, ncol = size(μ_pred)
            JobLoggerTools.assert_benji(
                size(σ_pred) == (nrow, ncol),
                "Shape mismatch for $(k_pred_err):err"
            )
            JobLoggerTools.assert_benji(
                size(μ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_avg):avg"
            )
            JobLoggerTools.assert_benji(
                size(σ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_err):err"
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

            jsd_dict[(key, pred_tag, keyword)] = jsd_mat
        end
    end

    return jsd_dict
end

"""
    build_jsd_dicts_for_measurements(
        ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
        keys::Vector{Symbol},
        kappa_list::Vector{String},
        pred_tags::Vector{Symbol},
        orig_tag::Symbol,
        labels::Vector,
        trains_ext::Vector;
        σ_floor::Float64 = 1e-12,
        k::Float64 = 8.0,
        n::Int = 1201
    ) -> Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}

Build base-2 ``\\mathrm{JSD}`` matrices for measurement summaries for single ensemble keyed by `(key, kind, tag, kappa_str)`.  
Keying and traversal mirror [`build_bhattacharyya_dicts_for_measurements`](@ref).

# What it builds
- `jsd_dict[(key, pred_tag, kappa_str)] :: Array{Float64,2}`

# Inputs / Definition / Grid traversal
Same as [`build_jsd_dicts`](@ref), replacing `keyword` with `kappa_str` and using internal keys:
- `(key, :avg/err, pred_tag, kappa_str)`, `(key, :avg/err, orig_tag, kappa_str)`.

The ``\\mathrm{JSD}`` definition, `σ_floor` handling, and numerical parameters `k`, `n` are identical.

# See also
- [`Deborah.Rebekah.Comparison.jsd_normals`](@ref)
"""
function build_jsd_dicts_for_measurements(
    ext_dict::Dict{Tuple{Symbol,Symbol,Symbol,String}, Array{Float64,2}},
    keys::Vector{Symbol},
    kappa_list::Vector{String},
    pred_tags::Vector{Symbol},
    orig_tag::Symbol,
    labels::Vector,
    trains_ext::Vector;
    σ_floor::Float64 = 1e-12,
    k::Float64 = 8.0,
    n::Int = 1201
) :: Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}

    jsd_dict = Dict{Tuple{Symbol,Symbol,String}, Array{Float64,2}}()

    for key in keys, kappa_str in kappa_list, pred_tag in pred_tags
        k_pred_avg = (key, :avg, pred_tag, kappa_str)
        k_pred_err = (key, :err, pred_tag, kappa_str)
        k_orig_avg = (key, :avg, orig_tag, kappa_str)
        k_orig_err = (key, :err, orig_tag, kappa_str)

        if haskey(ext_dict, k_pred_avg) && haskey(ext_dict, k_pred_err) &&
           haskey(ext_dict, k_orig_avg) && haskey(ext_dict, k_orig_err)

            μ_pred = ext_dict[k_pred_avg]
            σ_pred = ext_dict[k_pred_err]
            μ_orig = ext_dict[k_orig_avg]
            σ_orig = ext_dict[k_orig_err]

            nrow, ncol = size(μ_pred)
            JobLoggerTools.assert_benji(
                size(σ_pred) == (nrow, ncol),
                "Shape mismatch for $(k_pred_err):err"
            )
            JobLoggerTools.assert_benji(
                size(μ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_avg):avg"
            )
            JobLoggerTools.assert_benji(
                size(σ_orig) == (nrow, ncol),
                "Shape mismatch for $(k_orig_err):err"
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

            jsd_dict[(key, pred_tag, kappa_str)] = jsd_mat
        end
    end

    return jsd_dict
end

end  # module ComparisonRebekahMiriam