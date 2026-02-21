# =============================================================================
# src/Miriam/CumulantsBundleUtils.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# =============================================================================

module CumulantsBundleUtils

import ..Sarah.JobLoggerTools
import ..Ensemble
import ..ReweightingBundle

"""
    split_Q_full(
        Q_bundle_ORG::Union{Nothing, NTuple{5, Vector{T}}}, 
        Q_bundle::Vector,
        bundle::Union{Ensemble.EnsembleArrayBundle{T}, ReweightingBundle.ReweightingSolverBundle},
        ens_idx::Union{Nothing, Int},
        jobid::Union{Nothing, String}=nothing;
        has_weight::Bool=false
    ) where T -> NTuple{7, Vector{<:Tuple}}

Split Q-observable rows into labeled subsets by using `source_tags` and
`secondary_tags` attached to the ensemble bundles. Supports both raw-trace
mode ([`Deborah.Miriam.Ensemble.EnsembleArrayBundle`](@ref)) and reweighting mode ([`Deborah.Miriam.ReweightingBundle.ReweightingSolverBundle`](@ref)).

# Inputs
- `Q_bundle_ORG`:
  Optional original tuple `(Q1_ORG, Q2_ORG, Q3_ORG, Q4_ORG, w_ORG)`.
  If provided, `Q_Y_ORG` is built from these arrays (including `w_ORG`).
  If `nothing`, `Q_Y_ORG` is built from `Q_bundle[1]` (weight omitted).
- `Q_bundle::Vector`:
  A vector of tuples holding `(Q1, Q2, Q3, Q4)` and, when `has_weight=true`,
  an optional 5th array `w`. The four entries are aligned with the bundle
  variants as follows:
  - `Q_bundle[1]` ⇔ `FULL-LBOG-ULOG` (original; used for `Q_Y_ORG` and `Y_ul`)
  - `Q_bundle[2]` ⇔ `FULL-LBOG-ULML` (`Y_UL`-replaced; used for `YP_ul`)
  - `Q_bundle[3]` ⇔ `LABL-TROG-BCOG` (`Y_LB`-only, `Y_BC` original; used for `Y_tr`, `Y_bc`, and `Y_lb`)
  - `Q_bundle[4]` ⇔ `LABL-TROG-BCML` (`Y_LB`-only, `Y_BC` replaced; used for `YP_bc`)
- `bundle`:
  The corresponding bundle providing `conf_nums`, `source_tags` and `secondary_tags`.
  In reweighting mode (`has_weight=true`), these are read from `bundle.solvers[k].ens.data`;
  in raw-trace mode they are read from `bundle.arrays[k].data[ens_idx]`.
- `ens_idx`:
  Index for selecting an ensemble entry in raw-trace mode. Ignored in reweighting mode.
- `jobid`:
  Optional job ID used for logging context.
- `has_weight`:
  If `true`, each output tuple includes a weight component.

# Tag semantics
- `source_tags`: `0 → Y_tr`, `1 → Y_bc/YP_bc`, `2 → Y_ul/YP_ul`
- `secondary_tags`: `0 → Y_lb`, `1 → Y_ul`

## Invariants enforced
- In `Q_bundle[2]` (`UL`-replaced stream): rows with `source_tags==2` must have `secondary_tags==1`.
- In `LB`-only streams (`Q_bundle[3]`, `Q_bundle[4]`): any row with `secondary_tags==0` must not have `source_tags==2`.

# Returns
Each element is a vector of tuples:
- when `has_weight=false`: `(conf, Q1, Q2, Q3, Q4)`
- when `has_weight=true` : `(conf, Q1, Q2, Q3, Q4, w)`

Return order:
1. `Q_Y_tr`   — rows with `source_tags==0` from **`Q_bundle[3]`**
2. `Q_Y_bc`   — rows with `source_tags==1` from **`Q_bundle[3]`**
3. `Q_YP_bc`  — rows with `source_tags==1` from **`Q_bundle[4]`**
4. `Q_Y_ul`   — rows with `source_tags==2` from **`Q_bundle[1]`**
5. `Q_YP_ul`  — rows with `source_tags==2` from **`Q_bundle[2]`**
6. `Q_Y_lb`   — rows with `secondary_tags==0` from **`Q_bundle[3]`**
7. `Q_Y_ORG`  — original stream aligned to `conf_nums`
                (uses `Q_bundle_ORG` if provided; otherwise from `Q_bundle[1]`)

# Notes
- `Y_LB` now comes from `Q_bundle[3]` (`Y_LB`-only, `Y_BC` original). `Q_Y_lb` is a convenience
  union of all rows marked as `secondary_tags==0` in that stream, so elements in
  `Q_Y_tr` and `Q_Y_bc` will also appear in `Q_Y_lb` by design.
- Length consistency is checked between (`Y_bc`, `YP_bc`) and (`Y_ul`, `YP_ul`).
  If a predicted subset is empty (e.g., `TRP=0`), the check is skipped with a warning.
"""
function split_Q_full(
    Q_bundle_ORG::Union{Nothing, NTuple{5, Vector{T}}}, 
    Q_bundle::Vector,
    bundle::Union{Ensemble.EnsembleArrayBundle{T}, ReweightingBundle.ReweightingSolverBundle},
    ens_idx::Union{Nothing, Int},
    jobid::Union{Nothing, String} = nothing;
    has_weight::Bool = false
) where T

    # Define tuple type according to weight presence
    TupleType = has_weight ? Tuple{Int, Float64, Float64, Float64, Float64, Float64} : Tuple{Int, Float64, Float64, Float64, Float64}

    Q_Y_ORG = Vector{TupleType}()
    Q_Y_tr  = Vector{TupleType}()
    Q_Y_bc  = Vector{TupleType}()
    Q_YP_bc = Vector{TupleType}()
    Q_Y_ul  = Vector{TupleType}()
    Q_YP_ul = Vector{TupleType}()
    Q_Y_lb  = Vector{TupleType}()

    # Utility: extract components by index
    get = has_weight ?
        (Q1, Q2, Q3, Q4, w, idxs) -> (Q1[idxs], Q2[idxs], Q3[idxs], Q4[idxs], w[idxs]) :
        (Q1, Q2, Q3, Q4, _, idxs) -> (Q1[idxs], Q2[idxs], Q3[idxs], Q4[idxs])

    # === Extract from main array or solver[1]
    begin
        Q1, Q2, Q3, Q4 = Q_bundle[1][1:4]
        w = has_weight ? Q_bundle[1][5] : zeros(Float64, length(Q1))  # dummy w if not used

        if has_weight
            solver    = bundle.solvers[1]
            conf_nums = reduce(vcat, [e.conf_nums    for e in solver.ens.data])
            tags      = reduce(vcat, [e.source_tags  for e in solver.ens.data])
        else
            ens_arr   = bundle.arrays[1]
            conf_nums = ens_arr.data[ens_idx].conf_nums
            tags      = ens_arr.data[ens_idx].source_tags
        end

        JobLoggerTools.assert_benji(
            length(Q1) == length(tags) == length(conf_nums) == length(w),
            "length mismatch among Q/tags/conf/w", jobid
        )

        # --- ORG dump
        if isnothing(Q_bundle_ORG)
            for i in eachindex(conf_nums)
                push!(Q_Y_ORG, (conf_nums[i], Q1[i], Q2[i], Q3[i], Q4[i]))
            end
        else
            Q1_ORG, Q2_ORG, Q3_ORG, Q4_ORG, w_ORG = Q_bundle_ORG
            for i in eachindex(conf_nums)
                push!(Q_Y_ORG, (conf_nums[i], Q1_ORG[i], Q2_ORG[i], Q3_ORG[i], Q4_ORG[i], w_ORG[i]))
            end
        end

        ul_idx = findall(==(2), tags)
        conf_ul = conf_nums[ul_idx]
        data_ul = get(Q1, Q2, Q3, Q4, w, ul_idx)

        for i in eachindex(conf_ul)
            push!(Q_Y_ul, has_weight ?
                (conf_ul[i], data_ul[1][i], data_ul[2][i], data_ul[3][i], data_ul[4][i], data_ul[5][i]) :
                (conf_ul[i], data_ul[1][i], data_ul[2][i], data_ul[3][i], data_ul[4][i])
            )
        end
    end

    # === Extract from bundle[2] (for YP_ul)
    begin
        Q1, Q2, Q3, Q4 = Q_bundle[2][1:4]
        w = has_weight ? Q_bundle[2][5] : zeros(Float64, length(Q1))

        if has_weight
            solver    = bundle.solvers[2]
            conf_nums = reduce(vcat, [e.conf_nums       for e in solver.ens.data])
            tags      = reduce(vcat, [e.source_tags     for e in solver.ens.data])
            sec_tags  = reduce(vcat, [e.secondary_tags  for e in solver.ens.data])
        else
            ens_arr   = bundle.arrays[2]
            conf_nums = ens_arr.data[ens_idx].conf_nums
            tags      = ens_arr.data[ens_idx].source_tags
            sec_tags  = ens_arr.data[ens_idx].secondary_tags
        end

        bad_lb = findall(i -> sec_tags[i] == 0 && tags[i] == 2, eachindex(tags))
        bad_ul = findall(i -> sec_tags[i] == 1 && tags[i] != 2, eachindex(tags))
        JobLoggerTools.assert_benji(isempty(bad_lb) && isempty(bad_ul),
            "Inconsistent tag/secondary mapping in bundle[2]: bad_lb=$(bad_lb), bad_ul=$(bad_ul)", jobid)

        ul_idx = findall(==(2), tags)
        JobLoggerTools.assert_benji(all(sec_tags[ul_idx] .== 1), "YP_ul rows not all in secondary Y_ul", jobid)
        conf_ul = conf_nums[ul_idx]
        data_ul = get(Q1, Q2, Q3, Q4, w, ul_idx)

        for i in eachindex(conf_ul)
            push!(Q_YP_ul, has_weight ?
                (conf_ul[i], data_ul[1][i], data_ul[2][i], data_ul[3][i], data_ul[4][i], data_ul[5][i]) :
                (conf_ul[i], data_ul[1][i], data_ul[2][i], data_ul[3][i], data_ul[4][i])
            )
        end

        # # --- split by secondary tag (LB vs YP_UL)
        # lb_idx = findall(==(0), sec_tags)
        # conf_lb = conf_nums[lb_idx]
        # data_lb = get(Q1, Q2, Q3, Q4, w, lb_idx)
        # for i in eachindex(conf_lb)
        #     push!(Q_Y_lb, has_weight ?
        #         (conf_lb[i], data_lb[1][i], data_lb[2][i], data_lb[3][i], data_lb[4][i], data_lb[5][i]) :
        #         (conf_lb[i], data_lb[1][i], data_lb[2][i], data_lb[3][i], data_lb[4][i])
        #     )
        # end        
    end

    # === Extract from bundle[3] (for Y_lb)
    begin
        Q1, Q2, Q3, Q4 = Q_bundle[3][1:4]
        w = has_weight ? Q_bundle[3][5] : zeros(Float64, length(Q1))

        if has_weight
            solver    = bundle.solvers[3]
            conf_nums = reduce(vcat, [e.conf_nums       for e in solver.ens.data])
            tags      = reduce(vcat, [e.source_tags     for e in solver.ens.data])
            sec_tags  = reduce(vcat, [e.secondary_tags  for e in solver.ens.data])
        else
            ens_arr   = bundle.arrays[3]
            conf_nums = ens_arr.data[ens_idx].conf_nums
            tags      = ens_arr.data[ens_idx].source_tags
            sec_tags  = ens_arr.data[ens_idx].secondary_tags
        end

        bad_lb = findall(i -> sec_tags[i] == 0 && tags[i] == 2, eachindex(tags))
        JobLoggerTools.assert_benji(isempty(bad_lb),
            "Inconsistent tag/secondary mapping in bundle[3]: bad_lb=$(bad_lb)", jobid)

        tr_idx = findall(==(0), tags)
        bc_idx = findall(==(1), tags) 
        JobLoggerTools.assert_benji(all(sec_tags[tr_idx] .== 0), "Y_tr rows not all in secondary Y_lb", jobid)
        JobLoggerTools.assert_benji(all(sec_tags[bc_idx] .== 0), "Y_bc rows not all in secondary Y_lb", jobid)
        conf_tr, conf_bc = conf_nums[tr_idx], conf_nums[bc_idx]
        data_tr = get(Q1, Q2, Q3, Q4, w, tr_idx)
        data_bc = get(Q1, Q2, Q3, Q4, w, bc_idx)

        for i in eachindex(conf_tr)
            push!(Q_Y_tr, has_weight ?
                (conf_tr[i], data_tr[1][i], data_tr[2][i], data_tr[3][i], data_tr[4][i], data_tr[5][i]) :
                (conf_tr[i], data_tr[1][i], data_tr[2][i], data_tr[3][i], data_tr[4][i])
            )
        end
        for i in eachindex(conf_bc)
            push!(Q_Y_bc, has_weight ?
                (conf_bc[i], data_bc[1][i], data_bc[2][i], data_bc[3][i], data_bc[4][i], data_bc[5][i]) :
                (conf_bc[i], data_bc[1][i], data_bc[2][i], data_bc[3][i], data_bc[4][i])
            )
        end

        # --- split by secondary tag (LB vs YP_UL)
        lb_idx = findall(==(0), sec_tags)
        conf_lb = conf_nums[lb_idx]
        data_lb = get(Q1, Q2, Q3, Q4, w, lb_idx)
        for i in eachindex(conf_lb)
            push!(Q_Y_lb, has_weight ?
                (conf_lb[i], data_lb[1][i], data_lb[2][i], data_lb[3][i], data_lb[4][i], data_lb[5][i]) :
                (conf_lb[i], data_lb[1][i], data_lb[2][i], data_lb[3][i], data_lb[4][i])
            )
        end
    end

    # === Extract from bundle[4] (for YP_bc)
    begin
        Q1, Q2, Q3, Q4 = Q_bundle[4][1:4]
        w = has_weight ? Q_bundle[4][5] : zeros(Float64, length(Q1))

        if has_weight
            solver    = bundle.solvers[4]
            conf_nums = reduce(vcat, [e.conf_nums       for e in solver.ens.data])
            tags      = reduce(vcat, [e.source_tags     for e in solver.ens.data])
            sec_tags  = reduce(vcat, [e.secondary_tags  for e in solver.ens.data])
        else
            ens_arr   = bundle.arrays[4]
            conf_nums = ens_arr.data[ens_idx].conf_nums
            tags      = ens_arr.data[ens_idx].source_tags
            sec_tags  = ens_arr.data[ens_idx].secondary_tags
        end

        bad_lb = findall(i -> sec_tags[i] == 0 && tags[i] == 2, eachindex(tags))
        JobLoggerTools.assert_benji(isempty(bad_lb),
            "Inconsistent tag/secondary mapping in bundle[4]: bad_lb=$(bad_lb)", jobid)

        bc_idx = findall(==(1), tags) 
        JobLoggerTools.assert_benji(all(sec_tags[bc_idx] .== 0), "YP_bc rows not all in secondary Y_lb", jobid)
        conf_bc = conf_nums[bc_idx]
        data_bc = get(Q1, Q2, Q3, Q4, w, bc_idx)

        for i in eachindex(conf_bc)
            push!(Q_YP_bc, has_weight ?
                (conf_bc[i], data_bc[1][i], data_bc[2][i], data_bc[3][i], data_bc[4][i], data_bc[5][i]) :
                (conf_bc[i], data_bc[1][i], data_bc[2][i], data_bc[3][i], data_bc[4][i])
            )
        end
    end

    # === Sanity checks
    bc_Y      = length(Q_Y_bc)
    ul_Y      = length(Q_Y_ul)
    bc_YP_len = length(Q_YP_bc)
    ul_YP_len = length(Q_YP_ul)

    if bc_YP_len == 0
        if bc_Y > 0
            JobLoggerTools.warn_benji("TRP=0 detected: bc_YP is empty but bc_Y is non-empty — skipping length check", jobid)
        end
    else
        JobLoggerTools.assert_benji(bc_Y == bc_YP_len, "bc length mismatch", jobid)
    end

    if ul_YP_len == 0
        if ul_Y > 0
            JobLoggerTools.warn_benji("TRP=0 detected: ul_YP is empty but ul_Y is non-empty — skipping length check", jobid)
        end
    else
        JobLoggerTools.assert_benji(ul_Y == ul_YP_len, "ul length mismatch", jobid)
    end

    return Q_Y_tr, Q_Y_bc, Q_YP_bc, Q_Y_ul, Q_YP_ul, Q_Y_lb, Q_Y_ORG
end

"""
    flatten_Q4_columns(
        Q_data::Vector{Tuple{Int, Float64, Float64, Float64, Float64}}
    ) -> (
        Vector{Float64},  # column 2
        Vector{Float64},  # column 3
        Vector{Float64},  # column 4
        Vector{Float64}   # column 5
    )

Extract and group columns `2-5` from each 5-tuple of the form `(Int, Float64, Float64, Float64, Float64)`.

# Arguments
- `Q_data`: A vector of 5-element tuples where the first element is an `Int` and the remaining are `Float64` values.

# Returns
- A 4-tuple of vectors, each collecting the respective components from position `2` to `5` across all tuples.
"""
function flatten_Q4_columns(
    Q_data::Vector{Tuple{Int, Float64, Float64, Float64, Float64}}
)
    return ntuple(i -> [row[i+1] for row in Q_data], 4)
end

"""
    flatten_Q5_columns(
        Q_data::Vector{Tuple{Int, Float64, Float64, Float64, Float64, Float64}}
    ) -> (
        Vector{Float64},  # column 2
        Vector{Float64},  # column 3
        Vector{Float64},  # column 4
        Vector{Float64},  # column 5
        Vector{Float64}   # column 6
    )

Extract and group columns `2-6` from each 6-tuple of the form `(Int, Float64, Float64, Float64, Float64, Float64)`.

# Arguments
- `Q_data`: A vector of 6-element tuples where the first element is an `Int` and the rest are `Float64` values.

# Returns
- A 5-tuple of vectors, each collecting the respective components from position `2` to `6` across all tuples.
"""
function flatten_Q5_columns(
    Q_data::Vector{Tuple{Int, Float64, Float64, Float64, Float64, Float64}}
)
    return ntuple(i -> [row[i+1] for row in Q_data], 5)
end

end  # module CumulantsBundleUtils