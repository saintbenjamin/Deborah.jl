# ============================================================================
# src/Esther/BootstrapDerivedCalculator.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module BootstrapDerivedCalculator

import ..SingleCumulant
import ..Sarah.DatasetPartitioner
import ..Sarah.JobLoggerTools

"""
    compute_bootstrap_derived!(
        bd::Dict, 
        LatVol::Int
    ) -> Nothing

Compute and store derived observables (condensate, susceptibility, skewness, kurtosis)
from bootstrap-averaged quark moment data (``Q_{n} \\; (n=1,2,3,4)``).

# Arguments
- `bd::Dict`: A dictionary containing `:mean` key, which maps to another dictionary
  that must include entries of the form `"Qn:<label>"` for `n = 1` to `4` and various `<label>` strings.
- `LatVol::Int`: Lattice volume, used to normalize condensate and susceptibility values.

# Behavior
- For each label in the predefined list of observables (`"Y_info"`, `"YmYP"`, etc.),
  the function computes:
    - `"cond:<label>"`: chiral condensate using ``Q_1``,
    - `"susp:<label>"`: susceptibility using ``Q_1`` and ``Q_2``,
    - `"skew:<label>"`: skewness using ``Q_1``, ``Q_2``, ``Q_3``,
    - `"kurt:<label>"`: kurtosis using ``Q1``, ``Q_2``, ``Q_3``, ``Q_4``.
- The results are stored in-place under `bd[:mean]`.

# Returns
- `Nothing`: This function mutates `bd` in-place.

# Requirements
- The `bd[:mean]` dictionary must contain all required moment keys for each target label.
- The functions [`Deborah.Esther.SingleCumulant.calc_quark_condensate`](@ref), [`Deborah.Esther.SingleCumulant.calc_susceptibility`](@ref), [`Deborah.Esther.SingleCumulant.calc_skewness`](@ref), and [`Deborah.Esther.SingleCumulant.calc_kurtosis`](@ref) must be defined.
"""
function compute_bootstrap_derived!(
    bd::Dict,
    LatVol::Int
)
    targets = [
        "Y_info", "YmYP", "YP_ul", "Y_P1", "Y_P2", "Y_lb"
    ]

    for label in targets
        bd[:mean]["cond:$label"] = SingleCumulant.calc_quark_condensate(bd[:mean]["Q1:$label"], LatVol)
        bd[:mean]["susp:$label"] = SingleCumulant.calc_susceptibility(bd[:mean]["Q1:$label"], bd[:mean]["Q2:$label"], LatVol)
        bd[:mean]["skew:$label"] = SingleCumulant.calc_skewness(bd[:mean]["Q1:$label"], bd[:mean]["Q2:$label"], bd[:mean]["Q3:$label"])
        bd[:mean]["kurt:$label"] = SingleCumulant.calc_kurtosis(bd[:mean]["Q1:$label"], bd[:mean]["Q2:$label"], bd[:mean]["Q3:$label"], bd[:mean]["Q4:$label"])
    end
end

"""
    compute_bootstrap_derived!(
        bd::Dict,
        LatVol::Int,
        partition::DatasetPartitioner.DatasetPartitionInfo,
    ) -> Nothing

Compute bootstrap-averaged cumulants (condensate, susceptibility, skewness, kurtosis)
for each *subset* directly from ``Q``-moments (``Q_{n} \\; (n=1,2,3,4)``), then construct **post-cumulant**
bias-corrected combinations `Y_P1`, and `Y_P2` from those cumulants.

This variant differs intentionally from the simpler overload that only maps
``Q_n \\; (n=1,2,3,4) \\to \\{ \\Sigma , \\chi , S , K \\}`` for pre-defined labels. Here, bias correction
is applied **after** cumulants are formed (i.e., on ``\\{ \\Sigma , \\chi , S , K \\}``),
not at the ``Q``-moment level. This ordering is generally *not* recommended for
production use, but is kept for completeness and for controlled comparisons.

# Arguments
- `bd::Dict`: A dictionary that must contain a `:mean` table of bootstrap-averaged
  **moments** and will be mutated to include **cumulants** and their derived
  combinations. The following keys **must** exist for each subset label in
  `("Y_info", "Y_lb", "Y_bc", "Y_ul", "YP_bc", "YP_ul")`:
  - `"Q1:<label>"`, `"Q2:<label>"`, `"Q3:<label>"`, `"Q4:<label>"`
- `LatVol::Int`: Lattice volume used to normalize condensate and susceptibility.
- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo): Partition metadata used
  to compute weights and training coverage:
  - `N_cnf`: total configurations
  - `N_lb`: labeled configurations
  - `N_ul`: unlabeled configurations
  - `N_tr`: training configurations (subset of labeled)
  Internally we use:
  - ``w_{\\texttt{lb}} = \\dfrac{N_{\\texttt{lb}}}{N_{\\texttt{cnf}}}``
  - ``w_{\\texttt{ul}} = \\dfrac{N_{\\texttt{ul}}}{N_{\\texttt{cnf}}}``

# Behavior
1. **Per-subset cumulants.**
   For each `label` ``\\in`` `("Y_info","Y_lb","Y_bc","Y_ul","YP_bc","YP_ul")`, read
   ``Q_n \\; (n=1,2,3,4)`` from `bd[:mean]` and compute
   - [`"cond:<label>" = calc_quark_condensate(Q1, LatVol)`](@ref Deborah.Esther.SingleCumulant.calc_quark_condensate)
   - [`"susp:<label>" = calc_susceptibility(Q1, Q2, LatVol)`](@ref Deborah.Esther.SingleCumulant.calc_susceptibility)
   - [`"skew:<label>" = calc_skewness(Q1, Q2, Q3)`](@ref Deborah.Esther.SingleCumulant.calc_skewness)
   - [`"kurt:<label>" = calc_kurtosis(Q1, Q2, Q3, Q4)`](@ref Deborah.Esther.SingleCumulant.calc_kurtosis)

2. **Post-cumulant bias combinations.**
   For each `C` ``\\in`` `("cond","susp","skew","kurt")`, form:
   - `YmYP = (C:Y_bc) - (C:YP_bc)` (bias estimate on the **cumulant**)
   - `Y_P1 = (N_tr == N_lb) ? (C:YP_ul) : (C:YP_ul + YmYP)`
     (i.e., pass-through when **`TRP` = 100%**, ignoring `YmYP`)
   - `Y_P2 = w_lb * (C:Y_lb) + w_ul * Y_P1`
   Results are stored under keys `"\$C:YmYP"`, `"\$C:Y_P1"`, and `"\$C:Y_P2"`.

# Returns
- `Nothing`. The function mutates `bd[:mean]` in place, adding cumulant and
  derived-combination arrays for all supported labels.

# Notes
- **Ordering caveat.** Bias correction here is applied *after* mapping ``Q_{n} \\; (n=1,2,3,4)``
  to ``\\{ \\Sigma , \\chi , S , K \\}``. In most workflows, applying bias handling earlier
  (at the ``Q``-moment stage) is preferable to preserve the algebra of derived
  observables. Keep this function for record-keeping, audits, or `A`/`B` comparisons.
- **TRP=100% shortcut.** When `N_tr == N_lb`, we set `Y_P1 = YP_ul` (pass-through)
  and effectively ignore `YmYP` for `Y_P1`. This matches the intent that no
  bias-correction set is available/needed in that limit.
- All vector operations are broadcasted element-wise; array shapes of inputs
  (``Q_{n} \\; (n=1,2,3,4)``) must be consistent across labels.

# Requirements
- `bd[:mean]` must contain the required `"Qk:<label>"` keys for all listed labels.
- The following functions must be defined and accept bootstrap-averaged inputs:
  [`Deborah.Esther.SingleCumulant.calc_quark_condensate`](@ref), [`Deborah.Esther.SingleCumulant.calc_susceptibility`](@ref),
  [`Deborah.Esther.SingleCumulant.calc_skewness`](@ref), [`Deborah.Esther.SingleCumulant.calc_kurtosis`](@ref).

# Example
```julia
bd = Dict(:mean => Dict(
    "Q1:Y_lb"=>q1_lb, "Q2:Y_lb"=>q2_lb, "Q3:Y_lb"=>q3_lb, "Q4:Y_lb"=>q4_lb,
    "Q1:Y_bc"=>q1_bc, "Q2:Y_bc"=>q2_bc, "Q3:Y_bc"=>q3_bc, "Q4:Y_bc"=>q4_bc,
    "Q1:YP_bc"=>q1_ypbc, "Q2:YP_bc"=>q2_ypbc, "Q3:YP_bc"=>q3_ypbc, "Q4:YP_bc"=>q4_ypbc,
    "Q1:YP_ul"=>q1_ypul, "Q2:YP_ul"=>q2_ypul, "Q3:YP_ul"=>q3_ypul, "Q4:YP_ul"=>q4_ypul,
    "Q1:Y_ul"=>q1_ul, "Q2:Y_ul"=>q2_ul, "Q3:Y_ul"=>q3_ul, "Q4:Y_ul"=>q4_ul,
    "Q1:Y_info"=>q1_info, "Q2:Y_info"=>q2_info, "Q3:Y_info"=>q3_info, "Q4:Y_info"=>q4_info,
))

compute_bootstrap_derived!(bd, LatVol, partition)
# bd[:mean] now includes "cond:<label>", "susp:<label>", "skew:<label>", "kurt:<label>",
# as well as post-cumulant "YmYP", "Y_P1", and "Y_P2" for each cumulant.
```
"""
function compute_bootstrap_derived!(
    bd::Dict,
    LatVol::Int,
    partition::DatasetPartitioner.DatasetPartitionInfo,
)
    # We only need these three from partition
    w_lb = partition.N_lb / partition.N_cnf
    w_ul = partition.N_ul / partition.N_cnf
    N_tr = partition.N_tr
    N_lb = partition.N_lb

    # 1) Compute cumulants for each subset directly from Q-moments
    subset_labels = ("Y_info", "Y_lb", "Y_bc", "Y_ul", "YP_bc", "YP_ul")

    # Optional: lightweight guards for missing keys
    JobLoggerTools.assert_benji(haskey(bd, :mean), "bd[:mean] is missing")
    mean_tbl = bd[:mean]

    for label in subset_labels
        # Assert Q1..Q4 exist for each label
        JobLoggerTools.assert_benji(haskey(mean_tbl, "Q1:$label"), "Missing key: Q1:$label")
        JobLoggerTools.assert_benji(haskey(mean_tbl, "Q2:$label"), "Missing key: Q2:$label")
        JobLoggerTools.assert_benji(haskey(mean_tbl, "Q3:$label"), "Missing key: Q3:$label")
        JobLoggerTools.assert_benji(haskey(mean_tbl, "Q4:$label"), "Missing key: Q4:$label")

        q1 = mean_tbl["Q1:$label"]
        q2 = mean_tbl["Q2:$label"]
        q3 = mean_tbl["Q3:$label"]
        q4 = mean_tbl["Q4:$label"]

        mean_tbl["cond:$label"] = SingleCumulant.calc_quark_condensate(q1, LatVol)
        mean_tbl["susp:$label"] = SingleCumulant.calc_susceptibility(q1, q2, LatVol)
        mean_tbl["skew:$label"] = SingleCumulant.calc_skewness(q1, q2, q3)
        mean_tbl["kurt:$label"] = SingleCumulant.calc_kurtosis(q1, q2, q3, q4)
    end

    # 2) Derive YmYP, P1, and P2 from cumulant subsets
    models_C = ("cond", "susp", "skew", "kurt")

    for c in models_C
        JobLoggerTools.assert_benji(haskey(mean_tbl, "$c:Y_lb"),  "Missing key: $c:Y_lb")
        JobLoggerTools.assert_benji(haskey(mean_tbl, "$c:Y_bc"),  "Missing key: $c:Y_bc")
        JobLoggerTools.assert_benji(haskey(mean_tbl, "$c:YP_bc"), "Missing key: $c:YP_bc")
        JobLoggerTools.assert_benji(haskey(mean_tbl, "$c:YP_ul"), "Missing key: $c:YP_ul")

        Y_lb  = mean_tbl["$c:Y_lb"]
        Y_bc  = mean_tbl["$c:Y_bc"]
        YP_bc = mean_tbl["$c:YP_bc"]
        YP_ul = mean_tbl["$c:YP_ul"]

        if N_tr == 0
            # No training set: YmYP = 0, P1 = P2 = Y_lb
            mean_tbl["$c:YmYP"] = zero.(Y_lb)
            mean_tbl["$c:Y_P1"] = copy(Y_lb)
            mean_tbl["$c:Y_P2"] = copy(Y_lb)
        else
            # With training set
            YmYP = Y_bc .- YP_bc
            # Pass-through when TRP=100% (i.e., N_tr == N_lb): ignore YmYP in Y_P1
            Y_P1 = (N_tr == N_lb) ? YP_ul : (YP_ul .+ YmYP)
            Y_P2 = w_lb .* Y_lb .+ w_ul .* Y_P1

            mean_tbl["$c:YmYP"] = YmYP
            mean_tbl["$c:Y_P1"] = Y_P1
            mean_tbl["$c:Y_P2"] = Y_P2
        end
    end

    return nothing
end

end  # module BootstrapDerivedCalculator