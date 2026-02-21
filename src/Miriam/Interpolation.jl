# ============================================================================
# src/Miriam/Interpolation.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module Interpolation

import ..Sarah.JobLoggerTools

"""
    is_local_max(y, i) -> Bool

Check if the element at index `i` is a strict local maximum.

# Arguments
- `y`: Vector of numerical values.
- `i`: Index to test (must satisfy `2` ``\\le`` `i` ``\\le`` `length(y)-1`).

# Returns
- `Bool`: `true` if `y[i]` is strictly greater than both neighbors, otherwise `false`.
"""
@inline is_local_max(y, i) = y[i-1] < y[i] > y[i+1]

"""
    is_local_min(y, i) -> Bool

Check if the element at index `i` is a strict local minimum.

# Arguments
- `y`: Vector of numerical values.
- `i`: Index to test (must satisfy `2` ``\\le`` `i` ``\\le`` `length(y)-1`).

# Returns
- `Bool`: `true` if `y[i]` is strictly smaller than both neighbors, otherwise `false`.
"""
@inline is_local_min(y, i) = y[i-1] > y[i] < y[i+1]

"""
    abs_soft(s; eps=1e-12) -> typeof(s)

Compute a stabilized absolute value of `s` using a soft approximation.

# Arguments
- `s`: Input scalar value.
- `eps::Real=1e-12`: Small positive constant added to maintain smoothness
  and avoid non-differentiability at ``s = 0``.

# Returns
- `Float64`: Approximate absolute value, defined as ``\\sqrt{s^2 + \\varepsilon^2}``.

# Notes
- Useful in skewness folding or optimization contexts where a differentiable
  approximation to ``\\left|s\\right|`` is required.
"""
abs_soft(s; eps=1e-12) = sqrt(s*s + eps*eps)

"""
    is_local_min_abs(y, i) -> Bool

Check whether the element at index `i` is a strict local minimum 
with respect to the *soft absolute* values of its neighbors.

# Arguments
- `y`: Vector of numerical values.
- `i`: Index to test (must satisfy `2` ``\\le`` `i` ``\\le`` `length(y)-1`).

# Returns
- `Bool`: `true` if `|y[i]|` (softened) is strictly smaller than both neighbors,
  otherwise `false`.

# Notes
- Uses [`abs_soft`](@ref) to stabilize near-zero values.
- Primarily used in skewness detection, where folding is applied.
"""
@inline function is_local_min_abs(y, i)
    yi = abs_soft(y[i]); yl = abs_soft(y[i-1]); yr = abs_soft(y[i+1])
    return yl > yi < yr
end

"""
    argmin_by_distance_to_center(
        x::AbstractVector, 
        idxs::Vector{Int}
    ) -> Int

Pick the index from `idxs` whose ``x``-coordinate is closest to the midpoint
of the overall domain.

# Arguments
- `x::AbstractVector`: Monotonic ``x``-coordinate values.
- `idxs::Vector{Int}`: Candidate indices.

# Returns
- `Int`: Index from `idxs` that minimizes ``\\left| x_i - x_{\\text{center}} \\right|``.
"""
function argmin_by_distance_to_center(
    x::AbstractVector, 
    idxs::Vector{Int}
)
    xc = 0.5 * (first(x) + last(x))
    return idxs[argmin(abs.(x[idxs] .- xc))]
end

"""
    preprobe_discrete(
        x::AbstractVector, 
        y::AbstractVector;
        mode::String, 
        seed_strategy::Symbol=:center
    ) -> (Int, Int, Int)

Perform a **pre-probing** step using *average* values.

# Modes
- `"susp"` → susceptibility: choose a discrete maximum.
- `"skew"` → skewness: choose a discrete minimum of ``\\left|y\\right|``.
- `"kurt"` → kurtosis: choose a discrete minimum.

# Seed strategies
- `:center` (default): choose the local extremum closest to the center of ``x``.
- `:global`          : ignore local structure and directly choose the global extremum
  (global ``\\max`` for `"susp"`, global ``\\min \\, \\left|y\\right|`` for `"skew"`, global ``\\min`` for `"kurt"`).

# Arguments
- `x::AbstractVector`      : ``\\kappa`` grid (trajectory points).
- `y::AbstractVector`      : target values on ``x`` (one cumulant curve).
- `mode::String`           : `"susp" | "skew" | "kurt"`.
- `seed_strategy::Symbol`  : `:center` or `:global`.

# Returns
- `(L, R, i_star)`:
    - `L::Int` : Left guard index (nearest extremum to the left of `i_star`, or `1` if none).
    - `R::Int` : Right guard index (nearest extremum to the right of `i_star`, or `n` if none).
    - `i_star::Int` : Seed index chosen according to the mode and strategy.
"""
function preprobe_discrete(
    x::AbstractVector,         
    y::AbstractVector;
    mode::String, 
    seed_strategy::Symbol=:global
)
    n = length(x); JobLoggerTools.assert_benji(length(y) == n && n ≥ 3, "length(y) must equal n and n ≥ 3")

    # 1) Collect strict local extrema
    maxima, minima = Int[], Int[]
    for i in 2:n-1
        if y[i-1] < y[i] > y[i+1]; push!(maxima, i); end
        if y[i-1] > y[i] < y[i+1]; push!(minima, i); end
    end
    extrema = sort!(unique!(vcat(maxima, minima)))  # all internal extrema (excluding endpoints)

    # 2) Select seed index (i_star) depending on strategy and mode
    i_star = if seed_strategy === :global
        # Global strategy: pick extremum over the full domain
        if mode == "susp"      # susceptibility → global maximum
            argmax(y)
        elseif mode == "skew"  # skewness → global minimum of |y|
            argmin(abs.(y))
        elseif mode == "kurt"  # kurtosis → global minimum
            argmin(y)
        else
            JobLoggerTools.error_benji("unknown mode: $mode")
        end
    else  # :center strategy (default behavior)
        if mode == "susp"
            isempty(maxima) ? argmax(y) : argmin_by_distance_to_center(x, maxima)
        elseif mode == "skew"
            yabs = abs.(y)
            minima_abs = Int[]
            for i in 2:n-1
                if yabs[i-1] > yabs[i] < yabs[i+1]; push!(minima_abs, i); end
            end
            isempty(minima_abs) ? argmin(yabs) : argmin_by_distance_to_center(x, minima_abs)
        elseif mode == "kurt"
            isempty(minima) ? argmin(y) : argmin_by_distance_to_center(x, minima)
        else
            JobLoggerTools.error_benji("unknown mode: $mode")
        end
    end

    # 3) Guard window [L, R] around i_star
    #    If no extremum exists on one side, fallback to boundary (1 or n)
    L = 1
    for j in reverse(extrema)
        if j < i_star
            L = j; break
        end
    end
    R = n
    for j in extrema
        if j > i_star
            R = j; break
        end
    end

    return (L, R, i_star)
end

"""
    preprobe_discrete_from_bundle(
        x::AbstractVector,
        cumulants_avg::Vector{<:AbstractVector};
        mode::Symbol
    ) -> (Int, Int, Int, Int)

Dispatch pre-probing to the appropriate cumulant component based on `mode`,
assuming the bundle layout:
- ``j=1``: chiral condensate, 
- ``j=2``: susceptibility, 
- ``j=3``: skewness,
- ``j=4``: kurtosis, 
- ``j=5``: Binder.

# Arguments
- `x`             : ``\\kappa`` grid.
- `cumulants_avg` : length-5 vector of average-only curves (each `length = length(x)`).
- `mode`          : `:susp` | `:skew` | `:kurt`.

# Returns
- `(L, R, i_star, j_used)`:
    - `L, R, i_star` as in [`preprobe_discrete`](@ref).
    - `j_used` is the selected component index (`2` for `:susp`, `3` for `:skew`, `4` for `:kurt`).
"""
function preprobe_discrete_from_bundle(
    x::AbstractVector,
    cumulants_avg::Vector{<:AbstractVector};
    mode::Symbol
)
    JobLoggerTools.assert_benji(length(cumulants_avg) == 5, "length(cumulants_avg) must be 5")
    j = mode === :susp ? 2 :
        mode === :skew ? 3 :
        mode === :kurt ? 4 : JobLoggerTools.error_benji("unknown mode (use :susp, :skew, or :kurt)")

    y = cumulants_avg[j]
    JobLoggerTools.assert_benji(length(y) == length(x), "length(y) must equal length(x)")

    # map symbol to inner string mode expected by preprobe_discrete
    inner_mode = mode === :susp ? "susp" :
                 mode === :skew ? "skew" :
                                  "kurt"

    L, R, i_star = preprobe_discrete(x, y; mode=inner_mode)
    return (L, R, i_star)
end

"""
    seed_extremum_if_ok(
        y::AbstractVector, 
        mode::Symbol,
        seed::Int, 
        L::Int, 
        R::Int
    ) -> Union{Int, Nothing}

Check whether the candidate index `seed` is a valid local extremum within 
the specified range and return it if so.

# Arguments
- `y::AbstractVector`: Sequence of numerical values.
- `mode::Symbol`: Extremum type to check. Accepted values:
    - `:susp` → local maximum (susceptibility peak),
    - `:skew` → local minimum of folded absolute values (skewness),
    - otherwise → local minimum.
- `seed::Int`: Candidate index to test.
- `L::Int`: Left boundary (inclusive).
- `R::Int`: Right boundary (inclusive).

# Returns
- `Int`: `seed` index if it is a valid extremum.
- `Nothing`: if the check fails or the index is out of range.

# Notes
- Index must satisfy `2` ``\\le`` `seed` ``\\le`` `length(y)-1` to allow left/right neighbor checks.
"""
function seed_extremum_if_ok(
    y::AbstractVector, 
    mode::Symbol,
    seed::Int, 
    L::Int, 
    R::Int
)
    n = length(y)
    if 2 ≤ seed ≤ n-1 && L ≤ seed ≤ R
        ok = mode === :susp ? is_local_max(y, seed)  :
             mode === :skew ? is_local_min_abs(y, seed) :
                              is_local_min(y, seed)
        return ok ? seed : nothing
    end
    return nothing
end

"""
    solve_quad(
        x::NTuple{3,T}, 
        y::NTuple{3,T}
    ) -> NTuple{3,T}

Solve for quadratic coefficients `c = (c0,c1,c2)` such that
``y \\approx c_0 + c_1 \\, x + c_2 \\, x^2`` through the 3 points `(k[i], y[i])`.

Notes:
- Caller is responsible for ensuring ``x`` are distinct and well-conditioned.
"""
function solve_quad(
    x::AbstractVector, 
    y::AbstractVector
)
    JobLoggerTools.assert_benji(length(x) == 3 && length(y) == 3, "length(x) == 3 == length(y) must hold")
    Xmat = @inbounds [1.0  x[1]  x[1]^2;
                      1.0  x[2]  x[2]^2;
                      1.0  x[3]  x[3]^2]
    c = Xmat \ y
    return (c[1], c[2], c[3])  # (intercept, linear, quadratic)
end

"""
    first_extremum_from_seed(
        y::AbstractVector, 
        mode::Symbol, 
        seed::Int, 
        L::Int, 
        R::Int, 
        dir::Int
    ) -> Union{Int,Nothing}

Scan from `seed` toward `dir` (`dir = -1` for left, `+1` for right) within the
hard-guard window `[L, R]`, and return the **first** index `i` that matches
the shape required by `mode`:
- :susp → local maximum ([`is_local_max`](@ref))
- :skew → local minimum of ``\\left|y\\right|`` ([`is_local_min_abs`](@ref))
- :kurt → local minimum ([`is_local_min`](@ref))

If no such interior index exists in the scan range, return `nothing`.

# Notes
- Index must have neighbors: `2` ``\\le`` `i` ``\\le`` `length(y)-1`.
- The scan never steps outside `[L, R]`.
"""
function first_extremum_from_seed(
    y::AbstractVector, 
    mode::Symbol,
    seed::Int, 
    L::Int, 
    R::Int, 
    dir::Int
)
    n = length(y)
    JobLoggerTools.assert_benji(
        1 ≤ L && L ≤ seed && seed ≤ R && R ≤ n,
        "Condition 1 ≤ L ≤ seed ≤ R ≤ n must hold"
    )
    # interior valid region for local-extremum tests
    i_start = seed
    if dir < 0
        # scan left: seed-1 down to max(L+1, 2)
        for i in (i_start-1):-1:max(L+1, 2)
            ok = mode === :susp ? is_local_max(y, i)  :
                 mode === :skew ? is_local_min_abs(y, i) :
                                  is_local_min(y, i)
            if ok; return i; end
        end
    else
        # scan right: seed+1 up to min(R-1, n-1)
        for i in (i_start+1):min(R-1, n-1)
            ok = mode === :susp ? is_local_max(y, i)  :
                 mode === :skew ? is_local_min_abs(y, i) :
                                  is_local_min(y, i)
            if ok; return i; end
        end
    end
    return nothing
end

"""
    pick_best_candidate(
        iLcand::Union{Int,Nothing},
        iRcand::Union{Int,Nothing},
        x::AbstractVector, 
        seed::Int;
        prefer::String="near",
        mode::Symbol=:susp,
        y::Union{Nothing,AbstractVector}=nothing,
        lambda::Real=0.0
    ) -> Union{Int, Nothing}

Choose between left/right candidate indices.

- `prefer="near"`: pick the one closer (in ``x``) to the seed.
- `prefer="objective"`:
    - `:susp` → larger ``y`` is better
    - `:skew` → smaller ``\\left|y\\right|`` is better
    - `:kurt` → smaller ``y`` is better
- `prefer="hybrid"`: combine objective with distance penalty controlled by `lambda`:
    - `:susp` → maximize ``y_i - \\lambda \\, \\left| x_i - x_{\\text{seed}} \\right|``
    - `:skew` → minimize ``\\left|y_i\\right| + \\lambda \\, \\left| x_i - x_{\\text{seed}} \\right|``
    - `:kurt` → minimize ``y_i + \\lambda \\, \\left| x_i - x_{\\text{seed}} \\right|``

Returns `nothing` if both inputs are `nothing`.
"""
function pick_best_candidate(
    iLcand::Union{Int,Nothing},
    iRcand::Union{Int,Nothing},
    x::AbstractVector, 
    seed::Int;
    prefer::String="near",
    mode::Symbol=:susp,
    y::Union{Nothing,AbstractVector}=nothing,
    lambda::Real=0.0
)
    # handle missing
    if isnothing(iLcand) && isnothing(iRcand)
        return nothing
    elseif isnothing(iLcand)
        return iRcand
    elseif isnothing(iRcand)
        return iLcand
    end

    # both exist
    iL = iLcand::Int
    iR = iRcand::Int

    if prefer == "near"
        return abs(x[iL] - x[seed]) ≤ abs(x[iR] - x[seed]) ? iL : iR
    end

    JobLoggerTools.assert_benji(
        y !== nothing,
        "pick_best_candidate: y must be provided for prefer='$prefer'"
    )

    if prefer == "objective"
        if mode === :susp
            # larger is better
            return (y[iL] ≥ y[iR]) ? iL : iR
        elseif mode === :skew
            # smaller |y| is better
            return (abs(y[iL]) ≤ abs(y[iR])) ? iL : iR
        else # :kurt
            # smaller is better
            return (y[iL] ≤ y[iR]) ? iL : iR
        end
    elseif prefer == "hybrid"
        dL = abs(x[iL] - x[seed]); dR = abs(x[iR] - x[seed])
        if mode === :susp
            # maximize score
            sL = y[iL] - lambda*dL
            sR = y[iR] - lambda*dR
            return (sL ≥ sR) ? iL : iR
        elseif mode === :skew
            # minimize penalized |y|
            sL = abs(y[iL]) + lambda*dL
            sR = abs(y[iR]) + lambda*dR
            return (sL ≤ sR) ? iL : iR
        else # :kurt
            # minimize penalized y
            sL = y[iL] + lambda*dL
            sR = y[iR] + lambda*dR
            return (sL ≤ sR) ? iL : iR
        end
    else
        JobLoggerTools.error_benji("unknown prefer='$prefer' (use 'near' | 'objective' | 'hybrid')")
    end
end

"""
    find_transition_cumulants(
        cumulants_all::Vector{Vector{Vector{T}}},
        kappa::Vector{T},
        target_index::Int,              # 2: sus, 3: skew, 4: kurt
        avg_win_L::Int, avg_win_R::Int, # hard guard window from pre-probe
        avg_ext_cand::Int,              # seed index from pre-probe
        jobid::Union{Nothing,String}=nothing
    ) where T -> Vector{Vector{T}}

Refine the transition point per bootstrap resample using **only** the region
within the provided guard window `[avg_win_L, avg_win_R]` and scanning outward
from the seed `avg_ext_cand`:

1. For each resample `b`, build the target curve for the chosen observable
   (`target_index`: `2=susp`, `3=skew`, `4=kurt`).
2. Scan **left** from the seed up to `avg_win_L` and **right** up to `avg_win_R`,
   stopping at the **first** encountered local extremum that matches the mode.
   (Skewness uses a local minimum of `abs(y)`.)
3. Choose one of the found extremes (prefer near the seed), and perform a
   3-point local quadratic fit using its immediate neighbors `(i-1,i,i+1)`.
4. Extract ``\\kappa_T``:
   - skewness: the nearest real root of the quadratic to the center point
   - susceptibility/kurtosis: vertex ``\\kappa^{\\ast} = - \\dfrac{c_1}{2 \\, c_2}``
5. Interpolate all five observables at ``\\kappa_T`` via their own local quadratic
   fits re-using the same ``\\kappa``-triplet.
6. If a valid 3-point neighborhood cannot be formed (e.g., candidate at boundary),
   try the other side candidate; if still impossible, mark this resample as boundary
   failure and return `NaN`s for it.

# Returns
- `vec_valids::Vector{Vector{T}}` of length 6:
  `[κ_T, cond(κ_T), sus(κ_T), skew(κ_T), kurt(κ_T), binder(κ_T)]`,
  each with NaNs removed across resamples.
"""
function find_transition_cumulants(
    cumulants_all::Vector{Vector{Vector{T}}},
    kappa::Vector{T},
    target_index::Int,           # 2: susceptibility, 3: skewness, 4: kurtosis
    avg_win_L::Int, avg_win_R::Int, avg_ext_cand::Int,
    jobid::Union{Nothing, String}=nothing
) where T

    N_resample = length(cumulants_all[1][1])
    nkappaT    = length(cumulants_all[1])   # number of κ points
    interpolated = Vector{Vector{T}}(undef, N_resample)  # per-resample result
    is_boundary  = falses(N_resample)

    # Mode mapping
    mode = target_index == 2 ? :susp : (target_index == 3 ? :skew : :kurt)

    # Hard guards sanity
    JobLoggerTools.assert_benji(
        1 ≤ avg_win_L && avg_win_L ≤ avg_ext_cand && avg_ext_cand ≤ avg_win_R && avg_win_R ≤ nkappaT,
        "1 ≤ avg_win_L ≤ avg_ext_cand ≤ avg_win_R ≤ nkappaT must hold"
    )
    JobLoggerTools.assert_benji(
        nkappaT ≥ 3,
        "nkappaT must be ≥ 3"
    )

    for b in 1:N_resample
        # 0) Extract per-resample curves across κ for all 5 observables
        vals = [[cumulants_all[j][i][b] for i in 1:nkappaT] for j in 1:5]
        target_vals = vals[target_index]

        i_seed = seed_extremum_if_ok(target_vals, mode, avg_ext_cand, avg_win_L, avg_win_R)

        # 1) Scan left/right **inside [avg_win_L, avg_win_R]** only
        iLcand = first_extremum_from_seed(target_vals, mode, avg_ext_cand, avg_win_L, avg_win_R, -1)
        iRcand = first_extremum_from_seed(target_vals, mode, avg_ext_cand, avg_win_L, avg_win_R, +1)

        # 2) Pick one candidate (prefer near seed). Fallbacks later if needed.
        # i_cand = pick_best_candidate(iLcand, iRcand, kappa, avg_ext_cand; prefer="near")
        # i_cand = i_seed !== nothing ? i_seed :
        #         pick_best_candidate(iLcand, iRcand, kappa, avg_ext_cand; prefer="near")
        i_cand = if i_seed !== nothing
            i_seed
        else
            pick_best_candidate(iLcand, iRcand, kappa, avg_ext_cand;
                                prefer="near",                   # or "objective"/"hybrid"
                                mode=mode,
                                y=target_vals,
                                lambda=0.0)                      # tune if prefer="hybrid"
        end


        # 3) Try to form a valid 3-point neighborhood; if fail, try the other side
        #    (because the "nearer" one might sit at boundary in this resample)
        function try_fit_at(i0)::Union{Nothing, Tuple{T,T,T,T,T,T}}
            if i0 === nothing; return nothing; end
            i = i0::Int
            if !(2 ≤ i ≤ nkappaT-1)
                return nothing
            end
            κ_vec = @inbounds [kappa[i-1], kappa[i], kappa[i+1]]
            # --- Target quadratic fit to locate κ_T ---
            y_vec = @inbounds [target_vals[i-1], target_vals[i], target_vals[i+1]]
            c0, c1, c2 = solve_quad(κ_vec, y_vec)

            κ_T = zero(T)
            if mode === :skew
                # Solve c0 + c1*κ + c2*κ^2 = 0
                Δ = c1^2 - 4*c0*c2
                if Δ < 0
                    return nothing
                end
                r1 = 0.5*(-c1 + sqrt(Δ)) / c2
                r2 = 0.5*(-c1 - sqrt(Δ)) / c2
                κ_T = abs(r1 - κ_vec[2]) < abs(r2 - κ_vec[2]) ? r1 : r2
            else
                # Vertex
                κ_T = -0.5*c1 / c2
            end

            # Interpolate all observables at κ_T using the same κ triplet
            out = Vector{T}(undef, 6)
            out[1] = κ_T
            for j in 1:5
                yj = @inbounds [vals[j][i-1], vals[j][i], vals[j][i+1]]
                d0, d1, d2 = solve_quad(κ_vec, yj)
                out[j+1] = d0 + d1*κ_T + d2*κ_T^2
            end
            return (out[1], out[2], out[3], out[4], out[5], out[6])
        end

        result = try_fit_at(i_cand)
        if result === nothing
            # try the opposite side if available
            other = (i_cand === iLcand) ? iRcand : iLcand
            result = try_fit_at(other)
        end

        if result === nothing
            # Final fallback: mark boundary failure (keeps previous semantics)
            is_boundary[b] = true
            interpolated[b] = fill(T(NaN), 6)
            continue
        else
            interpolated[b] = [result[1], result[2], result[3], result[4], result[5], result[6]]
        end
    end

    # Build and return NaN-free vectors per component (1..6)
    vec_valids = Vector{Vector{T}}(undef, 6)
    for i in 1:6
        vec_i = getindex.(interpolated, i)
        vec_valids[i] = [v for v in vec_i if !isnan(v)]
    end
    return vec_valids
end

end  # module Interpolation