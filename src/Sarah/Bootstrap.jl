# ============================================================================
# src/Sarah/Bootstrap.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Bootstrap

import ..Random
import ..Statistics
import ..StatsBase
import ..Distributions

import ..JobLoggerTools

"""
    prefix_sums(
        arr::AbstractVector{<:Real}
    ) -> Vector{Float64}

Compute a one-based prefix-sum array for `arr` to enable ``O(1)`` block-sum queries.

# Arguments
- `arr::AbstractVector{<:Real}` : Input data.

# Behavior
- Returns a vector `ps` of length `length(arr)+1` with `ps[1] = 0.0` and
  `ps[i+1] = ps[i] + arr[i]` for `i = 1:length(arr)`.

# Returns
- `Vector{Float64}` : Prefix-sum array `ps` such that the sum of `arr[i:j]`
  equals `ps[j+1] - ps[i]`.
"""
@inline function prefix_sums(
    arr::AbstractVector{<:Real}
)
    ps = Vector{Float64}(undef, length(arr)+1)
    ps[1] = 0.0
    @inbounds for i in eachindex(arr)
        ps[i+1] = ps[i] + arr[i]
    end
    return ps
end

"""
    block_sum(
        ps::AbstractVector{<:Real}, 
        i::Int, 
        len::Int, 
        N::Int
    ) -> Float64

Return the sum of a block of length `len` starting at index `i`, using the
prefix-sum array `ps`. Supports circular wrap-around when the block exceeds `N`.

# Arguments
- `ps::AbstractVector{<:Real}` : Prefix-sum array of length `N+1`.
- `i::Int`                     : ``1``-based start index of the block.
- `len::Int`                   : Block length (``\\ge 0``).
- `N::Int`                     : Population size (length of the original array).

# Behavior
- If `len == 0`, returns `0.0`.
- If `i + len - 1 ≤ N`, returns `ps[i+len] - ps[i]`.
- Otherwise, wraps around and returns `(ps[N+1] - ps[i]) + ps[(i+len-1) - N]`.

# Returns
- `Float64` : Sum over the requested block.
"""
@inline function block_sum(
    ps::AbstractVector{<:Real}, 
    i::Int, 
    len::Int, 
    N::Int
)
    len == 0 && return 0.0
    j = i + len - 1
    if j <= N
        @inbounds return ps[j+1] - ps[i]
    else
        @inbounds return (ps[N+1] - ps[i]) + ps[j - N]
    end
end

"""
    nblk_lastlen(
        N::Int, 
        block::Int
    ) -> (Int, Int)

Compute the number of blocks `nblk` and the trailing block length `lastlen`
needed to cover `N` items with blocks of size `block`.

# Arguments
- `N::Int`     : Population size (``\\ge 0``).
- `block::Int` : Block size (``\\ge 1``).

# Behavior
- Uses `nblk = cld(N, block)`.
- The final block may be shorter: `lastlen = N - block*(nblk - 1)`.
- If `N == 0`, returns `(0, 0)`.

# Returns
- `(Int, Int)` : `(nblk, lastlen)`.
"""
@inline function nblk_lastlen(
    N::Int, 
    block::Int
)
    N == 0 && return 0, 0
    nblk = cld(N, block)
    lastlen = N - block*(nblk - 1)
    return nblk, lastlen
end

"""
    gen_block_plan!(
        starts::Matrix{Int}, 
        rng::Random.AbstractRNG,
        N::Int, 
        blk::Int, 
        nblk::Int; 
        method::String="moving"
    ) -> Matrix{Int}

Generate a plan of block start indices for block bootstrap. Each row of `starts`
holds `nblk` start positions for one bootstrap replicate.

# Arguments
- `starts::Matrix{Int}`        : Preallocated `N_bs` ``\\times`` `nblk` matrix to fill.
- `rng::Random.AbstractRNG`    : RNG instance.
- `N::Int`                     : Population size (``\\ge 0``).
- `blk::Int`                   : Block size (``\\ge 1``).
- `nblk::Int`                  : Number of blocks per replicate.
- `method::String="moving"`    : One of `"moving"`, `"nonoverlapping"` (or `"nbb"`),
                                 `"circular"` (or `"cbb"`). `blk==1` implies i.i.d.

# Behavior
- `blk == 1` (i.i.d.): each start is sampled i.i.d. from `1:N` (with replacement).
- `"moving"`: starts sampled i.i.d. from `1:(N-blk+1)` (no wrap).
- `"nonoverlapping"`: use only `N_eff = div(N, blk)*blk`; starts are
  `{1, 1+blk, …, 1+(div(N,blk)-1)*blk}` sampled with replacement (no wrap).
- `"circular"`: starts sampled i.i.d. from `1:N` (wrap applied when summing).
- If `N == 0`, returns `starts` unchanged.

# Returns
- `Matrix{Int}` : The filled `starts` matrix (same object).
"""
function gen_block_plan!(
    starts::Matrix{Int},
    rng::Random.AbstractRNG,
    N::Int, 
    blk::Int, 
    nblk::Int;
    method::String="moving"
)
    N == 0 && return starts
    @inbounds for t in 1:size(starts,1)
        if blk == 1
            for k in 1:nblk
                starts[t,k] = Random.rand(rng, 1:N)
            end

        elseif method == "moving"
            max_start = N - blk + 1
            for k in 1:nblk
                starts[t,k] = Random.rand(rng, 1:max_start)
            end

        elseif method == "nonoverlapping" || method == "nbb"
            nblocks = div(N, blk)
            JobLoggerTools.assert_benji(
                nblocks > 0,
                "div(N, blk) must be ≥ 1"
            )
            for k in 1:nblk
                b = Random.rand(rng, 1:nblocks)
                starts[t,k] = (b-1)*blk + 1
            end

        elseif method == "circular" || method == "cbb"
            for k in 1:nblk
                starts[t,k] = Random.rand(rng, 1:N)
            end

        else
            JobLoggerTools.error_benji("Unknown method = $method")
        end
    end
    return starts
end

"""
    ensure_plan(
        provided::Union{Nothing, AbstractMatrix{<:Integer}},
        rng::Random.AbstractRNG,
        N::Int, 
        block::Int, 
        nbs::Int; 
        method::String
    ) -> (Union{Nothing, Matrix{Int}}, Int, Int)

Ensure a valid block-start plan matrix for the given configuration. If `provided`
is missing or has incompatible shape, allocate and (re)generate a plan.

# Arguments
- `provided::Union{Nothing, AbstractMatrix{<:Integer}}` : Optional existing plan
  of shape `nbs` ``\\times`` `nblk`.
- `rng::Random.AbstractRNG`                             : RNG instance.
- `N::Int`                                             : Population size.
- `block::Int`                                         : Block size.
- `nbs::Int`                                           : Number of bootstrap replicates (rows).
- `method::String`                                     : Plan generation method
  (as in [`gen_block_plan!`](@ref)).

# Behavior
- Computes [`(nblk, lastlen) = nblk_lastlen(N, block)`](@ref nblk_lastlen).
- If `N == 0`, returns `(nothing, nblk, lastlen)`.
- If `provided` is `nothing` or not of size `nbs` ``\\times`` `nblk`, a new `starts`
  is allocated and filled via [`gen_block_plan!`](@ref); otherwise returns `provided`.

# Returns
- `(starts, nblk, lastlen)` where:
  - `starts::Union{Nothing, Matrix{Int}}` : Valid plan (or `nothing` if `N==0`).
  - `nblk::Int`                            : Number of blocks per replicate.
  - `lastlen::Int`                          : Length of the final (possibly short) block.
"""
function ensure_plan(
    provided::Union{Nothing, AbstractMatrix{<:Integer}},
    rng::Random.AbstractRNG,
    N::Int, 
    block::Int, 
    nbs::Int; 
    method::String
)
    nblk, last = nblk_lastlen(N, block)
    if N == 0
        return nothing, nblk, last
    end
    if provided === nothing || size(provided,1) != nbs || size(provided,2) != nblk
        starts = Matrix{Int}(undef, nbs, nblk)
        gen_block_plan!(starts, rng, N, block, nblk; method=method)
        return starts, nblk, last
    else
        return provided, nblk, last
    end
end

"""
    mean_from_plan(
        ps::Union{Nothing,AbstractVector{<:Real}},
        arr::AbstractVector{<:Real},
        starts::Union{Nothing,AbstractMatrix{<:Integer}},
        N::Int, 
        blk::Int, 
        nblk::Int, 
        lastlen::Int,
        ibs::Int; 
        method::String
    ) -> Real

Compute the block-resampled mean for a single bootstrap replicate `ibs` using a
pre-generated start-index plan.

# Arguments
- `ps::Union{Nothing,AbstractVector{<:Real}}`:
  Optional prefix-sum array of `arr` with length `N+1`, where `ps[k] = sum(arr[1:k-1])`.
  **Required** for `"moving"`, `"nonoverlapping"` methods.
  Ignored for `"circular"` and for the trivial case `blk == 1`.
- `arr::AbstractVector{<:Real}`:
  Source data of length `N` to be block-resampled.
- `starts::Union{Nothing,AbstractMatrix{<:Integer}}`:
  Block start indices of shape `nbs` ``\\times`` `nblk` (``1``-based).
  Row `ibs` encodes the sequence of block starts for the replicate.
  May be `nothing` when `N == 0`.
- `N::Int`:
  Population size, i.e. `length(arr)`.
- `blk::Int`:
  Nominal block size.
- `nblk::Int`:
  Number of blocks per replicate (including the final possibly short block).
- `lastlen::Int`:
  Length of the final block (may be `0` if the last block is absent;
  otherwise `1` ``\\le`` `lastlen` ``\\le`` `blk`).
- `ibs::Int`:
  ``1``-based index of the bootstrap replicate row to use from `starts`.
- `method::String`:
  Block scheme. One of `"moving"`, `"nonoverlapping"` (uses `ps`),
  or `"circular"` (wrap-around modulo indexing).

# Behavior
- Early exit: if `N == 0` **or** `starts === nothing` **or** `nblk == 0`, returns `0.0`.
- If `blk == 1`, gathers `nblk` elements directly at `starts[ibs, k]` and returns
  their average divided by `N`.
- For `"moving" | "nonoverlapping"`:
  uses prefix sums to accumulate `(nblk-1)` full blocks of length `blk` and, if
  `lastlen > 0`, one final partial block of length `lastlen`. Returns the total
  divided by `N`. Requires `ps !== nothing`.
- For `"circular"`:
  sums each block by explicit modulo indexing
  `i = (s + j - 1) % N + 1`, first for the `(nblk-1)` full blocks of length `blk`,
  then the optional final block of length `lastlen`. Returns the total divided by `N`.
- Throws `JobLoggerTools.error_benji("Unknown method = \$method")` for unrecognized methods.

# Returns
- `mean::Real`: The resampled mean for replicate `ibs`, normalized by `N`.

# Notes
- Contracts/assumptions:
  - `length(arr) == N`.
  - `starts` has shape `nbs` ``\\times`` `nblk`, and `ibs` ``\\in`` `1:nbs`.
  - Start indices are ``1``-based and valid for the chosen method.
  - For prefix-sum methods, `ps` must satisfy `length(ps) == N+1`.
- Complexity:
  - `O(nblk)` for prefix-sum methods (`moving/nonoverlapping`);
  - `O(nblk * blk)` for circular methods.
- Performance:
  Uses [`Base.@view`](https://docs.julialang.org/en/v1/base/arrays/#Base.@view) for `srow = starts[ibs, :]` and [`Base.@inbounds`](https://docs.julialang.org/en/v1/base/base/#Base.@inbounds) in inner loops.
"""
@inline function mean_from_plan(
    ps::Union{Nothing,AbstractVector{<:Real}},
    arr::AbstractVector{<:Real},
    starts::Union{Nothing,AbstractMatrix{<:Integer}},
    N::Int, 
    blk::Int, 
    nblk::Int, 
    lastlen::Int,
    ibs::Int; 
    method::String
)
    (N==0 || starts===nothing || nblk==0) && return 0.0
    srow = @view starts[ibs, :]

    if blk == 1
        acc = 0.0
        @inbounds for k in 1:nblk
            acc += arr[srow[k]]
        end
        return acc / N
    elseif method == "moving" || method == "nonoverlapping"
        JobLoggerTools.assert_benji(
            ps !== nothing, 
            "ps must not be nothing"
        )
        acc = 0.0
        @inbounds for k in 1:(nblk-1)
            s = srow[k]; e = s + blk - 1
            acc += ps[e+1] - ps[s]
        end
        if nblk>=1 && lastlen>0
            s = srow[nblk]; e = s + lastlen - 1
            acc += ps[e+1] - ps[s]
        end
        return acc / N
    elseif method == "circular"
        acc = 0.0
        @inbounds for k in 1:(nblk-1)
            s = srow[k]
            for j in 0:blk-1
                i = (s + j - 1) % N + 1
                acc += arr[i]
            end
        end
        if nblk>=1 && lastlen>0
            s = srow[nblk]
            for j in 0:lastlen-1
                i = (s + j - 1) % N + 1
                acc += arr[i]
            end
        end
        return acc / N
    else
        JobLoggerTools.error_benji("Unknown method = $method")
    end
end

"""
    const Plan4{T}

Type alias for a 4-tuple of vectors holding per-sample quantities
(e.g., cumulants or their components):

- `Plan4{T} == NTuple{4, AbstractVector{T}}`

# Typical usage:
- `Q_tuple::Plan4{Float64}`: four length-`N` vectors containing raw values
  (e.g., ``Q_n \\; (n=1,2,3,4)``) indexed from `1` to `N`.
- `ps_tuple::Plan4{Float64}`: four length-`N+1` prefix-sum vectors corresponding
  to the above (e.g., `ps[k] = sum(Q[1:k-1])` with `ps[1] = 0.0`).

# Notes
- This alias is used to express the interface of batching functions that
  consume four related series of equal length (for `Q_tuple`) or their
  prefix sums (for `ps_tuple`).
"""
const Plan4{T} = NTuple{4, AbstractVector{T}}

"""
    update_mean_from_plan4!(
        Q_tuple::Plan4{Float64},
        ps_tuple::Plan4{Float64},
        starts::Union{Nothing, AbstractMatrix{<:Integer}},
        N::Int, blk::Int, nblk::Int, lastlen::Int,
        outmat::Matrix{Float64}, ibs::Int;
        method::String = "nonoverlapping"
    ) -> nothing

Accumulate block-averaged means for four related series ``(Q_1, Q_2, Q_3, Q_4)``
into `outmat[:, ibs]`, using either direct indexing or prefix sums depending
on the block-sampling method.

# Arguments
- `Q_tuple`: `Plan4{Float64}` of raw series. Each vector must have length `N`.
- `ps_tuple`: `Plan4{Float64}` of prefix sums for the corresponding raw series.
  Each vector must have length `N + 1` and satisfy `ps[k+1] - ps[k] == Q[k]`.
- `starts`: Either `nothing` (degenerate case) or an integer matrix where row `ibs`
  contains ``1``-based start indices of each block (size ≥ `nblk`).
- `N`: Total number of samples per series.
- `blk`: Nominal block length (for all but possibly the last block).
- `nblk`: Number of blocks (including the possibly shorter last block).
- `lastlen`: Length of the last block; used when the last block is shorter than `blk`.
- `outmat`: Output matrix of size at least `4 × ?`; column `ibs` is overwritten with
  the four accumulated means (one per row).
- `ibs`: Column index in `outmat` to write results into.
- `method`: One of `"nonoverlapping"`, `"moving"`, `"nbb"` (prefix-sum paths),
  or `"circular"`, `"cbb"` (circular modular indexing). Any other string throws.

# Behavior
- If `N == 0`, `starts === nothing`, or `nblk == 0`, the function writes zeros
  to `outmat[1:4, ibs]` and returns.
- For `"nonoverlapping"`, `"moving"`, `"nbb"`:
  - Accumulation uses prefix sums for ``O(1)`` block-sum queries:
    `sum(Q[s:e]) == ps[e+1] - ps[s]`.
- For `"circular"`, `"cbb"`:
  - Accumulation uses modular indexing over `Q_tuple`:
    `i = (s + j - 1) % N + 1` for `j = 0:(len-1)`.
- The final four accumulators are divided by `N` and stored in `outmat[1:4, ibs]`.

# Requirements / Assumptions
- `length.(Q_tuple) == (N, N, N, N)`
- `length.(ps_tuple) == (N+1, N+1, N+1, N+1)`
- `starts[ibs, k]` is valid for `k = 1:nblk`, and each block range is within `1:N`
  after applying the chosen method’s indexing rule.
- `outmat` has at least 4 rows and a valid column `ibs`.

# Returns
- `Nothing` (mutates `outmat` in place).

# Complexity
- `"nonoverlapping"`, `"moving"`, `"nbb"`: `O(nblk)` due to prefix-sum use.
- `"circular"`, `"cbb"`: `O(nblk * blk)` (or with `lastlen` for the final block).

# Throws
- `JobLoggerTools.error_benji("Unknown method = \$method")` if `method` is not one of the recognized options.

# Example
```julia
Q1, Q2, Q3, Q4 = CumulantsBundleUtils.flatten_Q4_columns(Q_Y_ORG)  # each length N
ps = (Bootstrap.prefix_sums(Q1), Bootstrap.prefix_sums(Q2),
      Bootstrap.prefix_sums(Q3), Bootstrap.prefix_sums(Q4))        # each length N+1

out = zeros(4, nboots)
update_mean_from_plan4!((Q1, Q2, Q3, Q4), ps, starts_all, N, blk, nblk, lastlen,
                        out, ibs; method="nonoverlapping")
```

# Notes

- For robustness, you may add:

```julia
JobLoggerTools.assert_benji(
    all(length.(Q_tuple) .== N),
    "length.(Q_tuple) must all equal N"
)
JobLoggerTools.assert_benji(all(length.(ps_tuple) .== N .+ 1),
    "length.(ps_tuple) must all equal N+1"
)
```

near the top (disabled in production if needed).
"""
@inline function update_mean_from_plan4!(
    Q_tuple::Plan4{Float64}, 
    ps_tuple::Plan4{Float64}, 
    starts::Union{Nothing, AbstractMatrix{<:Integer}},
    N::Int, 
    blk::Int, 
    nblk::Int, 
    lastlen::Int,
    outmat::Matrix{Float64}, 
    ibs::Int; 
    method::String="nonoverlapping"
)
    if N == 0 || starts === nothing || nblk == 0
        @inbounds outmat[1,ibs]=0.0; outmat[2,ibs]=0.0; outmat[3,ibs]=0.0; outmat[4,ibs]=0.0
        return
    end
    ps1, ps2, ps3, ps4 = ps_tuple
    srow = @view starts[ibs, :]
    acc1=0.0; acc2=0.0; acc3=0.0; acc4=0.0

    if blk == 1
        @inbounds for k in 1:nblk
            i = srow[k]
            acc1 += Q_tuple[1][i]; acc2 += Q_tuple[2][i]
            acc3 += Q_tuple[3][i]; acc4 += Q_tuple[4][i]
        end
    elseif method == "moving" || method == "nonoverlapping" || method == "nbb"
        @inbounds for k in 1:(nblk-1)
            s = srow[k]; e = s + blk - 1
            acc1 += ps1[e+1]-ps1[s]; acc2 += ps2[e+1]-ps2[s]
            acc3 += ps3[e+1]-ps3[s]; acc4 += ps4[e+1]-ps4[s]
        end
        if nblk >= 1 && lastlen > 0
            s = srow[nblk]; e = s + lastlen - 1
            acc1 += ps1[e+1]-ps1[s]; acc2 += ps2[e+1]-ps2[s]
            acc3 += ps3[e+1]-ps3[s]; acc4 += ps4[e+1]-ps4[s]
        end
    elseif method == "circular" || method == "cbb"
        @inbounds for k in 1:(nblk-1)
            s = srow[k]
            for j in 0:blk-1
                i = (s + j - 1) % N + 1
                acc1 += Q_tuple[1][i]; acc2 += Q_tuple[2][i]
                acc3 += Q_tuple[3][i]; acc4 += Q_tuple[4][i]
            end
        end
        if nblk >= 1 && lastlen > 0
            s = srow[nblk]
            for j in 0:lastlen-1
                i = (s + j - 1) % N + 1
                acc1 += Q_tuple[1][i]; acc2 += Q_tuple[2][i]
                acc3 += Q_tuple[3][i]; acc4 += Q_tuple[4][i]
            end
        end
    else
        JobLoggerTools.error_benji("Unknown method = $method")
    end

    invN = (N == 0 ? 0.0 : 1.0 / N)
    @inbounds begin
        outmat[1,ibs] = acc1 * invN
        outmat[2,ibs] = acc2 * invN
        outmat[3,ibs] = acc3 * invN
        outmat[4,ibs] = acc4 * invN
    end
end

"""
    Plan5{T} = NTuple{5, AbstractVector{T}}

Type alias for a 5-tuple of vectors holding per-sample series, typically
``(Q_1, Q_2, Q_3, Q_4, w)`` where ``w`` is a reweighting factor.

- `Q_tuple::Plan5{Float64}`: five length-`N` vectors of raw values.
- `ps_tuple::Plan5{Float64}`: five length-`N+1` prefix-sum vectors where
  `ps[k+1] - ps[k] == Q[k]` and `ps[1] == 0.0`.

This alias makes signatures concise for functions that operate on four related
series plus a weight track.
"""
const Plan5{T} = NTuple{5, AbstractVector{T}}

"""
    update_mean_from_plan!(
        Q_tuple::Plan5{Float64},
        ps_tuple::Plan5{Float64},
        starts::Matrix{Int},
        N::Int,
        blk::Int,
        nblk::Int,
        lastlen::Int,
        outmat::Matrix{Float64},
        ibs::Int;
        method::String = "nonoverlapping",
    ) -> nothing

Accumulate block-averaged means for five related series ``(Q_1, Q_2, Q_3, Q_4, w)``
into `outmat[1:5, ibs]`, using either direct indexing (`blk == 1` or `"circular"`)
or prefix sums (`"nonoverlapping"` / `"moving"`).

# Arguments
- `Q_tuple`: `Plan5{Float64}` of raw series; each vector must have length `N`.
- `ps_tuple`: `Plan5{Float64}` of prefix sums corresponding to `Q_tuple`;
  each vector must have length `N+1` with `ps[k+1] - ps[k] == Q[k]`.
- `starts`: Matrix of ``1``-based start indices for each block; row `ibs` is used.
- `N`: Total number of samples per series.
- `blk`: Nominal block length (except possibly the last).
- `nblk`: Number of blocks (including the possibly shorter last block).
- `lastlen`: Length of the last block when shorter than `blk`.
- `outmat`: Output matrix with at least 5 rows; column `ibs` is overwritten.
- `ibs`: Column index into `outmat` to write results.
- `method`: `"nonoverlapping"` / `"moving"` (prefix-sum path) or `"circular"`
  (modular direct indexing). Any other string throws an error.

# Behavior
- For `blk == 1`: sums the point values directly at `starts[ibs, k]`.
- For `"nonoverlapping"` / `"moving"`: uses prefix sums for ``O(1)`` block sums,
  i.e., `sum(Q[s:e]) == ps[e+1] - ps[s]`.
- For `"circular"`: uses modular indexing `i = (s + j - 1) % N + 1` for block spans.
- Final accumulators are divided by `N` (assumed total length) and stored in
  `outmat[1:5, ibs]`.

# Assumptions
- `length.(Q_tuple) == (N, N, N, N, N)`
- `length.(ps_tuple) == (N+1, N+1, N+1, N+1, N+1)`
- `starts[ibs, 1:nblk]` are valid start indices under the chosen method.
- The denominator is `N` (total series length), not the total block length.

# Returns
- `Nothing` (mutates `outmat` in place).

# Throws
- `JobLoggerTools.error_benji("Unknown method = \$method")` for unsupported `method` values.
"""
function update_mean_from_plan!(
    Q_tuple::Plan5{Float64}, 
    ps_tuple::Plan5{Float64}, 
    starts::Matrix{Int},
    N::Int, 
    blk::Int, 
    nblk::Int, 
    lastlen::Int,
    outmat::Matrix{Float64}, 
    ibs::Int;
    method::String="nonoverlapping"
)
    ps1, ps2, ps3, ps4, psw = ps_tuple
    srow = @view starts[ibs, :]

    acc1=0.0; acc2=0.0; acc3=0.0; acc4=0.0; accw=0.0

    if blk == 1
        @inbounds for k in 1:nblk
            i = srow[k]
            acc1 += Q_tuple[1][i]; acc2 += Q_tuple[2][i]
            acc3 += Q_tuple[3][i]; acc4 += Q_tuple[4][i]; accw += Q_tuple[5][i]
        end

    elseif method == "moving" || method == "nonoverlapping"
        @inbounds for k in 1:(nblk-1)
            s = srow[k]; e = s + blk - 1
            acc1 += ps1[e+1]-ps1[s]; acc2 += ps2[e+1]-ps2[s]
            acc3 += ps3[e+1]-ps3[s]; acc4 += ps4[e+1]-ps4[s]; accw += psw[e+1]-psw[s]
        end
        if nblk >= 1 && lastlen > 0
            s = srow[nblk]; e = s + lastlen - 1
            acc1 += ps1[e+1]-ps1[s]; acc2 += ps2[e+1]-ps2[s]
            acc3 += ps3[e+1]-ps3[s]; acc4 += ps4[e+1]-ps4[s]; accw += psw[e+1]-psw[s]
        end

    elseif method == "circular"
        @inbounds for k in 1:(nblk-1)
            s = srow[k]
            for j in 0:blk-1
                i = (s + j - 1) % N + 1
                acc1 += Q_tuple[1][i]; acc2 += Q_tuple[2][i]
                acc3 += Q_tuple[3][i]; acc4 += Q_tuple[4][i]; accw += Q_tuple[5][i]
            end
        end
        if nblk >= 1 && lastlen > 0
            s = srow[nblk]
            for j in 0:lastlen-1
                i = (s + j - 1) % N + 1
                acc1 += Q_tuple[1][i]; acc2 += Q_tuple[2][i]
                acc3 += Q_tuple[3][i]; acc4 += Q_tuple[4][i]; accw += Q_tuple[5][i]
            end
        end

    else
        JobLoggerTools.error_benji("Unknown method = $method")
    end

    # assuming total_len = N
    outmat[1,ibs] = acc1 / N
    outmat[2,ibs] = acc2 / N
    outmat[3,ibs] = acc3 / N
    outmat[4,ibs] = acc4 / N
    outmat[5,ibs] = accw / N
end

"""
    bootstrap_average_error(
        arr::AbstractArray
    ) -> (Real, Real)

Compute the mean and standard deviation (as bootstrap error estimate)
from an array of bootstrap samples.

# Arguments
- `arr::AbstractArray`: Vector of bootstrap samples

# Returns
A tuple:
- `m`: Mean of samples
- `s`: Standard deviation with Bessel's correction
"""
function bootstrap_average_error(
    arr::AbstractArray
)
    if isempty(arr)
        return 0.0, 0.0
    end
    m = Statistics.mean(arr)
    v = Statistics.var(arr, corrected = true)  # Use unbiased estimator
    s = sqrt(v)
    return m, s
end

"""
    normalize_method(
        m::AbstractString
    ) -> String

Normalize bootstrap/blocking method aliases to canonical names.

# Arguments
- `m::AbstractString`:
  Input method name or alias. Recognized aliases:
  - `"nbb"` → `"nonoverlapping"`
  - `"cbb"` → `"circular"`

# Behavior
- Returns the canonical method string if a known alias is provided.
- Otherwise returns `String(m)` as-is.

# Returns
- `method::String`: Canonical method name.

# Examples
```julia
normalize_method("nbb") == "nonoverlapping"  # true
normalize_method("moving") == "moving"       # true
```
"""
@inline normalize_method(
    m::AbstractString
) =
    m == "nbb" ? "nonoverlapping" :
    m == "cbb" ? "circular"      : String(m)

"""
    bootstrap_average_error_from_raw(
        arr::AbstractVector,
        N_bs::Int,
        block::Int,
        rng::Random.AbstractRNG;
        method::String = "nonoverlapping"
    ) -> (Real, Real)

Estimate the mean and its bootstrap error from a single raw series `arr`
using block bootstrap with a given block size and method.

This routine builds (or reuses) a block-start plan, computes the resampled mean
for each bootstrap replicate via [`mean_from_plan`](@ref), and finally aggregates the
bootstrap estimate and its standard error.

# Arguments
- `arr::AbstractVector`:
  Raw data of length `N_all`. Must be non-empty.
- `N_bs::Int`:
  Number of bootstrap replicates.
- `block::Int`:
  Block size (must satisfy `1 ≤ block ≤ length(arr)`).
- `rng::Random.AbstractRNG`:
  RNG used to generate the block-start plan.
- `method::String` (keyword; default `"nonoverlapping"`):
  Blocking scheme. Canonical names:
  - `"moving"`: moving blocks (uses prefix sums)
  - `"nonoverlapping"`: non-overlapping blocks (uses prefix sums)
  - `"circular"`: circular blocks with wrap-around (no prefix sums)
  Aliases are normalized by `normalize_method`:
  `"nbb" → "nonoverlapping"`, `"cbb" → "circular"`.

# Behavior
- Validates inputs: non-empty `arr`; `block` within `[1, length(arr)]`.
- Normalizes `method` via [`normalize_method`](@ref).
- Builds [`ps = prefix_sums(arr)`](@ref prefix_sums) for prefix-sum methods; uses `nothing` for `"circular"`.
- Obtains `(starts, nblk, lastlen)` from [`ensure_plan(...)`](@ref ensure_plan).
- For each `ibs` ``\\in`` `1:N_bs`, computes the replicate mean by
  [`mean_from_plan(ps, arr, starts, N_all, block, nblk, lastlen, ibs; method)`](@ref mean_from_plan).
- Aggregates [`(m, s) = bootstrap_average_error(mean_arr)`](@ref bootstrap_average_error) and returns them.

# Returns
- `(m, s) :: (Real, Real)`:
  - `m`: Bootstrap estimate of the mean of `arr`.
  - `s`: Bootstrap standard error of the mean.

# Notes
- Contracts:
  - `length(arr) == N_all`.
  - [`ensure_plan`](@ref) must produce `starts::Matrix{Int}` of size `N_bs` ``\\times`` `nblk`
    with valid ``1``-based start indices for the selected method.
  - For prefix-sum methods, `ps` must satisfy `length(ps) == N_all + 1`.
- Complexity:
  - Plan generation: depends on [`ensure_plan`](@ref).
  - Per replicate: `O(nblk)` for prefix-sum methods; `O(nblk * block)` for circular.
- Performance:
  - Uses [`Base.@inbounds`](https://docs.julialang.org/en/v1/base/base/#Base.@inbounds) in the replicate loop.
  - The heavy lifting is delegated to [`mean_from_plan`](@ref), keeping logic de-duplicated.

# Examples
```julia
m, s = bootstrap_average_error_from_raw(
    randn(10_000), 
    1000, 
    50, 
    MersenneTwister();
    method="nbb"  # alias normalized to "nonoverlapping"
)
```
"""
function bootstrap_average_error_from_raw(
    arr::AbstractVector,
    N_bs::Int,
    block::Int,
    rng::Random.AbstractRNG;
    method::String = "nonoverlapping"
)
    N_all = length(arr)
    N_all == 0     && JobLoggerTools.error_benji("arr is empty")
    block < 1      && JobLoggerTools.error_benji("block must be ≥ 1")
    block > N_all  && JobLoggerTools.error_benji("block must be ≤ length(arr)")

    # --- prep ---
    method = normalize_method(method)
    ps = (method == "circular") ? nothing : prefix_sums(arr)
    starts, nblk, lastlen = ensure_plan(nothing, rng, N_all, block, N_bs; method=method)

    mean_arr = Vector{Float64}(undef, N_bs)

    @inbounds for ibs in 1:N_bs
        mean_arr[ibs] = mean_from_plan(ps, arr, starts, N_all, block, nblk, lastlen, ibs; method=method)
    end

    m, s = bootstrap_average_error(mean_arr)
    return m, s
end

end  # module Bootstrap