# ============================================================================
# src/Sarah/DatasetPartitioner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DatasetPartitioner

import ..JobLoggerTools

"""
    struct DatasetPartitionInfo

Holds full information about how configurations are partitioned into
labeled, training, bias-correction, and unlabeled subsets.

# Fields
- `M_tot::Int`         : Total number of data rows (`N_cnf` ``\\times`` `N_src`).
- `N_cnf::Int`         : Number of configurations.
- `N_src::Int`         : Number of source vectors per configuration (``1`` if not applicable).

- `N_lb::Int`          : Number of labeled set.
- `N_tr::Int`          : Number of training set.
- `N_lb_src::Int`      : Number of total labeled set (`N_lb` ``\\times`` `N_src`).
- `N_tr_src::Int`      : Number of total training set (`N_tr` ``\\times`` `N_src`).

- `N_bc::Int`          : Number of bias-correction set (`N_lb - N_tr`).
- `N_bc_persrc::Int`   : Number of bias correction set per source.
- `N_ul::Int`          : Number of unlabeled set (`N_cnf - N_lb`).
- `N_ul_persrc::Int`   : Number of unlabeled set per source (same as above).

- `lb_idx::Vector{Int}` : Indices used for labeled set (length `N_lb`).
- `tr_idx::Vector{Int}` : Subset of `lb_idx` used for training.
- `bc_idx::Vector{Int}` : Complement of `tr_idx` within `lb_idx`.
- `ul_idx::Vector{Int}` : Complement of `lb_idx` within all configurations.
"""
struct DatasetPartitionInfo
    M_tot::Int
    N_cnf::Int
    N_src::Int

    N_lb::Int
    N_tr::Int
    N_lb_src::Int
    N_tr_src::Int

    N_bc::Int
    N_bc_persrc::Int
    N_ul::Int
    N_ul_persrc::Int

    lb_idx::Vector{Int}
    tr_idx::Vector{Int}
    bc_idx::Vector{Int}
    ul_idx::Vector{Int}
end

"""
    gen_set_idx(
        N_cnf::Int, 
        N_lb::Int, 
        N_tr::Int, 
        N_bc::Int, 
        N_ul::Int,
        IDX_shift::Int, 
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}}

Generates index vectors for labeled (`lb`), training (`tr`), bias-correction (`bc`), and unlabeled (`ul`)
configurations in a deterministic but shiftable way.

# Arguments
- `N_cnf::Int`       : Total number of configurations.
- `N_lb::Int`        : Number of labeled set.
- `N_tr::Int`        : Number of training set (subset of labeled).
- `N_bc::Int`        : Number of bias correction set.
- `N_ul::Int`        : Number of unlabeled set.
- `IDX_shift::Int`  : Configuration shift amount.
- `jobid`            : Optional job logging tag.

# Returns
- Tuple of `(lb_idx, tr_idx, bc_idx, ul_idx)` as vectors of `Int`.
"""
function gen_set_idx(
    N_cnf::Int, 
    N_lb::Int, 
    N_tr::Int, 
    N_bc::Int, 
    N_ul::Int,
    IDX_shift::Int, 
    jobid::Union{Nothing, String}=nothing
)::Tuple{Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}}

    lb_idx = Int.(zeros(N_lb))
    lb_jump = N_cnf / N_lb
    for ilb in 1:N_lb
        raw_idx = floor((ilb - 1) * lb_jump + IDX_shift) % N_cnf + 1
        lb_idx[ilb] = Int(raw_idx)
    end

    tr_idx = Int[]
    if N_tr > 0
        tr_idx = Int.(zeros(N_tr))
        tr_jump = N_lb / N_tr
        for itr in 1:N_tr
            tr_idx[itr] = floor((itr - 1) * tr_jump + 1)
        end
    end

    bc_idx = Int[]
    if N_bc > 0
        bc_idx = Int.(zeros(N_bc))
        itr = 1
        ibc = 1
        for ilb in 1:N_lb
            if N_tr > 0 && itr <= N_tr && ilb == tr_idx[itr]
                if itr < N_tr
                    itr += 1
                end
            else
                bc_idx[ibc] = ilb
                if ibc < N_bc
                    ibc += 1
                end
            end
        end
    end


    lb_set = Set(lb_idx)
    ul_idx = Int[]
    for iconf in 1:N_cnf
        if iconf ∉ lb_set
            push!(ul_idx, iconf)
            if length(ul_idx) == N_ul
                break
            end
        end
    end

    sort!(lb_idx)
    sort!(ul_idx)

    actual_shift = find_equivalent_shift(lb_idx, N_cnf, N_lb, IDX_shift)
    if actual_shift !== IDX_shift
        JobLoggerTools.warn_benji("IDX_shift = $IDX_shift is equivalent to IDX_shift = $actual_shift (modulo sampling symmetry).", jobid)
    end

    return lb_idx, tr_idx, bc_idx, ul_idx
end

"""
    find_equivalent_shift(
        lb_idx::Vector{Int},
        N_cnf::Int,
        N_lb::Int,
        shift_given::Int
    ) -> Union{Int, Nothing}

Given a set of labeled indices, determine what shift amount would
generate the same set via modular sampling logic.

# Arguments
- `lb_idx::Vector{Int}` : Reference labeled indices.
- `N_cnf::Int`          : Total number of configurations.
- `N_lb::Int`           : Number of labeled configs.
- `shift_given::Int`    : Initial shift amount to test first.

# Returns
- `Int` if a matching shift is found, otherwise `nothing`.
"""
function find_equivalent_shift(
    lb_idx::Vector{Int},
    N_cnf::Int,
    N_lb::Int,
    shift_given::Int
)::Union{Int, Nothing}
    lb_jump = N_cnf / N_lb
    lb_idx_sorted = sort(lb_idx)

    if 0 <= shift_given < N_cnf
        lb_idx_given = [Int(floor((ilb - 1) * lb_jump + shift_given) % N_cnf + 1) for ilb in 1:N_lb]
        sort!(lb_idx_given)
        if lb_idx_given == lb_idx_sorted
            return shift_given
        end
    end

    for shift_candidate in 0:(N_cnf - 1)
        lb_idx_candidate = [Int(floor((ilb - 1) * lb_jump + shift_candidate) % N_cnf + 1) for ilb in 1:N_lb]
        sort!(lb_idx_candidate)
        if lb_idx_candidate == lb_idx_sorted
            return shift_candidate
        end
    end
    return nothing
end

end  # module DatasetPartitioner