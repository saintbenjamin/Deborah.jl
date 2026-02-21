# ============================================================================
# src/DeborahCore/DatasetPartitionerDeborah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DatasetPartitionerDeborah

import ..Sarah.DatasetPartitioner
import ..TOMLConfigDeborah

"""
    partition_dataset(
        path::String, 
        data::TOMLConfigDeborah.TraceDataConfig, 
        jobid::Union{Nothing, String}=nothing
    ) -> DatasetPartitioner.DatasetPartitionInfo

Reads the number of configurations from the target file and computes the
partitioning of configurations into labeled set, training set, bias-correction
set, and unlabeled set.

# Arguments
- `path::String` : Path to the directory containing trace files.
- [`data::TOMLConfigDeborah.TraceDataConfig`](@ref Deborah.DeborahCore.TOMLConfigDeborah.TraceDataConfig) : Full trace configuration data.
- `jobid::Union{Nothing, String}` : Optional logging tag.

# Returns
- [`Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo`](@ref) : Struct holding partition sizes and index vectors.
"""
function partition_dataset(
    path::String,
    data::TOMLConfigDeborah.TraceDataConfig, 
    jobid::Union{Nothing, String}=nothing
)::DatasetPartitioner.DatasetPartitionInfo
    N_lb = data.LBP
    N_tr = data.TRP
    obj  = data.Y
    IDX_shift = data.IDX_shift

    M_tot = countlines(joinpath(path, obj))
    N_cnf = countlines(joinpath(path, obj))
    N_src = div(M_tot, N_cnf)

    N_lb_percent = Float64(N_lb)
    N_tr_percent = Float64(N_tr)

    N_lb = Int(round(M_tot * N_lb_percent / 100))
    N_tr = Int(round(N_lb  * N_tr_percent / 100))

    N_lb_src    = N_lb     * N_src
    N_tr_src    = N_tr     * N_src
    N_bc        = N_lb_src - N_tr_src
    N_bc_persrc = N_lb     - N_tr
    N_ul        = M_tot    - N_lb_src
    N_ul_persrc = N_cnf    - N_lb

    lb_idx, tr_idx, bc_idx, ul_idx = 
    DatasetPartitioner.gen_set_idx(
        N_cnf, 
        N_lb, 
        N_tr, 
        N_bc, 
        N_ul, 
        IDX_shift, 
        jobid
    )

    return DatasetPartitioner.DatasetPartitionInfo(
        M_tot, 
        N_cnf, 
        N_src, 
        N_lb,  
        N_tr, 
        N_lb_src, 
        N_tr_src, 
        N_bc,  
        N_bc_persrc, 
        N_ul,  
        N_ul_persrc,
        lb_idx, 
        tr_idx, 
        bc_idx, 
        ul_idx
    )
end

end