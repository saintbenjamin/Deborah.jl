# ============================================================================
# src/DeborahCore/DatasetPartitionerEsther.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DatasetPartitionerEsther

import ..Sarah.DatasetPartitioner
import ..TOMLConfigEsther

"""
    infer_partition_info_from_trace(
        trace_data::Dict
    ) -> DatasetPartitionInfo

Infers minimal partition info from already-loaded trace data dictionary,
usually when running in analysis-only mode (without rebuilding partitions).

# Arguments
- `trace_data::Dict` : Dictionary holding keys like `YP_tr`, `YP_bc`, `YP_ul`.

# Returns
- [`DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo) : Struct with inferred `N_tr`, `N_bc`, `N_ul`, and `N_cnf`.
"""
function infer_partition_info_from_trace(
    trace_data::Dict
)::DatasetPartitioner.DatasetPartitionInfo
    N_tr = haskey(trace_data, "Y_tr") && !isempty(trace_data["Y_tr"]) ? length(trace_data["Y_tr"][1]) : 0
    N_bc = haskey(trace_data, "Y_bc") && !isempty(trace_data["Y_bc"]) ? length(trace_data["Y_bc"][1]) : 0
    N_ul = haskey(trace_data, "Y_ul") && !isempty(trace_data["Y_ul"]) ? length(trace_data["Y_ul"][1]) : 0

    N_lb  = N_tr + N_bc

    N_cnf = haskey(trace_data, "Y_info") && !isempty(trace_data["Y_info"]) ? length(trace_data["Y_info"][1]) : 0

    return DatasetPartitioner.DatasetPartitionInfo(
        0, 
        N_cnf, 
        0,
        N_lb,  
        N_tr,
        0, 
        0,
        N_bc, 
        0,
        N_ul, 
        0,
        Int[], 
        Int[], 
        Int[], 
        Int[]
    )
end

end