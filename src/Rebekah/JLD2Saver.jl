# ============================================================================
# src/Rebekah/JLD2Saver.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module JLD2Saver

import JLD2
import ..Sarah.JobLoggerTools

"""
    save_results(
        filepath::String,
        summary::Dict,
        labels::Vector{String},
        trains::Vector{String}
    ) → Nothing

Save analysis results and associated metadata into a single `.jld2` file.

# Arguments
- `filepath`: Path to the output `.jld2` file (e.g., `"results.jld2"`).
- `summary`: Dictionary containing final analysis output.
- `labels`: Vector of labeled set ratio strings (e.g., `"10"`, `"30"`)
- `trains`: Vector of training set ratio strings (e.g., `"10"`, `"30"`).

# Returns
- `Nothing`. Data is written directly to the specified file.
"""
function save_results(
    filepath::String,
    summary::Dict,
    labels::Vector{String},
    trains::Vector{String}
)::Nothing
    jobid = nothing
    JobLoggerTools.info_benji("Saving summary and metadata to $filepath ...", jobid)
    JLD2.jldsave(filepath;
        summary = summary,
        labels  = labels,
        trains  = trains
    )
    JobLoggerTools.info_benji("Save completed.", jobid)
end

end