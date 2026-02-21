# ============================================================================
# src/DeborahEsther/DeborahEstherRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DeborahEstherRunner

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.ControllerCommon
import ..Esther.EstherRunner
import ..EstherDependencyManager

"""
    run_Deborah_Esther(
        toml_path::String
    ) -> Nothing

Runs the [`Deborah.DeborahCore`](@ref) → [`Deborah.Esther`](@ref) pipeline up to the [`Deborah.Esther`](@ref) stage.
Ensures that all required ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` results exist (or regenerates them),
then executes the core [`Deborah.Esther`](@ref) computation.

# Arguments
- `toml_path::String` : Path to the [`TOML`](https://toml.io/en/) configuration file.

# Returns
- `Nothing` : All outputs are written to disk. Standard output and error are flushed.
"""
function run_Deborah_Esther(
    toml_path::String
)
    # Ensure required trace files exist or are created
    EstherDependencyManager.ensure_TrM_exists(toml_path)

    # Run main computation for Esther
    EstherRunner.run_Esther(toml_path)
    flush(stdout)
    flush(stderr)
end

end  # module DeborahEsther