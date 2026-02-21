# ============================================================================
# src/DeborahEstherMiriam/DeborahEstherMiriamRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module DeborahEstherMiriamRunner

import ..Sarah.JobLoggerTools
import ..Sarah.StringTranscoder
import ..Sarah.NameParser
import ..Sarah.ControllerCommon
import ..Miriam.MiriamRunner
import ..MiriamDependencyManager

"""
    run_Deborah_Esther_Miriam(
        toml_path::String
    ) -> Nothing

Full pipeline executor for [`Deborah.DeborahCore`](@ref) → [`Deborah.Esther`](@ref) → [`Deborah.Miriam`](@ref) process.
Ensures all necessary single-ensemble-level prerequisites are in place (including
``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` ML estimation by [`Deborah.DeborahCore`](@ref) if needed), then executes the final [`Deborah.Miriam`](@ref) routine.

# Arguments
- `toml_path::String` : Path to the [`TOML`](https://toml.io/en/) configuration file.

# Returns
- `Nothing` : All results are written to disk. Standard output and error are flushed.
"""
function run_Deborah_Esther_Miriam(
    toml_path::String
)
    # Ensure all ensemble-level prerequisites (TrM existence or regeneration)
    MiriamDependencyManager.ensure_ensemble_exists(toml_path)

    # Execute the core Miriam logic as final step
    MiriamRunner.run_Miriam(toml_path)
    flush(stdout)
    flush(stderr)
end

end  # module DeborahEstherMiriam