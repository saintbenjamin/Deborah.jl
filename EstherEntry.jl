# ============================================================================
# EstherEntry.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

# ----------------------------------------------------------------------------
# For development: auto-reload on source file changes
# ----------------------------------------------------------------------------
# using Revise
import Deborah.DeborahEsther.DeborahEstherRunner; flush(stdout); flush(stderr)

# ----------------------------------------------------------------------------
# Entry point function: expects a TOML config path as first argument
# ----------------------------------------------------------------------------
function main()
    toml_path = get(ARGS, 1, nothing)
    if toml_path === nothing
        error("Usage: julia EstherEntry.jl path/to/config.toml")
        flush(stderr)
    end
    DeborahEstherRunner.run_Deborah_Esther(toml_path); flush(stdout); flush(stderr)
end

# ----------------------------------------------------------------------------
# Launch the program with benchmarking output
# ----------------------------------------------------------------------------
@time main()