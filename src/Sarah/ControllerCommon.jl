# ============================================================================
# src/Sarah/ControllerCommon.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module ControllerCommon

import TOML

"""
    save_toml_file(
        toml_dict::Dict, 
        path::String
    ) -> Nothing

Saves a [`TOML`](https://toml.io/en/) dictionary to a specified file path.

# Arguments
- `toml_dict::Dict`   : Dictionary containing configuration data to be saved.
- `path::String`      : Destination file path for the [`TOML`](https://toml.io/en/) output.

# Behavior
- Opens the file at `path` in write mode.
- Writes the contents of `toml_dict` in [`TOML`](https://toml.io/en/) format.

# Returns
- `Nothing` : Side-effect function.
"""
function save_toml_file(
    toml_dict::Dict, 
    path::String
)
    open(path, "w") do io
        TOML.print(io, toml_dict)
    end
end

end