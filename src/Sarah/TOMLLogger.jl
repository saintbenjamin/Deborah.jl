# ============================================================================
# src/Sarah/TOMLLogger.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module TOMLLogger

import ..OrderedCollections

"""
    append_section_to_toml(
        info_file::String,
        section::String,
        content::OrderedCollections.OrderedDict,
    ) -> Nothing

Append a new section to an existing [`TOML`](https://toml.io/en/) file.

# Arguments
- `info_file`: Path to the [`TOML`](https://toml.io/en/) file to which the section will be appended.
- `section`: Name of the [`TOML`](https://toml.io/en/) section (e.g., `"metadata"`).
- `content`: [`OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/latest/ordered_containers.html) of key-value pairs to be written under the section.

# Behavior
- Opens the file in append mode.
- Inserts a blank line, followed by a `[section]` header.
- Each key-value pair is written as `key = repr(value)` for [`TOML`](https://toml.io/en/) compatibility.
"""
function append_section_to_toml(
    info_file::String, 
    section::String, 
    content::OrderedCollections.OrderedDict
)
    open(info_file, "a") do io
        println(io, "")
        println(io, "[$section]")
        for (k, v) in content
            println(io, "$k = $(repr(v))")
        end
    end
end

end  # module TOMLLogger