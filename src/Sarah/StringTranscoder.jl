# ============================================================================
# src/Sarah/StringTranscoder.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module StringTranscoder

import ..JobLoggerTools

"""
    parse_string_dict(
        dict_any::Dict{String, Any}
    ) -> Dict{String, String}

Convert all values in a dictionary to strings.

# Arguments
- `dict_any`: Dictionary with string keys and arbitrary values.

# Returns
- `Dict{String, String}`: Dictionary where all values have been stringified.
"""
function parse_string_dict(
    dict_any::Dict{String,Any}
)::Dict{String,String}
    return Dict(k => String(v) for (k, v) in dict_any)
end

"""
    struct AbbreviationConfig

Configuration for encoding and decoding input names using abbreviations and numeric IDs.

# Fields
- `name_to_code`: Map from file name (e.g. `"plaq.dat"`) to abbreviation (e.g. `"Plaq"`).
- `code_to_name`: Reverse map from abbreviation to file name.
- `name_to_num`: Map from file name to integer ID.
- `num_to_name`: Map from integer ID to file name.
- `code_to_num`: Map from abbreviation to integer ID.
- `num_to_code`: Map from integer ID to abbreviation.
"""
struct AbbreviationConfig
    name_to_code::Dict{String, String}
    code_to_name::Dict{String, String}
    name_to_num::Dict{String, Int}
    num_to_name::Dict{Int, String}
    code_to_num::Dict{String, Int}
    num_to_code::Dict{Int, String}
end

"""
    abbreviation_map(
        letter_map::Dict{String, Any},
        jobid::Union{Nothing, String}=nothing
    ) -> AbbreviationConfig

Construct an [`AbbreviationConfig`](@ref) from a dictionary mapping names to abbreviations.

# Arguments
- `letter_map`: Dictionary mapping file names to abbreviation codes.
- `jobid::Union{Nothing, String}` : Optional job identifier for structured logging.

# Returns
- [`AbbreviationConfig`](@ref): Struct with complete forward/reverse maps and numeric IDs.

# Throws
- `ErrorException` if any value is not a string or if a code is empty.
"""
function abbreviation_map(
    letter_map::Dict{String, Any},
    jobid::Union{Nothing, String}=nothing
)::AbbreviationConfig
    # Filter only string values
    letter_map_str = Dict{String, String}()
    for (k, v) in letter_map
        if isa(v, String)
            letter_map_str[k] = v
        else
            JobLoggerTools.error_benji("Invalid abbreviation entry: $k => $v (not a String)", jobid)
        end
    end

    name_to_code = copy(letter_map_str)
    code_to_name = Dict{String, String}()
    name_to_num  = Dict{String, Int}()
    num_to_name  = Dict{Int, String}()
    code_to_num  = Dict{String, Int}()
    num_to_code  = Dict{Int, String}()

    i = 1
    for (name, code) in letter_map_str
        if isempty(code)
            JobLoggerTools.error_benji("Empty abbreviation code for name: $name", jobid)
        end
        code_to_name[code] = name
        name_to_num[name] = i
        num_to_name[i] = name
        code_to_num[code] = i
        num_to_code[i] = code
        i += 1
    end

    return AbbreviationConfig(
        name_to_code,
        code_to_name,
        name_to_num,
        num_to_name,
        code_to_num,
        num_to_code,
    )
end

"""
    input_encoder_abbrev(
        X::Vector{String},
        Y::String,
        abbrev_config::AbbreviationConfig,
        jobid::Union{Nothing, String}=nothing
    ) -> String

Encode a combination of `X` and `Y` input names into a hyphen-joined abbreviation string.

# Arguments
- `X`: Vector of input file names (e.g. `["plaq.dat", "rect.dat"]`).
- `Y`: Output file name (e.g. `"pbp.dat"`).
- `abbrev_config`: Abbreviation configuration struct.
- `jobid`: Optional job ID for logging.

# Returns
- `String`: Hyphen-separated encoded string (e.g. `"Plaq-Rect-TrM1"`).

# Throws
- [`JobLoggerTools.error_benji`](@ref) if any input has no abbreviation.
"""
function input_encoder_abbrev(
    X::Vector{String},
    Y::String,
    abbrev_config::AbbreviationConfig,
    jobid::Union{Nothing, String}=nothing
)::String

    all_names = vcat(X, Y)
    encoded = String[]

    for name in all_names
        if haskey(abbrev_config.name_to_code, name)
            push!(encoded, abbrev_config.name_to_code[name])
        else
            JobLoggerTools.error_benji("Warning: No abbreviation for name: $name", jobid)
        end
    end

    return join(encoded, "-")
end

"""
    input_decoder_abbrev(
        code_str::String, 
        abbrev_config::AbbreviationConfig, 
        jobid::Union{Nothing, String}=nothing
    ) -> String

Decode an abbreviation string back into the underscore-joined original file names.

# Arguments
- `code_str`: Hyphen-separated abbreviation string (e.g. `"Plaq-Rect-TrM1"`).
- `abbrev_config`: Abbreviation configuration struct.
- `jobid`: Optional job ID for logging.

# Returns
- `String`: Underscore-separated file names (e.g. `"plaq.dat_rect.dat_pbp.dat"`).

# Throws
- [`JobLoggerTools.error_benji`](@ref) if any code is not found in the reverse map.
"""
function input_decoder_abbrev(
    code_str::String, 
    abbrev_config::AbbreviationConfig, 
    jobid::Union{Nothing, String}=nothing
)::String

    codes = split(code_str, "-")
    result = String[]

    for code in codes
        if haskey(abbrev_config.code_to_name, code)
            push!(result, abbrev_config.code_to_name[code])
        else
            JobLoggerTools.error_benji("No reverse abbreviation found for code: $code", jobid)
        end
    end

    return join(result, "_")
end

"""
    input_encoder_abbrev_dict(
        X::Vector{String},
        Y::String,
        abbrev_map::Dict{String, String},
        jobid::Union{Nothing, String}=nothing
    ) -> String

Encode input names using a simple abbreviation dictionary.

# Arguments
- `X`: Vector of input file names.
- `Y`: Output file name.
- `abbrev_map`: Dictionary mapping names to abbreviation codes.
- `jobid`: Optional job ID for logging.

# Returns
- `String`: Hyphen-separated abbreviation string.

# Throws
- [`JobLoggerTools.error_benji`](@ref) if any name is not found in the map.
"""
function input_encoder_abbrev_dict(
    X::Vector{String},
    Y::String,
    abbrev_map::Dict{String, String},
    jobid::Union{Nothing, String}=nothing
)::String

    all_names = vcat(X, Y)
    encoded = String[]

    for name in all_names
        if haskey(abbrev_map, name)
            push!(encoded, abbrev_map[name])
        else
            JobLoggerTools.error_benji("No abbreviation for name: $name", jobid)
        end
    end

    return join(encoded, "-")
end

"""
    input_decoder_abbrev_dict(
        code_str::String,
        reverse_map::Dict{String, String},
        jobid::Union{Nothing, String}=nothing
    ) -> String

Decode an abbreviation string into the full underscore-joined file name string using a reverse map.

# Arguments
- `code_str`: Hyphen-separated abbreviation string.
- `reverse_map`: Dictionary mapping abbreviation codes back to names.
- `jobid`: Optional job ID for logging.

# Returns
- `String`: Underscore-joined decoded names.

# Throws
- [`JobLoggerTools.error_benji`](@ref) if any code is not found in the reverse map.
"""
function input_decoder_abbrev_dict(
    code_str::String,
    reverse_map::Dict{String, String},
    jobid::Union{Nothing, String}=nothing
)::String

    codes = split(code_str, "-")
    result = String[]

    for code in codes
        if haskey(reverse_map, code)
            push!(result, reverse_map[code])
        else
            JobLoggerTools.error_benji("No reverse abbreviation found for code: $code", jobid)
        end
    end

    return join(result, "_")
end

end