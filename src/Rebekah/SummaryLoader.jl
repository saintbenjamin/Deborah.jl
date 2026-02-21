# ============================================================================
# src/Rebekah/SummaryLoader.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SummaryLoader

"""
    load_summary(
        work::String,
        analysis_ensemble::String,
        cumulant_name::String,
        overall_name::String,
        keys::Vector{String},
        labels::Vector{String},
        trains::Vector{String}
    ) -> Dict{String, Array{Float64,2}}

Load summary statistics (mean and error) from `.dat` files produced by [`Deborah.Esther`](@ref) or [`Deborah.DeborahCore`](@ref).

Each summary file is expected to contain comment lines in the format: `#<key>   <mean>   <stderr>`

These lines are parsed into 2D matrices indexed by `(label, train)` combinations, and stored as:
- `"key:avg"` → matrix of means
- `"key:err"` → matrix of errors

# Arguments
- `work::String` : Root working directory
- `analysis_ensemble::String` : Ensemble directory (e.g., `"L8T4b1.60k13570"`)
- `cumulant_name::String` : Observable subfolder (e.g., `"cond"`, `"susp"`)
- `overall_name::String` : Base name used in filenames
- `keys::Vector{String}` : Observable key(s), usually one or more of `"cond"`, `"susp"`, or `"Deborah"`
- `labels::Vector{String}` : Labeled set ratio tags (`LBP`, row index)
- `trains::Vector{String}` : Training set ratio tags (`TRP`, column index)

# Returns
- `Dict{String, Array{Float64,2}}` : Dictionary with keys like `"cond:avg"`, `"cond:err"` or `"Deborah:avg"`, each mapped to a matrix of shape `(length(labels), length(trains))`
"""
function load_summary(
    work::String, 
    analysis_ensemble::String, 
    cumulant_name::String, 
    overall_name::String, 
    keys::Vector{String},
    labels::Vector{String}, 
    trains::Vector{String}
)::Dict{String, Array{Float64,2}}

    raw_dict = Dict{String, Array{Float64,2}}()

    for (i, label) in enumerate(labels)
        for (j, train) in enumerate(trains)
            location = "$(work)/$(analysis_ensemble)/$(cumulant_name)/$(overall_name)_LBP_$(label)_TRP_$(train)"
            if keys == ["Deborah"]
                file = "summary_Deborah_$(overall_name)_LBP_$(label)_TRP_$(train).dat"
            else
                file = "summary_Esther_$(overall_name)_LBP_$(label)_TRP_$(train).dat"
            end
            filename = "$(location)/$(file)"

            if isfile(filename)
                for line in eachline(filename)
                    if startswith(line, "#")
                        parts = split(line)
                        key = replace(parts[1], "#" => "")
                        avg = length(parts) > 1 ? tryparse(Float64, parts[2]) : NaN
                        err = length(parts) > 2 ? tryparse(Float64, parts[3]) : NaN

                        if !haskey(raw_dict, key * ":avg")
                            raw_dict[key * ":avg"] = fill(NaN, length(labels), length(trains))
                            raw_dict[key * ":err"] = fill(NaN, length(labels), length(trains))
                        end

                        raw_dict[key * ":avg"][i, j] = avg
                        raw_dict[key * ":err"][i, j] = err
                    end
                end
            end
        end
    end

    return raw_dict
end

end