# ============================================================================
# src/Sarah/SummaryFormatter.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module SummaryFormatter

import Printf: @sprintf
import ..JobLoggerTools
import ..AvgErrFormatter
import ..Jackknife
import ..Bootstrap

"""
    print_bootstrap_average_error(
        arr::AbstractArray, 
        name::String, 
        arr_type::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{Float64, Float64, String}

Compute and print the bootstrap mean and standard deviation in formatted form with name tags.

# Arguments
- `arr::AbstractArray`: Bootstrap sample array.
- `name::String`: Label for the observable (e.g., `"Y_P1"`).
- `arr_type::String`: Group label for display (e.g., `"OBS"`).
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `Tuple{Float64, Float64, String}`: `(mean, stddev, formatted_string)`.

# Notes
- If `stddev == 0`, prints a warning and uses placeholder string.
"""
function print_bootstrap_average_error(
    arr::AbstractArray, 
    name::String, 
    arr_type::String, 
    jobid::Union{Nothing, String}=nothing
)::Tuple{Float64, Float64, String}
    
    m, s = Bootstrap.bootstrap_average_error(arr)
    
    if isnan(m)
        m = 0.0
    end

    if isnan(s)
        s = 0.0
    end

    m_str = @sprintf("%.14e",m)
    s_str = @sprintf("%.14e",s)
    
    if s == 0.0
        JobLoggerTools.warn_benji("$(name) AVG ONLY = $(m_str)", jobid)
        m_s_e2d = "[NOT determined!!]"
    else
        m_s_e2d = AvgErrFormatter.avgerr_e2d(m_str,s_str)
        JobLoggerTools.println_benji("$(name) AVG(ERR) = $(m_s_e2d) ($(arr_type))", jobid)
    end
    
    return m, s, m_s_e2d

end

# ---------------------------------------------------------
# JK error estimation session for check of the data 1
# This is Sumimoto's legacy
"""
    print_jackknife_average_error(
        jks::AbstractArray, 
        name::String, 
        arr_type::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{Float64, Float64, String}

Estimate average and standard error from precomputed jackknife samples and print result.

# Arguments
- `jks::AbstractArray`: Jackknife resample array.
- `name::String`: Label for observable.
- `arr_type::String`: Group label for display (e.g., `"OJK"`).
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `Tuple{Float64, Float64, String}`: `(mean, stddev, formatted_string)`.

# Notes
- If `stddev == 0`, prints a warning and uses placeholder string.
"""
function print_jackknife_average_error(
    jks::AbstractArray, 
    name::String, 
    arr_type::String, 
    jobid::Union{Nothing, String}=nothing
)::Tuple{Float64, Float64, String}
    
    m, s = Jackknife.jackknife_average_error(jks)

    m_str = @sprintf("%.14e",m)
    s_str = @sprintf("%.14e",s)

    if s == 0.0
        JobLoggerTools.warn_benji("$(name) AVG ONLY = $(m_str)", jobid)
        m_s_e2d = "[NOT determined!!]"
    else
        m_s_e2d = AvgErrFormatter.avgerr_e2d(m_str,s_str)
        JobLoggerTools.println_benji("$(name) AVG(ERR) = $(m_s_e2d) ($(arr_type))", jobid)
    end
    
    return m, s, m_s_e2d

end

# --------------------------------------------------------

"""
    print_jackknife_average_error_from_raw(
        arr::AbstractArray, 
        block::Int, 
        name::String, 
        arr_type::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{Float64, Float64, String}

Perform jackknife error estimation from raw data and print the result with label.

# Arguments
- `arr::AbstractArray`: Raw observable array.
- `block::Int`: Block size for jackknife binning.
- `name::String`: Label for the observable.
- `arr_type::String`: Group label for display (e.g., `"OJK"`).
- `jobid::Union{Nothing, String}`: Optional job ID for logging.

# Returns
- `Tuple{Float64, Float64, String}`: `(mean, stddev, formatted_string)`.

# Notes
- If `stddev == 0`, prints a warning and uses placeholder string.
"""
function print_jackknife_average_error_from_raw(
    arr::AbstractArray, 
    block::Int, 
    name::String, 
    arr_type::String, 
    jobid::Union{Nothing, String}=nothing
)::Tuple{Float64, Float64, String}
    
    m, s = Jackknife.jackknife_average_error_from_raw(arr, block)

    m_str = @sprintf("%.14e",m)
    s_str = @sprintf("%.14e",s)

    if s == 0.0
        JobLoggerTools.warn_benji("$(name) AVG ONLY = $(m_str)", jobid)
        m_s_e2d = "[NOT determined!!]"
    else
        m_s_e2d = AvgErrFormatter.avgerr_e2d(m_str,s_str)
        JobLoggerTools.println_benji("$(name) AVG(ERR) = $(m_s_e2d) ($(arr_type))", jobid)
    end
    
    return m, s, m_s_e2d

end

end  # module SummaryFormatter