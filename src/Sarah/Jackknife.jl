# ============================================================================
# src/Sarah/Jackknife.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module Jackknife

import ..Statistics

"""
    make_jackknife_samples(
        bin_size::Int, 
        data::Vector{T}
    ) where T -> Vector{T}

Generate jackknife resamples by systematically leaving out bins of data.

# Arguments
- `bin_size::Int`: Size of each jackknife bin.
- `data::Vector{T}`: Input data vector to resample.

# Returns
- `data_jk::Vector{T}`: Jackknife samples (`length = number of bins`).
"""
function make_jackknife_samples(
    bin_size::Int, 
    data::Vector{T}
) where T
    ndata = length(data)
    njk = div(ndata, bin_size)
    ndata = bin_size * njk  # truncate to full bins

    sum_data = sum(data[1:ndata])
    data_jk = Vector{T}(undef, njk)

    for i in 1:njk
        data_jk[i] = sum_data
        for j in ((i - 1) * bin_size + 1):(i * bin_size)
            data_jk[i] -= data[j]
        end
        data_jk[i] /= (ndata - bin_size)
    end

    return data_jk
end

"""
    _make_jackknife_samples(
        data::Vector{T}, 
        bin_size::Integer=1
    ) -> Vector{T}

Create jackknife samples by leaving out non-overlapping bins of size `bin_size`.
# Arguments
- `bin_size`: Number of consecutive points to leave out per sample (default = `1`).
- `data`: Original vector of data points.

# Returns
- `Vector{T}` of jackknife sample means.

# Notes
- This is Takayuki Sumimoto's legacy. Currently unused.
"""
function _make_jackknife_samples(
    bin_size::Integer, 
    data::Vector{T}
) where T 

    ndata = length(data)
    njk = ndata ÷ bin_size
    ndata = njk * bin_size
    nsamples = ndata - bin_size

    s = sum(@view data[begin:ndata])

    return [(s - sum(@view data[bin_size * (i - 1) + 1:bin_size * i])) / nsamples for i = 1:njk]

end

"""
    jackknife_average_error(
        jk::Vector{Float64}
    ) -> Tuple{Float64, Float64}

Compute the average and jackknife error from jackknife resamples.

# Arguments
- `jk`: Vector of jackknife sample means.

# Returns
- `(mean, error)`: Tuple of the overall mean and jackknife standard error.
"""
function jackknife_average_error(
    jk::Vector{Float64}
)::Tuple{Float64, Float64}

    njk = length(jk)

    ave = Statistics.mean(jk)
    err = sqrt.((njk - 1) * Statistics.varm(jk, ave, corrected=false))

    return ave, err

end

"""
    jackknife_average_error_from_raw(
        arr::AbstractArray, 
        block::Int
    ) -> Tuple{Float64, Float64}

Compute jackknife average and standard error from raw data

# Arguments
- `arr`: Raw observable array.
- `block`: Block size for jackknife binning.

# Returns
- `(mean, stddev)`
"""
function jackknife_average_error_from_raw(
    arr::AbstractArray, 
    block::Int
)
    
    jks = make_jackknife_samples(block, arr);
    m, s = jackknife_average_error(jks)
    
    return m, s

end

"""
    jackknife_standard_error_from_sample_variance(
        jk::Vector{Float64}
    ) -> Float64

Compute the jackknife estimate of the sample standard deviation.

# Arguments
- `jk`: Vector of jackknife sample means.

# Returns
- Jackknife estimate of the standard deviation.
"""
function jackknife_standard_error_from_sample_variance(
    jk::Vector{Float64}
)::Float64

    njk = length(jk)
    ave = Statistics.mean(jk)
    var = sqrt.(njk * (njk - 1) * Statistics.varm(jk, ave, corrected=false))

    return var

end

# Benji added 240919
"""
    jackknife_restore_original_single(
        ajk::Vector{Float64}
    ) -> Vector{Float64}

Reconstruct the original sample values from single-elimination jackknife averages.

# Arguments
- `ajk`: Vector of jackknife sample means (assumed from single-point exclusion).

# Returns
- Reconstructed original data values.

# Notes
- Currently unused. Benjamin's legacy from his PhD days' experience.
"""
function jackknife_restore_original_single(
    ajk::Vector{Float64}
)::Vector{Float64}

    njk = length(ajk)

    ave = Statistics.mean(ajk)

    ojk = ave .+ ( njk - 1 ) * ( ave .- ajk ) # available only for single elimination

    return ojk

end

end  # module Jackknife