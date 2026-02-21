# ============================================================================
# src/Esther/JackknifeRunner.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module JackknifeRunner

import ..Sarah.Jackknife
import ..SingleCumulant

"""
    compute_jackknife_observables(
        Q_moment::Dict{String, Vector{Float64}}, 
        LatVol::Int, 
        bin_size::Int
    ) -> Dict{String, Vector{Float64}}

Compute jackknife estimates of physical observables from ``Q``-moment data.

# Arguments
- `Q_moment::Dict{String, Vector{Float64}}`: Dictionary containing moment data with keys like `"Q1:Y_info"`, `"Q2:Y_info"`, etc.
- `LatVol::Int`: Lattice volume (typically ``V = N_S^3 \\times N_T``), used in normalization.
- `bin_size::Int`: Number of samples per jackknife bin.

# Returns
- `Dict{String, Vector{Float64}}`: Dictionary with jackknife samples of derived observables:
  - `"cond:Y_info"` - quark condensate
  - `"susp:Y_info"` - susceptibility
  - `"skew:Y_info"` - skewness
  - `"kurt:Y_info"` - kurtosis

Each value is a `Vector{Float64}` representing the jackknife resampled series.
"""
function compute_jackknife_observables(
    Q_moment::Dict{String, Vector{Float64}}, 
    LatVol::Int, 
    bin_size::Int
)

    jk_dict = Dict{String, Vector{Float64}}()

    Q1 = Q_moment["Q1:Y_info"]
    Q2 = Q_moment["Q2:Y_info"]
    Q3 = Q_moment["Q3:Y_info"]
    Q4 = Q_moment["Q4:Y_info"]

    jk1 = Jackknife.make_jackknife_samples(bin_size, Q1)
    jk2 = Jackknife.make_jackknife_samples(bin_size, Q2)
    jk3 = Jackknife.make_jackknife_samples(bin_size, Q3)
    jk4 = Jackknife.make_jackknife_samples(bin_size, Q4)

    jk_dict["cond:Y_info"] = SingleCumulant.calc_quark_condensate(jk1, LatVol)
    jk_dict["susp:Y_info"] = SingleCumulant.calc_susceptibility(jk1, jk2, LatVol)
    jk_dict["skew:Y_info"] = SingleCumulant.calc_skewness(jk1, jk2, jk3)
    jk_dict["kurt:Y_info"] = SingleCumulant.calc_kurtosis(jk1, jk2, jk3, jk4)

    return jk_dict
end

end  # module JackknifeRunner