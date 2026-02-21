# ============================================================================
# src/Miriam/Ensemble.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# ============================================================================

module Ensemble

import ..Sarah.JobLoggerTools

"""
    Params{T}

Mutable container for lattice geometry and fermion parameters used in a
single simulation ensemble.

This struct encapsulates the minimal set of parameters required to define
a lattice-QCD ensemble at the level needed by higher-level analysis
pipelines (e.g. normalization of trace moments, volume factors, and
reweighting metadata). It is designed to be lightweight and is typically
embedded inside [`EnsembleStruct{T}`](@ref).

# Type Parameters
- `T`: Numeric type for floating-point parameters (e.g., `Float64`)

# Constructor
    Params(
        ns::Int,
        nt::Int,
        nf::Int,
        beta::T,
        csw::T,
        kappa::T
    ) -> Params{T}

Construct a `Params{T}` object describing lattice geometry and fermion
parameters for one ensemble.

## Constructor Arguments
- `ns`: Spatial lattice size
- `nt`: Temporal lattice size
- `nf`: Number of quark flavors
- `beta`: Gauge coupling ``\\beta``
- `csw`: Clover improvement coefficient ``c_{\\text{sw}}``
- `kappa`: Hopping parameter ``\\kappa``

## Constructor Returns
- A `Params{T}` instance containing the provided lattice and fermion parameters.

# Fields
- `ns::Int`: Spatial lattice size
- `nt::Int`: Temporal lattice size
- `nf::Int`: Number of quark flavors
- `beta::T`: Gauge coupling ``\\beta``
- `csw::T`: Clover improvement coefficient ``c_{\\text{sw}}``
- `kappa::T`: Hopping parameter ``\\kappa``

# Notes
- The total lattice volume is typically ``V = N_\\text{S}^3 N_\\text{T}`` and is often used
  downstream for rescaling trace moments and thermodynamic observables.
- This struct is intentionally mutable to allow controlled parameter
  updates in exploratory workflows or interactive analysis sessions.
"""
mutable struct Params{T}
    ns::Int
    nt::Int
    nf::Int
    beta::T
    csw::T
    kappa::T

    """
        Params(
            ns::Int, 
            nt::Int, 
            nf::Int, 
            beta::T, 
            csw::T, 
            kappa::T
        ) -> Params{T}

    Construct a `Params{T}` object for describing lattice geometry and fermion parameters.

    # Arguments
    - `ns`: Spatial lattice size
    - `nt`: Temporal lattice size
    - `nf`: Number of quark flavors
    - `beta`: Gauge coupling ``\\beta``
    - `csw`: Clover improvement coefficient ``c_{\\text{sw}}``
    - `kappa`: Hopping parameter ``\\kappa``

    # Returns
    - A `Params{T}` instance containing the provided values
    """
    function Params(
        ns::Int, 
        nt::Int, 
        nf::Int, 
        beta::T, 
        csw::T, 
        kappa::T
    ) where {T}
        return new{T}(ns, nt, nf, beta, csw, kappa)
    end
end

"""
    mutable struct EnsembleStruct{T}

Mutable structure representing a single ensemble in lattice-QCD workflows.

This struct stores one ensemble's gauge observables and trace-moment rows,
together with per-row provenance tags and configuration indices. It keeps both
a **scaled** representation (`trMi`) and an **unscaled/raw** representation
(`trMi_raw`) so downstream pipelines can choose the appropriate normalization.

# Type Parameters
- `T`: Numeric type for floating-point parameters (e.g., `Float64`)

# Constructor
    EnsembleStruct(
        nconf::Int,
        f::T,
        param::Params{T},
        plaq::Vector{T},
        rect::Vector{T},
        gact::Vector{T},
        ploop::Vector{Complex{T}},
        trMi::Vector{Vector{T}},
        trMi_raw::Vector{Vector{T}},
        source_tags::Vector{UInt8},
        secondary_tags::Vector{UInt8},
        conf_nums::Vector{Int}
    ) -> EnsembleStruct{T}

Construct an `EnsembleStruct{T}` to store lattice QCD ensemble data.

## Constructor Arguments
- `nconf`: Number of configurations (rows)
- `f`: Free energy or associated scalar (placeholder allowed)
- `param`: Simulation parameters (`Params{T}`)
- `plaq`, `rect`, `gact`: Gauge observables per configuration
- `ploop`: Polyakov loop per configuration
- `trMi`: **Scaled** trace moments (length-5 rows)
- `trMi_raw`: **Unscaled** (raw) trace moments (length-5 rows; commonly
  ``\\left[ 1.0, \\text{Tr} \\, M^{-1}, \\text{Tr} \\, M^{-2}, \\text{Tr} \\, M^{-3}, \\text{Tr} \\, M^{-4} \\right]``.)
- `source_tags`: Fine-grained source code per row (`0`: `"Y_tr"`, `1`: `"Y_bc"`/`"YP_bc"`, `2`: `"Y_ul"`/`"YP_ul"`)
- `secondary_tags`: Coarse class tag per row (`0`: `"Y_lb"`, `1`: `"Y_ul"`)
- `conf_nums`: Original configuration numbers aligned with rows

## Constructor Returns
- A fully populated `EnsembleStruct{T}` instance.

# Fields
- `nconf::Int`: Number of configurations (rows)
- `f::T`: Free-energy-like scalar (placeholder allowed)
- [`param::Params{T}`](@ref Deborah.Miriam.Ensemble.Params): Simulation parameters of the ensemble
- `plaq::Vector{T}`: Plaquette per configuration
- `rect::Vector{T}`: Rectangle observable per configuration
- `gact::Vector{T}`: Gauge action per configuration
- `ploop::Vector{Complex{T}}`: Polyakov loop per configuration
- `trMi::Vector{Vector{T}}`: rescaled trace-moment rows (length 5 each).
    The layout is ``\\left[ 12 \\, N_{\\text{f}} \\, V \\,,\\; \\left( 2 \\, \\kappa \\right)^1 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-1} \\,, \\cdots \\,, \\; \\left( 2 \\, \\kappa \\right)^4 \\, N_{\\text{f}} \\, \\text{Tr} \\, M^{-4} \\right]``.
    When a source provides only 4 components, the loader may append a
    placeholder 5th component (see [`Deborah.Miriam.MultiEnsembleLoader.generate_trMi_vector`](@ref)).
- `trMi_raw::Vector{Vector{T}}`: Un-rescaled (raw) trace-moment rows (length 5 each).
    The layout is typically ``\\left[ 1.0 \\,,\\; \\text{Tr} \\, M^{-1} \\,,\\; \\cdots ,\\; \\text{Tr} \\, M^{-4} \\right]``.
    No ``(12 \\, N_{\\text{f}} \\, V)`` or ``\\left( 2 \\, \\kappa \\right)^p`` rescaling is applied. If an input source
    provides only 4 raw components, the loader must expand it to length 5
    accordingly.
- `source_tags::Vector{UInt8}`: Fine-grained source identifier per row:
    - `0` → `"Y_tr"`
    - `1` → `"Y_bc"`, `"YP_bc"`
    - `2` → `"Y_ul"`, `"YP_ul"`
- `secondary_tags::Vector{UInt8}`: Coarse class tag per row:
    - `0` → `"Y_lb"` (originated from `"Y_tr"`, `"Y_bc"`, `"YP_bc"`)
    - `1` → `"Y_ul"` (originated from `"Y_ul"`, `"YP_ul"`)
- `conf_nums::Vector{Int}`: Original configuration numbers aligned with rows

# Invariants
- `length(trMi) == length(trMi_raw) == length(source_tags) == length(secondary_tags) == length(conf_nums) == nconf`
- `length(plaq) == length(rect) == length(gact) == nconf`
- `length(ploop) == nconf`
- Each `trMi[i]` and `trMi_raw[i]` has length `5`.

# Notes
- Input rows are typically sorted by configuration number in the loader.
- Replacement prefixes (`"YP_*"`) map to the same source/secondary classes
  as their non-replacement counterparts.
- Keeping both scaled (`trMi`) and unscaled (`trMi_raw`) representations
  allows downstream pipelines to choose the appropriate normalization.
"""
mutable struct EnsembleStruct{T}
    nconf::Int
    f::T
    param::Params{T}
    plaq::Vector{T}
    rect::Vector{T}
    gact::Vector{T}
    ploop::Vector{Complex{T}}
    trMi::Vector{Vector{T}}
    trMi_raw::Vector{Vector{T}}
    source_tags::Vector{UInt8}
    secondary_tags::Vector{UInt8}
    conf_nums::Vector{Int}

    """
        EnsembleStruct(
            nconf::Int,
            f::T,
            param::Params{T},
            plaq::Vector{T},
            rect::Vector{T},
            gact::Vector{T},
            ploop::Vector{Complex{T}},
            trMi::Vector{Vector{T}},
            trMi_raw::Vector{Vector{T}},
            source_tags::Vector{UInt8},
            secondary_tags::Vector{UInt8},
            conf_nums::Vector{Int}
        ) -> EnsembleStruct{T}

    Construct an `EnsembleStruct{T}` to store lattice QCD ensemble data.

    # Arguments
    - `nconf`: Number of configurations
    - `f`: Free energy or associated scalar
    - `param`: Simulation parameters (`Params{T}`)
    - `plaq`, `rect`, `gact`: Gauge observables
    - `ploop`: Polyakov loop per configuration
    - `trMi`: **Scaled** trace moments (length-5 rows)
    - `trMi_raw`: **Unscaled** (raw) trace moments (length-5 rows; commonly ``\\left[ 1.0, \\text{Tr} \\, M^{-1}, \\text{Tr} \\, M^{-2}, \\text{Tr} \\, M^{-3}, \\text{Tr} \\, M^{-4} \\right]``.)
    - `source_tags`: Source code for trace (`0`: `tr`, `1`: `bc`, `2`: `ul`)
    - `secondary_tags`: Secondary class tag (`0`: `Y_lb`, `1`: `Y_ul`)
    - `conf_nums`: Configuration indices

    # Returns
    - A fully populated `EnsembleStruct{T}` instance.
    """
    function EnsembleStruct(
        nconf::Int,
        f::T,
        param::Params{T},
        plaq::Vector{T},
        rect::Vector{T},
        gact::Vector{T},
        ploop::Vector{Complex{T}},
        trMi::Vector{Vector{T}},
        trMi_raw::Vector{Vector{T}},
        source_tags::Vector{UInt8},
        secondary_tags::Vector{UInt8},
        conf_nums::Vector{Int}
    ) where {T}
        return new{T}(nconf, f, param, plaq, rect, gact, ploop, trMi, trMi_raw, source_tags, secondary_tags, conf_nums)
    end
end

"""
    mutable struct EnsembleArray{T}

Mutable container holding a homogeneous list of lattice-QCD ensembles.

This struct is a thin wrapper around a `Vector{EnsembleStruct{T}}`, used to
group multiple ensembles that should be treated together in downstream
analysis steps (e.g. iteration, filtering, or bundling). It carries no
additional metadata beyond the ordered list itself.

# Type Parameters
- `T`: Numeric type for floating-point parameters (e.g., `Float64`)

# Constructor
    EnsembleArray(
        data::Vector{EnsembleStruct{T}}
    ) -> EnsembleArray{T}

Construct an `EnsembleArray{T}` from a given list of ensemble structures.

## Constructor Arguments
- `data`: A vector of `EnsembleStruct{T}` instances

## Constructor Returns
- A new `EnsembleArray{T}` instance wrapping the provided ensemble list.

# Fields
- [`data::Vector{EnsembleStruct{T}}`](@ref Deborah.Miriam.Ensemble.EnsembleStruct):
  Ordered list of ensembles.

# Notes
- No consistency checks across ensembles are enforced at this level
  (e.g. matching lattice sizes or parameters); such validation is expected
  to be handled by higher-level logic.
- This container mainly exists to provide a clear semantic unit when
  grouping ensembles before bundling or tagging.
"""
mutable struct EnsembleArray{T}
    data::Vector{EnsembleStruct{T}}

    """
        EnsembleArray(data::Vector{EnsembleStruct{T}}) -> EnsembleArray{T}

    Construct an `EnsembleArray{T}` from a given list of `EnsembleStruct{T}` instances.

    # Arguments
    - `data`: A vector of ensemble structures

    # Returns
    - A new `EnsembleArray{T}` instance
    """    
    function EnsembleArray(data::Vector{EnsembleStruct{T}}) where {T}
        return new{T}(data)
    end
end

"""
    mutable struct EnsembleArrayBundle{T}

Mutable container pairing multiple ensemble arrays with string tags.

This struct bundles several [`EnsembleArray{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArray)
objects together, each labeled by a corresponding string tag. It is typically
used to organize ensemble groups by role or origin (e.g. target vs reference,
interpolation sets, training partitions) in multi-ensemble workflows.

# Type Parameters
- `T`: Numeric type for floating-point parameters (e.g., `Float64`)

# Constructor
    EnsembleArrayBundle(
        arrays::Vector{EnsembleArray{T}},
        tags::Vector{String},
        jobid::Union{Nothing, String} = nothing
    ) -> EnsembleArrayBundle{T}

Construct an `EnsembleArrayBundle{T}` by pairing multiple ensemble arrays
with corresponding tags.

## Constructor Arguments
- `arrays`: Vector of `EnsembleArray{T}` instances
- `tags`: Vector of `String` tags, one for each array
- `jobid`: Optional job ID used for contextual logging and error reporting

## Constructor Returns
- A new `EnsembleArrayBundle{T}` instance.

## Constructor Checks
- An `AssertionError` is raised if `length(arrays) != length(tags)`.

# Fields
- [`arrays::Vector{EnsembleArray{T}}`](@ref Deborah.Miriam.Ensemble.EnsembleArray):
  Collection of ensemble arrays.
- `tags::Vector{String}`:
  Human-readable tags describing each ensemble array (must match in length).

# Notes
- The positional correspondence between `arrays[i]` and `tags[i]` is
  semantically significant and must be preserved.
- Tags are intentionally free-form strings to allow flexible labeling
  schemes across different analysis stages.
- This struct is designed as a lightweight organizational layer and does
  not impose constraints on the contents of individual arrays.
"""
mutable struct EnsembleArrayBundle{T}
    arrays::Vector{EnsembleArray{T}}
    tags::Vector{String}

    """
        EnsembleArrayBundle(
            arrays::Vector{EnsembleArray{T}}, 
            tags::Vector{String},
            jobid::Union{Nothing, String}=nothing
        ) -> EnsembleArrayBundle{T}

    Construct an `EnsembleArrayBundle{T}` by pairing multiple ensemble arrays with corresponding tags.

    # Arguments
    - `arrays::Vector{EnsembleArray{T}}`: Vector of `EnsembleArray{T}` instances
    - `tags::Vector{String}`: Vector of `String` tags, one for each array
    - `jobid::Union{Nothing, String}`: Optional job ID for logging.
    
    # Returns
    - A new `EnsembleArrayBundle{T}` instance

    # Throws
    - `AssertionError` if the lengths of `arrays` and `tags` do not match
    """
    function EnsembleArrayBundle(
        arrays::Vector{EnsembleArray{T}}, 
        tags::Vector{String},
        jobid::Union{Nothing, String}=nothing
    ) where {T}
        JobLoggerTools.assert_benji(length(arrays) == length(tags), "Number of arrays and tags must match", jobid)
        return new{T}(arrays, tags)
    end
end

end  # module Ensemble