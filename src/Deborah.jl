# ============================================================================
# src/Deborah.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

"""
       module Deborah

# `Deborah.jl` — End-to-end ``\\text{Tr} \\, M^{-n}``-driven analysis suite (ML → cumulants → reweighting → docs).

`Deborah.jl` is the top-level module that wires together the full workflow:

1. [`Deborah.DeborahCore`](@ref) — ML-based (supervised-learning) trace estimation with bias correction (build features, run [`Ridge`](https://juliaai.github.io/MLJ.jl/stable/models/RidgeRegressor_MLJLinearModels/#RidgeRegressor_MLJLinearModels)/[`Lasso`](https://juliaai.github.io/MLJ.jl/stable/models/LassoRegressor_MLJLinearModels/#LassoRegressor_MLJLinearModels)/[`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM), write summaries). Although designed primarily for trace estimation used by [`Deborah.Esther`](@ref) and [`Deborah.Miriam`](@ref), [`Deborah.DeborahCore`](@ref) can in fact handle any configuration-indexed observables with the same data layout, enabling a general supervised-learning and bias-correction workflow beyond traces.
2. [`Deborah.Esther`](@ref) — moment/cumulant analysis for a single ensemble based on ``\\text{Tr} \\, M^{-n} \\; (n=1,2,3,4)`` (load/scale traces, compute ``Q_n \\; (n=1,2,3,4)`` & cumulants, bootstrap/jackknife).  
3. [`Deborah.Miriam`](@ref) — multi-ensemble reweighting & interpolation (``\\kappa``-reweighting, `OG`/`P1`/`P2` estimators for reweighted observables, transition ``\\kappa`` determination).  
4. [`Deborah.DeborahThreads`](@ref) / [`Deborah.EstherThreads`](@ref) / [`Deborah.MiriamThreads`](@ref) — batch/threaded dispatchers for each stage.  
5. [`Deborah.DeborahDocument`](@ref) / [`Deborah.EstherDocument`](@ref) / [`Deborah.MiriamDocument`](@ref) — prepare [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshots.  
6. [`Deborah.Rebekah`](@ref) / [`Deborah.RebekahMiriam`](@ref) — plotting & I/O helpers (heatmaps, reweighting curves).  
7. [`Deborah.Rahab`](@ref) — pre-analysis reconnaissance module (inter-observable correlation checks, block-size determination for block bootstrap, ``T=0`` analysis for estimating lattice spacing and pseudo-scalar meson mass).
8. [`Deborah.Sarah`](@ref) — shared utilities (logging, [`TOML`](https://toml.io/en/) helpers, naming/abbreviations, data I/O, bootstrap/jackknife, summaries).

The module re-exports convenient entry points so typical users can run pipelines
with a single function call, or generate configs interactively via [`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/) "wizards".

# Submodules (bird's-eye)
- [`Deborah.Sarah`](@ref) — logging, config, naming, partitioning, resampling, summary utils  
- [`Deborah.DeborahCore`](@ref) — config → features/targets → ML sequences → bias-corrected outputs  
- [`Deborah.Esther`](@ref) — traces → Q-moments → cumulants → resampled summaries (single-ensemble workflow)
- [`Deborah.Miriam`](@ref) — multi-ensemble ``\\kappa``-reweighting, `OG`/`P1`/`P2`, transition-point ``\\kappa`` finding 
- [`Deborah.DeborahThreads`](@ref) / [`Deborah.EstherThreads`](@ref) / [`Deborah.MiriamThreads`](@ref) — threaded batch runners  
- [`Deborah.DeborahDocument`](@ref) / [`Deborah.EstherDocument`](@ref) / [`Deborah.MiriamDocument`](@ref) — document prep ([`JLD2`](https://juliaio.github.io/JLD2.jl/stable) dirs)  
- [`Deborah.Rebekah`](@ref) / [`Deborah.RebekahMiriam`](@ref) — loaders, heatmaps, reweighting plots ([``\\LaTeX``](https://www.latex-project.org/)-friendly)  
- [`Deborah.Rahab`](@ref) — correlation matrices, histograms, histories/``\\tau_\\text{int}``, block/bin scans, ``T=0`` scaling

# Exported Entry Points
## Core single-run
- [`Deborah.DeborahCore.DeborahRunner.run_Deborah`](@ref)
- [`Deborah.Esther.EstherRunner.run_Esther`](@ref)
- [`Deborah.Miriam.MiriamRunner.run_Miriam`](@ref)

## Bridged pipelines
- [`Deborah.DeborahEsther.DeborahEstherRunner.run_Deborah_Esther`](@ref)
- [`Deborah.DeborahEstherMiriam.DeborahEstherMiriamRunner.run_Deborah_Esther_Miriam`](@ref)

## Threaded/batch runners
- [`Deborah.DeborahThreads.DeborahThreadsRunner.run_DeborahThreads`](@ref)
- [`Deborah.EstherThreads.EstherThreadsRunner.run_EstherThreads`](@ref)
- [`Deborah.MiriamThreads.MiriamThreadsRunner.run_MiriamThreads`](@ref)
- [`Deborah.MiriamThreads.MiriamThreadsRunner.run_MiriamThreadsCheck`](@ref)

## Document layers
- [`Deborah.DeborahDocument.DeborahDocumentRunner.run_DeborahDocument`](@ref)
- [`Deborah.EstherDocument.EstherDocumentRunner.run_EstherDocument`](@ref)
- [`Deborah.MiriamDocument.MiriamDocumentRunner.run_MiriamDocument`](@ref)

## Interactive wizards ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
- [`Deborah.Elijah.DeborahWizardRunner.run_DeborahWizard`](@ref)
- [`Deborah.Elijah.EstherWizardRunner.run_EstherWizard`](@ref)
- [`Deborah.Elijah.MiriamWizardRunner.run_MiriamWizard`](@ref)
- [`Deborah.Elijah.DeborahThreadsWizardRunner.run_DeborahThreadsWizard`](@ref)
- [`Deborah.Elijah.EstherThreadsWizardRunner.run_EstherThreadsWizard`](@ref)
- [`Deborah.Elijah.MiriamThreadsWizardRunner.run_MiriamThreadsWizard`](@ref)

# Minimal Usage ([`REPL`](https://docs.julialang.org/en/v1/stdlib/REPL/))
```julia
julia> using Deborah

# Create configs via wizards
julia> run_DeborahWizard()
julia> run_EstherWizard()
julia> run_MiriamWizard()

# Run single pipelines
julia> run_Deborah("config_Deborah.toml")
julia> run_Esther("config_Esther.toml")
julia> run_Miriam("config_Miriam.toml")

# Run bridged variants
julia> run_Deborah_Esther("config_Esther.toml")
julia> run_Deborah_Esther_Miriam("config_Miriam.toml")

# Run threaded variants
julia> run_DeborahThreads("config_DeborahThreads.toml")
julia> run_EstherThreads("config_EstherThreads.toml")
julia> run_MiriamThreads("config_MiriamThreads.toml")
```

# Notes

* **Indices**: all column indices are **``1``-based** (files & arrays).
* **Abbreviations**: optional maps shorten long feature/target tokens in filenames and paths; wizards & runners respect this consistently.
* **Threading**: threaded runners dispatch up to [`Base.Threads.nthreads()`](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.nthreads) jobs in parallel; set [`JULIA_NUM_THREADS`](https://docs.julialang.org/en/v1/manual/environment-variables/#JULIA_NUM_THREADS) to control width.
* **Artifacts**: each stage writes structured text tables and [`JLD2`](https://juliaio.github.io/JLD2.jl/stable) snapshots under a stable `<location>/<analysis_header>_...` layout.
* **Plots/docs**: plotting helpers expect [`JLD2`](https://juliaio.github.io/JLD2.jl/stable)/text outputs; document runners prepare figure folders and [``\\LaTeX``](https://www.latex-project.org/) scaffolding for downstream PDF builds.
* **Backends**: [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM) is available via [`JuliaAI/MLJ.jl`](https://juliaai.github.io/MLJ.jl/stable/) or [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl) (when configured).

# See Also

* [`Deborah.DeborahCore`](@ref), 
* [`Deborah.Esther`](@ref), 
* [`Deborah.Miriam`](@ref), 
* [`Deborah.DeborahEsther`](@ref), 
* [`Deborah.DeborahEstherMiriam`](@ref),
* [`Deborah.DeborahThreads`](@ref), 
* [`Deborah.EstherThreads`](@ref), 
* [`Deborah.MiriamThreads`](@ref),
* [`Deborah.DeborahDocument`](@ref), 
* [`Deborah.EstherDocument`](@ref), 
* [`Deborah.MiriamDocument`](@ref),
* [`Deborah.Rebekah`](@ref), 
* [`Deborah.RebekahMiriam`](@ref), 
* [`Deborah.Rahab`](@ref),
* [`Deborah.Elijah`](@ref),  
* [`Deborah.Sarah`](@ref).
"""
module Deborah

include("Sarah/Sarah.jl")
include("Rebekah/Rebekah.jl")

using .Sarah
using .Rebekah

include("DeborahCore/DeborahCore.jl")
include("Esther/Esther.jl")
include("Miriam/Miriam.jl")

using .DeborahCore
using .Esther
using .Miriam

import .DeborahCore.DeborahRunner: run_Deborah
export run_Deborah

include("DeborahEsther/DeborahEsther.jl")
include("DeborahEstherMiriam/DeborahEstherMiriam.jl")

using .DeborahEsther
using .DeborahEstherMiriam

import .DeborahEsther.DeborahEstherRunner: run_Deborah_Esther
import .DeborahEstherMiriam.DeborahEstherMiriamRunner: run_Deborah_Esther_Miriam
export run_Deborah_Esther, run_Deborah_Esther_Miriam

include("DeborahThreads/DeborahThreads.jl")
include("EstherThreads/EstherThreads.jl")
include("MiriamThreads/MiriamThreads.jl")    

using .DeborahThreads
using .EstherThreads
using .MiriamThreads

import .DeborahThreads.DeborahThreadsRunner: run_DeborahThreads
import .EstherThreads.EstherThreadsRunner:   run_EstherThreads
import .MiriamThreads.MiriamThreadsRunner:   run_MiriamThreads, run_MiriamThreadsCheck
export run_DeborahThreads, run_EstherThreads, run_MiriamThreads, run_MiriamThreadsCheck

include("RebekahMiriam/RebekahMiriam.jl")

using .RebekahMiriam

include("Rahab/Rahab.jl")

using .Rahab

include("EstherDocument/EstherDocument.jl")
include("DeborahDocument/DeborahDocument.jl")
include("MiriamDocument/MiriamDocument.jl")

using .DeborahDocument
using .EstherDocument
using .MiriamDocument

import .DeborahDocument.DeborahDocumentRunner: run_DeborahDocument
import .EstherDocument.EstherDocumentRunner:   run_EstherDocument
import .MiriamDocument.MiriamDocumentRunner:   run_MiriamDocument
export run_DeborahDocument, run_EstherDocument, run_MiriamDocument

include("Elijah/Elijah.jl")
using .Elijah

import .Elijah.DeborahWizardRunner: run_DeborahWizard
import .Elijah.EstherWizardRunner:  run_EstherWizard
import .Elijah.MiriamWizardRunner:  run_MiriamWizard
import .Elijah.DeborahThreadsWizardRunner: run_DeborahThreadsWizard
import .Elijah.EstherThreadsWizardRunner:  run_EstherThreadsWizard
import .Elijah.MiriamThreadsWizardRunner:  run_MiriamThreadsWizard
export run_DeborahWizard, run_DeborahThreadsWizard, 
       run_EstherWizard,  run_EstherThreadsWizard, 
       run_MiriamWizard,  run_MiriamThreadsWizard

end  # module Deborah