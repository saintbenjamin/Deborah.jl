# =============================================================================
# src/Miriam/Reweighting.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License with Attribution Requirement
# =============================================================================

module Reweighting

import Printf: @sprintf
import OrderedCollections
import NLsolve
import ..Sarah.JobLoggerTools
import ..Sarah.TOMLLogger
import ..Ensemble
import ..EnsembleUtils

"""
    mutable struct ReweightingSolver{T}

Mutable solver object for computing reweighting factors over a collection
of lattice-QCD ensembles.

This struct encapsulates all state required to iteratively solve for
reweighting weights across an [`EnsembleArray{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArray),
including convergence control parameters and the flattened global weight
vector. It is designed to be initialized once per reweighting task and then
updated in-place during the solve loop.

# Type Parameters
- `T`: Numeric type for floating-point parameters (e.g., `Float64`)

# Constructor
    ReweightingSolver(
        ens::Ensemble.EnsembleArray{T},
        maxiter::Int = 500,
        eps::T = 1e-12
    ) -> ReweightingSolver{T}

Construct a `ReweightingSolver{T}` for a given ensemble array.

## Constructor Arguments
- `ens`: Target [`EnsembleArray{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArray) to operate on
- `maxiter`: Maximum number of iterations allowed in the solver
- `eps`: Convergence threshold for the iterative update

## Constructor Behavior
- Computes `nconf_all` as the total number of configurations across all
  ensembles contained in `ens`
- Initializes the global reweighting vector `w` to a length-`nconf_all`
  vector of ones

## Constructor Returns
- A fully initialized `ReweightingSolver{T}` instance ready for iteration.

# Fields
- [`ens::Ensemble.EnsembleArray{T}`](@ref Deborah.Miriam.Ensemble.EnsembleArray):
  Ensemble array on which multi-ensemble reweighting is performed
- `maxiter::Int`: Maximum number of iterations for solving
- `eps::T`: Convergence threshold
- `w::Vector{T}`: Current flattened reweighting vector (length `nconf_all`)
- `nconf_all::Int`: Total number of configurations across all ensembles

# Notes
- The weight vector `w` is stored in a flattened form that concatenates
  configurations from all ensembles in `ens` in their internal order.
- This struct is intentionally mutable, as `w` and internal solver state
  are updated iteratively during the reweighting procedure.
"""
mutable struct ReweightingSolver{T}
    ens::Ensemble.EnsembleArray{T}
    maxiter::Int
    eps::T
    w::Vector{T}
    nconf_all::Int

    """
        ReweightingSolver(
            ens::Ensemble.EnsembleArray{T},
            maxiter::Int = 500,
            eps::T = 1e-12
        ) -> ReweightingSolver{T}

    Construct a `ReweightingSolver{T}` for iterative multi-ensemble reweighting.

    This constructor initializes all solver state required to compute
    reweighting factors over the given ensemble array. The global number of
    configurations is inferred automatically, and the reweighting vector is
    initialized uniformly.

    # Arguments
    - `ens`: Target [`EnsembleArray{T}`] containing one or more ensembles
    - `maxiter`: Maximum number of solver iterations
    - `eps`: Convergence threshold for stopping the iteration

    # Behavior
    - Computes the total number of configurations
    ``n_{\\text{conf,all}} = \\sum_e n_{\\text{conf}}^{(e)}``
    across all ensembles in `ens`
    - Allocates the reweighting vector `w` as a length-`nconf_all` vector
    initialized to `one(T)`

    # Returns
    - A fully initialized `ReweightingSolver{T}` instance, ready to run the
    reweighting iteration.
    """
    function ReweightingSolver(ens::Ensemble.EnsembleArray{T}, maxiter::Int = 500, eps::T = 1e-12) where T
        nconf_all = sum(e.nconf for e in ens.data)
        w = ones(T, nconf_all)
        return new{T}(ens, maxiter, eps, w, nconf_all)
    end
end

"""
    fdf!(
        F::Vector{T}, 
        f::Vector{T}, 
        rw::ReweightingSolver{T}
    ) -> (Vector{T}, Matrix{T})

Compute the residual vector `F` and Jacobian matrix `J` for the reweighting equations
in a system of `n` ensembles, using the current guess `f` for the free energy offsets.

# Arguments
- `F::Vector{T}`: Output residual vector (size ``n_{\\mathrm{ens}} - 1``)
- `f::Vector{T}`: Current guess for free energy offsets (size ``n_{\\mathrm{ens}} - 1``)
- [`rw::ReweightingSolver{T}`](@ref Deborah.Miriam.Reweighting.ReweightingSolver): Reweighting solver containing ensemble data

# Returns
- `F::Vector{T}`: Updated residual vector
- `J::Matrix{T}`: Computed Jacobian matrix (size ``(n_{\\mathrm{ens}} - 1) \\times (n_{\\mathrm{ens}} - 1)``)

---

# Notes

This is part of a Newton-Raphson solver for determining optimal reweighting factors.

The equations implemented here are mathematically identical to the
MBAR (Multistate Bennett Acceptance Ratio) self-consistency equations.
MBAR provides the statistically optimal (minimum-variance, asymptotically
unbiased) estimator for the relative free energies of multiple ensembles,
given finite sampling from each.  

In the Lattice QCD literature, the same formalism is more often referred to
simply as multi-ensemble reweighting or [Ferrenberg-Swendsen reweighting](https://inspirehep.net/literature/270327)
rather than by the name MBAR, but the underlying equations coincide.

## Problem Setup

We have ``n_{ens}`` ensembles indexed by ``k=1,\\dots,n_{\\mathrm{ens}}``.  
Each ensemble has:
- Number of configurations: ``N_k = \\texttt{ens[k].nconf}``,
- Parameter (e.g. coupling, potential parameter): ``\\theta_k = \\texttt{ens[k].param}``. Here we mainly perform reweighting in ``\\kappa``. Hence, we write ``\\kappa_k`` for clarity hereafter.

For each configuration ``\\mathcal{U}_{j;m}`` drawn from ensemble ``j``,  
(where ``m`` denotes the gauge configuration index),  
we can evaluate the reduced action (or potential) under another parameter ``\\kappa_k``:
```math
\\Delta S_{j;m}(\\kappa_k) \\equiv S(\\kappa_k; \\mathcal{U}_{j;m}) - S(\\kappa_j;\\mathcal{U}_{j;m})
```

We fix one ensemble (say ``k=n_{\\mathrm{ens}}``, the last index) as the reference
and solve for the free energy offsets
```math
\\mathbf{f} = (f_1, f_2, \\dots, f_{n_{\\mathrm{ens}} - 1}).
```

## Residual Equations

For each target ensemble ``i = 1,\\dots,n_{\\mathrm{ens}} - 1``, define
```math
D_{j;m}^{(i)}(f) = \\sum_{k=1}^{n_{\\mathrm{ens}}}
N_k \\exp\\!\\left[ E_{j;m}^{(i,k)} \\right].
```
where
```math
E_{j;m}^{(i,k)} = \\Delta S_{j;m}(\\kappa_i) - \\Delta S_{j;m}(\\kappa_k) - f_i + f_k \\,.
```

Then define the normalization sum
```math
T_i(f) = \\sum_{j=1}^{n_{\\mathrm{ens}}}\\;
\\sum_{m=1}^{N_j}
\\frac{1}{D_{j;m}^{(i)}(f)}.
```

The residual equations are
```math
F_i(f) = \\log T_i(f), \\qquad i=1,\\dots,n_{\\mathrm{ens}} - 1.
```

The system of equations to be solved is
```math
F_i(f) = 0, \\qquad i=1,\\dots,n_{\\mathrm{ens}} - 1,
```
which enforces
```math
T_i(f) = 1.
```

This is precisely the **MBAR self-consistency condition** for the free energies.

## Jacobian Matrix

We compute the Jacobian entries
```math
J_{i \\ell}(f) = \\frac{\\partial F_i}{\\partial f_\\ell}.
```

From the definition,
```math
\\frac{\\partial F_i}{\\partial f_\\ell}
= \\frac{1}{T_i(f)} \\sum_{j,m}
\\left(
-\\frac{1}{(D_{j;m}^{(i)}(f))^2}
\\frac{\\partial D_{j;m}^{(i)}}{\\partial f_\\ell}
\\right).
```

The derivative of the denominator is
```math
\\frac{\\partial D_{j;m}^{(i)}}{\\partial f_\\ell}
= N_\\ell \\; \\exp\\!\\left[ E_{j;m}^{(i,\\ell)} \\right]
- \\delta_{i\\ell}\\,D_{j;m}^{(i)}(f).
```
where
```math
E_{j;m}^{(i,\\ell)} = \\Delta S_{(j \\to i);m} - \\Delta S_{(j \\to \\ell);m} - f_i + f_\\ell.
```

Substituting back,
```math
J_{i \\ell}(f) =
\\frac{1}{T_i(f)}\\sum_{j,m}
\\left(
-\\frac{N_\\ell \\; \\exp\\!\\left[E_{j;m}^{(i,\\ell)}\\right]}{\\left(D_{j;m}^{(i)}(f)\\right)^2}
\\right)
+ \\delta_{i\\ell} \\,.
```

Thus:
- Off-diagonal entries (``i \\neq \\ell``) come only from the exponential ratio term,
- Diagonal entries (``i=\\ell``) acquire an additional ``+1``.

## Summary

- Unknowns: free energy offsets ``f_1,\\dots,f_{n_{\\mathrm{ens}} - 1}``,
- Equations: ``F_i(f)=0``, enforcing MBAR consistency,
- Residual vector:
  ```math
  F_i(f) = \\log\\left(\\sum_{j,m} \\frac{1}{D_{j;m}^{(i)}(f)}\\right),
  ```
- Jacobian matrix:
  ```math
  J_{i\\ell}(f) =
  \\frac{1}{T_i(f)} \\sum_{j,m}
  \\left(-\\frac{N_\\ell \\; \\exp\\!\\left[E_{j;m}^{(i,\\ell)}\\right]}{(D_{j;m}^{(i)}(f))^2}\\right)
  + \\delta_{i\\ell}.
  ```

This system is solved iteratively using Newton-Raphson updates:
```math
f \\;\\mapsto\\; f - J^{-1}(f)\\,F(f).
```
"""
function fdf!(
    F::Vector{T}, 
    f::Vector{T}, 
    rw::ReweightingSolver{T}
)::Tuple{Vector{T}, Matrix{T}} where T
    ens = rw.ens.data
    nens = length(ens)
    n = nens - 1  # Fix one ensemble as reference

    # Update current free energy guesses
    for i in 1:n
        ens[i].f = f[i]
    end

    # Initialize Jacobian matrix and numerical safety bounds
    J = zeros(T, n, n)
    max_exponent =  709.78  # Upper bound to avoid overflow in exp
    min_exponent = -745.13  # Lower bound to avoid underflow in exp

    for i in 1:n
        tmp = zero(T)                # Accumulator for residual F[i]
        Jarray = zeros(T, n)         # Temporary Jacobian row

        for j in 1:nens
            for iconf in 1:ens[j].nconf
                tmp2 = zero(T)       # Denominator accumulator
                tmp3 = zeros(T, nens)  # Numerator terms for softmax-like ratio

                for k in 1:nens
                    ds1 = EnsembleUtils.dS(ens[j], ens[i].param, iconf)
                    ds2 = EnsembleUtils.dS(ens[j], ens[k].param, iconf)
                    exponent = ds1 - ds2 - ens[i].f + ens[k].f
                    exponent = clamp(exponent, min_exponent, max_exponent)

                    tmp3[k] = ens[k].nconf * exp(exponent)
                    tmp2 += tmp3[k]
                end

                tmp2 = max(tmp2, eps(T))
                tmp2 = inv(tmp2)
                tmp += tmp2

                for k in 1:n
                    Jarray[k] -= tmp3[k] * tmp2^2
                end
            end
        end

        # Avoid log(0) by bounding tmp
        tmp = max(tmp, eps(T))
        F[i] = log(tmp)

        # Store the Jacobian row
        for j in 1:n
            J[i, j] = Jarray[j] / tmp
            if i == j
                J[i, j] += one(T)
            end
        end
    end

    return F, J
end

"""
    set_init_guess!(
        rw::ReweightingSolver{T}
    ) -> Nothing where T

Initialize the free energy estimates `f` in the given [`ReweightingSolver`](@ref).

# Arguments
- [`rw::ReweightingSolver{T}`](@ref Deborah.Miriam.Reweighting.ReweightingSolver): The solver object containing all ensembles

# Returns
- `Nothing` (modifies ensemble fields in-place)

---

# Notes

For each ensemble ``i``, the initial value ``f_i`` is set to the maximum action
difference [`Deborah.Miriam.EnsembleUtils.dS`](@ref) across all configurations ``j`` from all ensembles, evaluated under
the parameter ``\\theta_i``, or ``\\kappa_i`` more specifically:
```math
f_i^{(0)} = \\max_{1 \\le j \\le n_{\\mathrm{ens}}} \\left( \\max_{1 \\le m \\le N_j}
\\Delta S_{j;m}(\\kappa_i) \\right) \\,, \\qquad \\Delta S_{j;m}(\\kappa_i) \\equiv S(\\kappa_i; \\mathcal{U}_{j;m}) - S(\\kappa_j;\\mathcal{U}_{j;m}) \\,.
```
This provides a robust starting point for the Newton iteration used in
multi-ensemble reweighting (MBAR-type equations). Choosing the maximum value
ensures numerical stability by avoiding underestimation of the exponential
weights.
"""
function set_init_guess!(
    rw::ReweightingSolver{T}
)::Nothing where T
    ens = rw.ens.data
    nens = length(ens)

    for i in 1:nens
        dSmax = zero(T)

        for j in 1:nens
            for iconf in 1:ens[j].nconf
                tmp = EnsembleUtils.dS(ens[j], ens[i].param, iconf)
                dSmax = max(dSmax, tmp)
            end
        end

        ens[i].f = dSmax
    end

    return nothing
end

"""
    calc_f!(
        rw::ReweightingSolver{T}, 
        info_file::String, 
        tag::String, 
        jobid::Union{Nothing, String}=nothing
    ) -> Nothing where T

Solve for the free energy differences (``f_i``) between ensembles using
Newton-Raphson iteration, with analytic Jacobian and result logging.

# Arguments

* [`rw::ReweightingSolver{T}`](@ref Deborah.Miriam.Reweighting.ReweightingSolver): Solver instance with ensemble data and parameters.
* `info_file::String`: Path to the [`TOML`](https://toml.io/en/) file for logging results.
* `tag::String`: Tag label for the [`TOML`](https://toml.io/en/) solver block.
* `jobid::Union{Nothing,String}`: Optional job identifier for logging context.

# Behavior

- Initializes ``f`` with maximum action differences.
- Runs Newton iteration with analytic Jacobian.
- Updates ensemble fields `ens[i].f` in place (except fixed ensemble).
- Logs solver details and residuals to [`TOML`](https://toml.io/en/).

# Returns

- `Nothing` (modifies solver state and logs status in place).

---

# Notes

This function provides the **core solver** for the MBAR self-consistency equations.
For each target ensemble ``i = 1, \\dots, n_{\\mathrm{ens}} - 1``, we define
the denominator
```math
D_{j;m}^{(i)}(f)
= \\sum_{k=1}^{n_{\\mathrm{ens}}}
N_k \\; \\exp\\!\\left[
\\Delta S_{j;m}(\\kappa_i)
- \\Delta S_{j;m}(\\kappa_k)
- f_i + f_k
\\right] \\, ,
```
where ``j`` labels the source ensemble and ``m`` indexes configurations
within that ensemble.
Using this quantity, the normalization sum for ensemble ``i`` is given by
```math
T_i(f)
= \\sum_{j=1}^{n_{\\mathrm{ens}}}
\\sum_{m=1}^{N_j}
\\frac{1}{D_{j;m}^{(i)}(f)} \\, .
```
The MBAR residual equations are then written in the compact form
```math
F_i(f) = \\log T_i(f) = 0,
\\qquad i = 1, \\dots, n_{\\mathrm{ens}} - 1 \\, .
```
These nonlinear equations enforce the self-consistency condition
``T_i(f) = 1`` for each target ensemble, corresponding to statistical
equilibrium among all ensembles in the reweighting framework.
The solution determines the optimal free-energy offsets
``(f_1, \\dots, f_{n_{\\mathrm{ens}}-1})``, with one ensemble fixed to set
the overall normalization.



## Workflow

1. **Initialization**
   Calls [`set_init_guess!`](@ref) to provide robust starting values
   ```math
   f_i^{(0)} = \\max_{1 \\le j \\le n_{\\mathrm{ens}}} \\left( \\max_{1 \\le m \\le N_j}
   \\Delta S_{j;m}(\\kappa_i) \\right) \\,.
   ```
   ensuring stability of the first Newton step.

2. **Residual and Jacobian**
   Uses [`fdf!`](@ref) to evaluate the MBAR residual vector ``F(f)`` and its
   analytic Jacobian ``J(f) = \\dfrac{\\partial F}{\\partial f}``.

3. **Nonlinear Solve**
   Calls [`NLsolve.nlsolve`](https://docs.sciml.ai/NonlinearSolve/stable/api/nlsolve/)
   to iteratively update

   ```math
   f \\;\\mapsto\\; f - J^{-1}(f)\\,F(f).
   ```

4. **Normalization**
   One ensemble's free energy (typically the last) is held fixed so that only
   ``n_{\\mathrm{ens}}-1`` unknowns are solved.

5. **Logging**
   Solver diagnostics (convergence, iterations, residual norms, solutions) are
   appended to the specified [`TOML`](https://toml.io/en/) file under a tagged section.
"""
function calc_f!(
    rw::ReweightingSolver{T}, 
    info_file::String,
    tag::String,
    jobid::Union{Nothing, String}=nothing
) where T
    ens = rw.ens.data
    nens = length(ens)
    n = nens - 1  # One f is fixed due to normalization
    maxiter = rw.maxiter
    eps = rw.eps

    # Initialize f values with max EnsembleUtils.dS
    set_init_guess!(rw)
    fvec = [ens[i].f for i in 1:n]

    # If all initial guesses are zero → skip solve
    skip_nlsolve = all(iszero, fvec)

    if skip_nlsolve
        JobLoggerTools.warn_benji("Skip NLsolve: all f initial guesses are zero", jobid)

        summary_dict = OrderedCollections.OrderedDict(
            "converged"      => "skipped",
            "iterations"     => "0",
            "residual_norm"  => "NaN",
            "f_calls"        => "0",
            "method"         => "skipped",
            "f_converged"    => "false",
            "x_converged"    => "false",
        )

        for i in 1:n
            summary_dict["initial_guess_$i"] = @sprintf("%.12e", fvec[i])
        end
        summary_dict["initial_guess_$(nens)"] = @sprintf("%.12e", ens[nens].f)

        for i in 1:n
            summary_dict["solution_$i"] = @sprintf("%.12e", 0.0)
        end
        summary_dict["solution_$(nens)"] = @sprintf("%.12e", ens[nens].f)

        for i in 1:n
            summary_dict["residual_$i"] = "NaN"
        end
        summary_dict["residual_$(nens)"] = "fixed"

        TOMLLogger.append_section_to_toml(info_file, "NLsolve.nlsolve_f_solver_$(tag)", summary_dict)
        return nothing
    end

    # Solve nonlinear system with Jacobian (defined in fdf!)
    result = NLsolve.nlsolve(
        (F, f) -> fdf!(F, f, rw),
        fvec,
        ftol = eps,
        iterations = maxiter,
        show_trace = jobid === nothing,
    ); flush(stdout); flush(stderr)

    if !NLsolve.converged(result)
        JobLoggerTools.warn_benji("Optimization did not converge!!!", jobid)
    end

    # Update solved f values in ensemble structs
    for i in 1:n
        ens[i].f = result.zero[i]
    end

    # Dump solver status to TOML
    summary_dict = OrderedCollections.OrderedDict(
        "converged"      => string(NLsolve.converged(result)),
        "iterations"     => string(result.iterations),
        "residual_norm"  => @sprintf("%.12e", result.residual_norm),
        "f_calls"        => string(result.f_calls),
        "method"         => string(result.method),
        "f_converged"    => string(result.f_converged),
        "x_converged"    => string(result.x_converged),
    )

    for (i, val) in enumerate(fvec)
        summary_dict["initial_guess_$i"] = @sprintf("%.12e", val)
    end
    summary_dict["initial_guess_$(nens)"] = @sprintf("%.12e", ens[nens].f)

    for (i, val) in enumerate(result.zero)
        summary_dict["solution_$i"] = @sprintf("%.12e", val)
    end
    summary_dict["solution_$(nens)"] = @sprintf("%.12e", ens[nens].f)

    resid = similar(result.zero)

    fdf!(resid, result.zero, rw)
    for (i, val) in enumerate(resid)
        summary_dict["residual_$i"] = @sprintf("%.12e", val)
    end
    summary_dict["residual_$(nens)"] = "fixed"

    TOMLLogger.append_section_to_toml(info_file, "NLsolve.nlsolve_f_solver_$(tag)", summary_dict)

    return nothing
end

"""
    calc_w!(
        rw::ReweightingSolver{T}, 
        paramT::Ensemble.Params{T}
    ) -> Nothing

Compute normalized reweighting weights ``w_{j;m}`` for all configurations,
given a target parameter set ``\\theta_T``.

# Arguments

- [`rw::ReweightingSolver{T}`](@ref Deborah.Miriam.Reweighting.ReweightingSolver): The solver containing ensemble data and offsets ``f_k``.
- [`paramT::Ensemble.Params{T}`](@ref Deborah.Miriam.Ensemble.Params): Target parameter set for``\\kappa_T``.

# Returns

- `Nothing` (fills `rw.w` in place with weights of length `rw.nconf_all`).

---

# Notes

Each configuration ``\\mathcal{U}_{j;m}`` with ensemble ``j``, gauge configuration index ``m`` is
assigned a weight that incorporates action differences and the previously determined
free energy offsets ``f_k``. These weights allow expectation values at ``\\kappa_T`` to
be estimated from configurations generated at multiple ensembles.

## Formula

For a configuration ``\\mathcal{U}_{j;m}``, the reweighting weight is given by
```math
  w_{j;m}(\\kappa_T) = \\frac{\\exp \\!\\left[ - \\Delta S_{j;m}(\\kappa_T)
      \\right]}{ \\displaystyle\\sum_{k=1}^{n_{\\mathrm{ens}}} N_k \\; \\exp \\!\\left[ -
      \\Delta S_{j;m}(\\kappa_k) + f_k - \\mathcal{X} \\right] } \\,,
```
where

* ``N_k``: number of configurations in ensemble ``k``,
* ``f_k``: free energy offset for ensemble ``k``.
* ``\\mathcal{X}``: stabilizer of the exponentials for numerical safety (see below).
* ``\\Delta S_{j;m}(\\kappa_k) \\equiv S(\\kappa_k; \\mathcal{U}_{j;m}) - S(\\kappa_j;\\mathcal{U}_{j;m})``: action difference evaluated on configuration ``m`` from ensemble ``j`` under parameter ``\\kappa_k`` where we note that
```math
  \\exp \\left[ -\\,\\Delta S_{j;m}(\\kappa_k) \\right] = \\left[ \\left(
    \\frac{\\det M(\\kappa_k)}{\\det M(\\kappa_j)} \\right)^{N_{\\text{f}}}
    \\right]_{j;m} \\,.
```

## Numerical Stability

Direct evaluation of exponentials can cause overflow or underflow.
To stabilize, the implementation uses the **log-sum-exp trick**:
first determine the global maximum exponent across all configurations
```math
\\mathcal{X}
= \\max_{1 \\le i,j \\le n_{\\mathrm{ens}}} \\left( \\max_{1 \\le m \\le N_j}
\\left\\{
\\Delta S_{i;m}(\\kappa_T) - \\Delta S_{i;m}(\\kappa_j) + f_j
\\right\\} \\right) \\,, 
```
and then subtract this value before exponentiation:
```math
\\sum_{j=k}^{n_{\\mathrm{ens}}}
N_k \\exp \\left[
\\Delta S_{j;m}(\\kappa_T) - \\Delta S_{j;m}(\\kappa_k) + f_j - \\mathcal{X}
\\right].
```
Since every term in the denominator is shifted by the same constant
(``-\\mathcal{X}``), the final weights
```math
w_{j;m}(\\kappa_T) =
\\frac{1}{\\displaystyle{\\sum_k N_k \\exp[\\cdots - \\mathcal{X}]}}
```
remain unchanged. Only the intermediate exponential evaluations are affected,
which makes the computation numerically safe.

## Usage in Observables

The target expectation value is then estimated as

```math
  \\left\\langle \\Omega(\\kappa_T;\\mathcal{U}) \\right\\rangle =
  \\frac{\\displaystyle \\sum_{j=1}^{n_{\\mathrm{ens}}} \\sum_{m=1}^{N_j}
    w_{j;m}(\\kappa_T) \\; \\Omega(\\kappa_T;\\mathcal{U}_{j;m})}
       {\\displaystyle \\sum_{j=1}^{n_{\\mathrm{ens}}} \\sum_{m=1}^{N_j}
         w_{j;m}(\\kappa_T)} \\, .
```
"""
function calc_w!(
    rw::ReweightingSolver{T}, 
    paramT::Ensemble.Params{T}
)::Nothing where T
    ens = rw.ens.data
    nens = length(ens)

    # ── Step 1: Determine normalization factor (log-sum-exp trick) ──
    norm = -typemax(T)  # Initial minimum value
    for i in 1:nens
        for iconf in 1:ens[i].nconf
            for j in 1:nens
                tmp = EnsembleUtils.dS(ens[i], paramT, iconf) - EnsembleUtils.dS(ens[i], ens[j].param, iconf) + ens[j].f
                norm = max(norm, tmp)
            end
        end
    end

    # ── Step 2: Compute weights with normalization ──
    rw.w = Vector{T}(undef, rw.nconf_all)
    ii = 1  # Flat configuration index across all ensembles

    for i in 1:nens
        for iconf in 1:ens[i].nconf
            denom = zero(T)
            for j in 1:nens
                exponent = EnsembleUtils.dS(ens[i], paramT, iconf) - EnsembleUtils.dS(ens[i], ens[j].param, iconf) + ens[j].f - norm
                denom += ens[j].nconf * exp(exponent)
            end
            rw.w[ii] = 1 / denom
            ii += 1
        end
    end

    return nothing
end

end  # module Reweighting