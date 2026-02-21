# ============================================================================
# src/DeborahCore/XYMLInfoGenerator.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module XYMLInfoGenerator

import Printf: @printf, @sprintf
import ..Sarah.JobLoggerTools

"""
    gen_XY_ML_info(
        X_info::Array{T,3},
        conf_arr::Vector{Int},
        lb_idx::Vector{Int},
        tr_idx::Vector{Int},
        bc_idx::Vector{Int},
        ul_idx::Vector{Int},
        N_lb::Int, N_tr::Int, N_bc::Int, N_ul::Int,
        name_prefix::String, overall_name::String,
        analysis_dir::String,
        read_column::Int;
        use_avg::Bool=true, dump::Bool=true,
        jobid::Union{Nothing, String}=nothing
    ) -> Tuple{
        Array{T,2}, Array{T,2}, Array{T,2}, Array{T,2},
        Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}
    } where T<:Real

Split a full 2D input matrix `X_info` into four named subsets (`LB`, `TR`, `BC`, `UL`) and optionally write them to `.dat` files.

# Arguments
- `X_info::Array{T,2}`  
    Input matrix of shape ``(N_\\text{cnf}, N_\\text{src})``, typically representing a single observable across configs and sources.

- `conf_arr::Vector{Int}`  
    Full list of configuration indices (``\\text{length} = N_\\text{cnf}``), aligned with the rows of `X_info`.

- `lb_idx::Vector{Int}`  
    Index list for the Labeled (`LB`) set.

- `tr_idx::Vector{Int}`  
    Index list for the Training (`TR`) set.

- `bc_idx::Vector{Int}`  
    Index list for the Bias Correction (`BC`) set.

- `ul_idx::Vector{Int}`  
    Index list for the Unlabeled (`UL`) set.

- `N_lb::Int`, `N_tr::Int`, `N_bc::Int`, `N_ul::Int`  
    Number of configurations in each group (used for preallocation and checks).

- `name_prefix::String`  
    Prefix used in output file names.

- `overall_name::String`  
    Global name used for logging and file identification.

- `analysis_dir::String`  
    Output directory path where files will be dumped (if `dump=true`).

- `read_column::Int`  
    ``1``-based index of the column used to extract the observable (for file naming or meta info).

- `use_avg::Bool=true`  
    If true, marks `jval` as `-1` in the output file to signal "averaged" content.

- `dump::Bool=true`  
    If true, writes `.dat` files to `analysis_dir`.

- `jobid::Union{Nothing,String}`  
    Optional job ID for logging or progress tracking.

# Returns
`Tuple` of eight elements:
1. `X_lb_info  :: Array{T,3}`
2. `X_tr_info  :: Array{T,3}`
3. `X_bc_info  :: Array{T,3}`
4. `X_ul_info  :: Array{T,3}`
5. `lb_conf_arr:: Vector{Int}`
6. `tr_conf_arr:: Vector{Int}`
7. `bc_conf_arr:: Vector{Int}`
8. `ul_conf_arr:: Vector{Int}`

Each `X_*_info` has shape ``(\\texttt{column\\_idx}, N_\\text{set}, N_\\text{src})`` and corresponds to the respective index group.
"""
function gen_XY_ML_info(
    X_info::Array{T, 3},
    conf_arr::Vector{Int},
    lb_idx::Vector{Int},
    tr_idx::Vector{Int},
    bc_idx::Vector{Int},
    ul_idx::Vector{Int},
    N_lb::Int, N_tr::Int, N_bc::Int, N_ul::Int,
    name_prefix::String, overall_name::String,
    analysis_dir::String,
    read_column::Int;
    use_avg::Bool=true, dump::Bool=true, 
    jobid::Union{Nothing, String}=nothing
)::Tuple{
    Array{T,3}, Array{T,3}, Array{T,3}, Array{T,3},
    Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}
} where T<:Real

    N_cnf = size(X_info,2)
    N_src = size(X_info,3)
    
    X_lb_info = zeros(1, N_lb, N_src)
    X_tr_info = zeros(1, N_tr, N_src)
    X_bc_info = zeros(1, N_bc, N_src)
    X_ul_info = zeros(1, N_ul, N_src)

    lb_conf_arr = zeros(N_lb)
    tr_conf_arr = zeros(N_tr)
    bc_conf_arr = zeros(N_bc)
    ul_conf_arr = zeros(N_ul)

    if dump
        mkpath(analysis_dir)
        X_dat_file = analysis_dir*"/"*name_prefix*overall_name*".dat"
        open(X_dat_file, "w") do io_X
            ilb=1
            iul=1
            itr=1
            ibc=1
            for iconf in 1:N_cnf
                if ilb <= length(lb_idx) && iconf == lb_idx[ilb]
                    if itr <= length(tr_idx) && ilb == tr_idx[itr]
                        for jsrc in 1:1:N_src
                            X_tr_info[read_column,itr,jsrc] = X_info[read_column,iconf,jsrc]
                            jval = use_avg ? "A" : string(jsrc - 1)
                            @printf(io_X,"%.14e\t%d\t%s\t%d\t%s%d%s\n",X_info[read_column,iconf,jsrc],conf_arr[iconf],jval,iconf,"LB-TR[",itr,"] *")
                        end
                        tr_conf_arr[itr] = conf_arr[iconf]
                        if itr < N_tr
                            itr += 1
                        end 
                    elseif ibc <= length(bc_idx) && ilb == bc_idx[ibc]
                        for jsrc in 1:1:N_src
                            X_bc_info[read_column,ibc,jsrc] = X_info[read_column,iconf,jsrc]
                            jval = use_avg ? "A" : string(jsrc - 1)
                            @printf(io_X,"%.14e\t%d\t%s\t%d\t%s%d%s\n",X_info[read_column,iconf,jsrc],conf_arr[iconf],jval,iconf,"LB-BC[",ibc,"]")
                        end 
                        bc_conf_arr[ibc] = conf_arr[iconf]
                        if ibc < N_bc
                            ibc += 1
                        end 
                    end
                    for jsrc in 1:1:N_src
                        X_lb_info[read_column,ilb,jsrc] = X_info[read_column,iconf,jsrc]
                    end
                    lb_conf_arr[ilb] = conf_arr[iconf]
                    if ilb < N_lb
                        ilb += 1
                    end 
                elseif iul <= length(ul_idx) && iconf == ul_idx[iul]
                    for jsrc in 1:1:N_src
                        X_ul_info[read_column,iul,jsrc] = X_info[read_column,iconf,jsrc]
                        jval = use_avg ? "A" : string(jsrc - 1)
                        @printf(io_X,"%.14e\t%d\t%s\t%d\t%s%d%s\n",X_info[read_column,iconf,jsrc],conf_arr[iconf],jval,iconf,"UL[",iul,"]")
                    end 
                    ul_conf_arr[iul] = conf_arr[iconf]
                    if iul < N_ul
                        iul += 1
                    end 
                else
                    for jsrc in 1:1:N_src
                        JobLoggerTools.println_benji(@sprintf("%.14e\t%d\t%d\t%d\t%s\n",X_info[read_column,iconf,jsrc],conf_arr[iconf],(jsrc-1),iconf,"This cannot be included any set. Please check the initial condition!!"),jobid)
                    end 
                end
            end
        end
    else        
        ilb=1
        iul=1
        itr=1
        ibc=1
        for iconf in 1:N_cnf
            if ilb <= length(lb_idx) && iconf == lb_idx[ilb]
                if itr <= length(tr_idx) && ilb == tr_idx[itr]
                    for jsrc in 1:1:N_src
                        X_tr_info[read_column,itr,jsrc] = X_info[read_column,iconf,jsrc]
                    end
                    tr_conf_arr[itr] = conf_arr[iconf]
                    if itr < N_tr
                        itr += 1
                    end 
                elseif ibc <= length(bc_idx) && ilb == bc_idx[ibc]
                    for jsrc in 1:1:N_src
                        X_bc_info[read_column,ibc,jsrc] = X_info[read_column,iconf,jsrc]
                    end 
                    bc_conf_arr[ibc] = conf_arr[iconf]
                    if ibc < N_bc
                        ibc += 1
                    end 
                end
                for jsrc in 1:1:N_src
                    X_lb_info[read_column,ilb,jsrc] = X_info[read_column,iconf,jsrc]
                end
                lb_conf_arr[ilb] = conf_arr[iconf]
                if ilb < N_lb
                    ilb += 1
                end 
            elseif iul <= length(ul_idx) && iconf == ul_idx[iul]
                for jsrc in 1:1:N_src
                    X_ul_info[read_column,iul,jsrc] = X_info[read_column,iconf,jsrc]
                end 
                ul_conf_arr[iul] = conf_arr[iconf]
                if iul < N_ul
                    iul += 1
                end 
            else
                for jsrc in 1:1:N_src
                    JobLoggerTools.println_benji(@sprintf("%.14e\t%d\t%d\t%d\t%s\n",X_info[read_column,iconf,jsrc],conf_arr[iconf],(jsrc-1),iconf,"This cannot be included any set. Please check the initial condition!!"),jobid)
                end 
            end
        end
    end

    return X_lb_info, X_tr_info, X_bc_info, X_ul_info, lb_conf_arr, tr_conf_arr, bc_conf_arr, ul_conf_arr
end

end  # module XYMLInfoGenerator