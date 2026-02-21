# ============================================================================
# src/DeborahCore/MLSequence.jl
#
# Author: Benjamin Jaedon Choi (https://github.com/saintbenjamin)
# Affiliation: Center for Computational Sciences, University of Tsukuba
# Address: 1-1-1 Tennodai, Tsukuba, Ibaraki 305-8577 Japan
# Contact: benchoi [at] ccs.tsukuba.ac.jp (replace [at] with @)
# License: MIT License
# ============================================================================

module MLSequence

import ..Sarah.JobLoggerTools
import ..Sarah.DatasetPartitioner
import ..Sarah.NameParser
import ..PathConfigBuilderDeborah
import ..MLInputPreparer
import ..MLSequenceLightGBM
import ..MLSequenceMiddleGBM
import ..MLSequencePyCallLightGBM
import ..MLSequenceLasso
import ..MLSequenceRidge

"""
    ml_sequence(
        cfg_model::String,
        ML_inputs::MLInputPreparer.MLInputBundle,
        partition::DatasetPartitioner.DatasetPartitionInfo,
        paths::PathConfigBuilderDeborah.DeborahPathConfig,
        X_list::Vector{String},
        jobid::Union{Nothing, String}=nothing
    ) -> Dict{Symbol, Matrix}

Main ML execution dispatcher for [`Deborah.DeborahCore`](@ref).
Chooses the appropriate model backend (e.g., [`LightGBM`](https://juliaai.github.io/MLJ.jl/stable/models/LGBMRegressor_LightGBM/#LGBMRegressor_LightGBM), [`Lasso`](https://juliaai.github.io/MLJ.jl/stable/models/LassoRegressor_MLJLinearModels/#LassoRegressor_MLJLinearModels), [`Ridge`](https://juliaai.github.io/MLJ.jl/stable/models/RidgeRegressor_MLJLinearModels/#RidgeRegressor_MLJLinearModels), etc.)
based on `cfg_model`, and runs training ``+`` prediction to generate
output matrices for each configuration group.

# Arguments
- `cfg_model::String` : Model name string (e.g., `"LightGBM"`, `"Lasso"`, `"PyGBM"`, etc.)
- [`ML_inputs::MLInputPreparer.MLInputBundle`](@ref Deborah.DeborahCore.MLInputPreparer.MLInputBundle) : Preprocessed input data bundle (``X``, ``Y`` vectors and indices).
- [`partition::DatasetPartitioner.DatasetPartitionInfo`](@ref Deborah.Sarah.DatasetPartitioner.DatasetPartitionInfo) : Partitioning metadata (`lb`/`tr`/`bc`/`ul` counts and indices).
- [`paths::PathConfigBuilderDeborah.DeborahPathConfig`](@ref Deborah.DeborahCore.PathConfigBuilderDeborah.DeborahPathConfig) : File path configuration (output directory, info file, etc.).
- `X_list::Vector{String}` : List of input feature keys.
- `jobid::Union{Nothing, String}` : Optional job ID used for logging.

# Returns
- `Dict{Symbol, Matrix}` : A dictionary containing predicted ``Y`` matrices for each group.
    - Keys: `:YP_tr`, `:YP_bc`, `:YP_ul`
    - Each value is a matrix of size ``(N_\\text{cfg}, N_\\text{src})`` reconstructed from prediction vectors.
"""
function ml_sequence(
    cfg_model::String,
    ML_inputs::MLInputPreparer.MLInputBundle,
    partition::DatasetPartitioner.DatasetPartitionInfo,
    paths::PathConfigBuilderDeborah.DeborahPathConfig,
    X_list::Vector{String}, 
    jobid::Union{Nothing, String}=nothing
)::Dict{Symbol, Matrix}

    X_data      = ML_inputs.X_data
    Y_tr_vec    = ML_inputs.Y_tr_vec
    Y_bc_vec    = ML_inputs.Y_bc_vec
    Y_ul_vec    = ML_inputs.Y_ul_vec
    Y_lb_vec    = ML_inputs.Y_lb_vec
    tr_conf_arr = ML_inputs.tr_conf_arr
    bc_conf_arr = ML_inputs.bc_conf_arr
    ul_conf_arr = ML_inputs.ul_conf_arr
    model_tag   = NameParser.model_suffix(cfg_model, jobid)

    if cfg_model == "LightGBM"
        _, Y_mats = MLSequenceLightGBM.ml_sequence_LightGBM(
            model_tag=model_tag,
            X_data=X_data,
            Y_tr_vec=Y_tr_vec,
            Y_bc_vec=Y_bc_vec,
            Y_ul_vec=Y_ul_vec,
            Y_lb_vec=Y_lb_vec,
            tr_conf_arr=tr_conf_arr,
            bc_conf_arr=bc_conf_arr,
            ul_conf_arr=ul_conf_arr,
            partition=partition,
            X_list=X_list,
            paths=paths,
            jobid=jobid
        )
        return Y_mats
    elseif cfg_model == "MiddleGBM"
        _, Y_mats = MLSequenceMiddleGBM.ml_sequence_MiddleGBM(
            model_tag=model_tag,
            X_data=X_data,
            Y_tr_vec=Y_tr_vec,
            Y_bc_vec=Y_bc_vec,
            Y_ul_vec=Y_ul_vec,
            Y_lb_vec=Y_lb_vec,
            tr_conf_arr=tr_conf_arr,
            bc_conf_arr=bc_conf_arr,
            ul_conf_arr=ul_conf_arr,
            partition=partition,
            X_list=X_list,
            paths=paths,
            jobid=jobid
        )
        return Y_mats
    elseif cfg_model == "PyGBM"
        return MLSequencePyCallLightGBM.ml_sequence_PyCallLightGBM(
            model_tag=model_tag,
            X_data=X_data,
            Y_tr_vec=Y_tr_vec,
            Y_bc_vec=Y_bc_vec,
            Y_ul_vec=Y_ul_vec,
            Y_lb_vec=Y_lb_vec,
            tr_conf_arr=tr_conf_arr,
            bc_conf_arr=bc_conf_arr,
            ul_conf_arr=ul_conf_arr,
            partition=partition,
            X_list=X_list,
            paths=paths,
            jobid=jobid
        )
    elseif cfg_model == "Lasso"
        _, Y_mats = MLSequenceLasso.ml_sequence_Lasso(
            model_tag=model_tag,
            X_data=X_data,
            Y_tr_vec=Y_tr_vec,
            Y_bc_vec=Y_bc_vec,
            Y_ul_vec=Y_ul_vec,
            Y_lb_vec=Y_lb_vec,
            tr_conf_arr=tr_conf_arr,
            bc_conf_arr=bc_conf_arr,
            ul_conf_arr=ul_conf_arr,
            partition=partition,
            X_list=X_list,
            paths=paths,
            jobid=jobid
        )
        return Y_mats
    elseif cfg_model == "Ridge"
        _, Y_mats = MLSequenceRidge.ml_sequence_Ridge(
            model_tag=model_tag,
            X_data=X_data,
            Y_tr_vec=Y_tr_vec,
            Y_bc_vec=Y_bc_vec,
            Y_ul_vec=Y_ul_vec,
            Y_lb_vec=Y_lb_vec,
            tr_conf_arr=tr_conf_arr,
            bc_conf_arr=bc_conf_arr,
            ul_conf_arr=ul_conf_arr,
            partition=partition,
            X_list=X_list,
            paths=paths,
            jobid=jobid
        )
        return Y_mats
    else
        JobLoggerTools.error_benji("Unknown model", jobid)
    end
end

end  # module MLSequence