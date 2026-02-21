#!/bin/bash
program=MiriamEntry.jl
batch_size=55

location=nf4_clover_wilson_finiteT
ranseed="850528"
N_bs="1000"
bootstrap_method="nonoverlapping"

dump_original="false"
maxiter="1000"
eps="1e-13"
nkappaT="100"

# shellcheck disable=SC2034
# label=("5"  "10" "15" "20" "25" 
#        "30" "35" "40" "45" "50" 
#        "55" "60" "65" "70" "75")
# label=("1" "2" "3" "4" "5"
#        "6" "7" "8" "9" "10"
#        "11" "12" "13" "14" "15"
#        "16" "17" "18" "19" "20"
#        "21" "22" "23" "24" "25")
label=("5"  "10" "15" "20" "25" 
       "30" "35" "40" "45" "50")
# label=("55" "60" "65" "70" "75" 
#        "80" "85" "90" "95")
# label=("5"  "10" "15" "20" "25" 
#        "30" "35" "40" "45" "50" 
#        "55" "60" "65" "70" "75" 
#        "80" "85" "90" "95")
# label=("5"  "10" "15" "20" "25")
# shellcheck disable=SC2034
train=("0" "10" "20" "30" "40" "50" "60" "70" "80" "90" "100")


# factors used in Deborah only
analysis_header="analysis"
IDX_shift="0"
read_column="1"
index_column="3"
dump_X="false"
use_abbreviation="true"

# Resolve real path
get_script_dir() {
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do
        DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
    done
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    echo "$DIR"
}
SCRIPT_DIR=$(get_script_dir)

# Source shared lib
[ -f "./common.shlib" ] && source "./common.shlib"
[ -f "${SCRIPT_DIR}/common.shlib" ] && source "${SCRIPT_DIR}/common.shlib"

[ -f "./utils-Miriam.shlib" ] && source "./utils-Miriam.shlib"
[ -f "${SCRIPT_DIR}/utils-Miriam.shlib" ] && source "${SCRIPT_DIR}/utils-Miriam.shlib"

mode="$1"
shift

if [[ "$mode" != "single" && "$mode" != "parallel" ]]; then
    echo "[ERROR] First argument must be 'single' or 'parallel'"
    print_usage_Miriam
fi

if [ "$mode" = "single" ]; then
    manual_label="$1"
    manual_train="$2"
    M1_str="$3"
    M1_model="$4"
    M2_str="$5"
    M2_model="$6"
    M3_str="$7"
    M3_model="$8"
    M4_str="$9"
    M4_model="${10}"
    target_gidx="${11}"
    binning="${12}"
else
    M1_str="$1"
    M1_model="$2"
    M2_str="$3"
    M2_model="$4"
    M3_str="$5"
    M3_model="$6"
    M4_str="$7"
    M4_model="$8"
    target_gidx="$9"
    binning="${10}"
fi

blk_size=$binning
bin_size=$binning

# shellcheck disable=SC2034
declare -a M1_X
# shellcheck disable=SC2034
M1_Y=""
parse_abbreviation "$M1_str" M1_X M1_Y
M1_X_joined=$(printf '"%s",' "${M1_X[@]}")
M1_X_joined="[${M1_X_joined%,}]"
M1_X_underbar=$(printf '%s_' "${M1_X[@]}")
M1_X_underbar="${M1_X_underbar%_}"
M1_X_Y="${M1_X_underbar}_${M1_Y}"
M1_tag=$(parse_model_tag "$M1_model")
M1_read_column_X=$(yes "$read_column" | head -n "${#M1_X[@]}" | paste -sd, -)
M1_read_column_X="[$M1_read_column_X]"
M1_read_column_Y="${read_column}"
M1_index_column="${index_column}"

# shellcheck disable=SC2034
declare -a M2_X
# shellcheck disable=SC2034
M2_Y=""
parse_abbreviation "$M2_str" M2_X M2_Y
M2_X_joined=$(printf '"%s",' "${M2_X[@]}")
M2_X_joined="[${M2_X_joined%,}]" 
M2_X_underbar=$(printf '%s_' "${M2_X[@]}")
M2_X_underbar="${M2_X_underbar%_}"
M2_X_Y="${M1_X_underbar}_${M2_Y}"
M2_tag=$(parse_model_tag "$M2_model")
M2_read_column_X=$(yes "$read_column" | head -n "${#M2_X[@]}" | paste -sd, -)
M2_read_column_X="[$M2_read_column_X]"
M2_read_column_Y="${read_column}"
M2_index_column="${index_column}"

# shellcheck disable=SC2034
declare -a M3_X
# shellcheck disable=SC2034
M3_Y=""
parse_abbreviation "$M3_str" M3_X M3_Y
M3_X_joined=$(printf '"%s",' "${M3_X[@]}")
M3_X_joined="[${M3_X_joined%,}]" 
M3_X_underbar=$(printf '%s_' "${M3_X[@]}")
M3_X_underbar="${M3_X_underbar%_}"
M3_X_Y="${M3_X_underbar}_${M3_Y}"
M3_tag=$(parse_model_tag "$M3_model")
M3_read_column_X=$(yes "$read_column" | head -n "${#M3_X[@]}" | paste -sd, -)
M3_read_column_X="[$M3_read_column_X]"
M3_read_column_Y="${read_column}"
M3_index_column="${index_column}"

# shellcheck disable=SC2034
declare -a M4_X
# shellcheck disable=SC2034
M4_Y=""
parse_abbreviation "$M4_str" M4_X M4_Y
M4_X_joined=$(printf '"%s",' "${M4_X[@]}")
M4_X_joined="[${M4_X_joined%,}]" 
M4_X_underbar=$(printf '%s_' "${M4_X[@]}")
M4_X_underbar="${M4_X_underbar%_}"
M4_X_Y="${M4_X_underbar}_${M4_Y}"
M4_tag=$(parse_model_tag "$M4_model")
M4_read_column_X=$(yes "$read_column" | head -n "${#M4_X[@]}" | paste -sd, -)
M4_read_column_X="[$M4_read_column_X]"
M4_read_column_Y="${read_column}"
M4_index_column="${index_column}"

# ----------------------------------------
# Setup directory and file layout
# ----------------------------------------
HERE=$(pwd -P)
HERE_location=${HERE}/${location}

# shellcheck disable=SC2034
declare -g -a ens_arr_zero_idx

parse_ensemble_info_zero_idx "${HERE_location}"

group_and_parse_ensembles "${HERE_location}" "${target_gidx}"

# shellcheck disable=SC2154
if [[ "${target_gidx}" =~ ^[0-9]+$ ]]; then
  multi_ensemble="${prefix_list[$target_gidx]}"
else
  multi_ensemble="${target_gidx}"
fi

# shellcheck disable=SC2154
ensembles=$(printf '"%s",' "${arr_ensem[@]}")
ensembles="[${ensembles%,}]"

# shellcheck disable=SC2154
kappa_list=$(printf '%s,' "${arr_kappa[@]}")
kappa_list="[${kappa_list%,}]"


# ----------------------------------------
# Partition definitions
# ----------------------------------------
my_col_dir="${HERE_location}"/${analysis_header}_"${multi_ensemble}"
# mkdir -p "${my_col_dir}"

if [ "$use_abbreviation" == "true" ]; then
    my_anly_dir="${my_col_dir}"/${analysis_header}_"${multi_ensemble}"_"${M1_str}"_${M1_tag}_"${M2_str}"_${M2_tag}_"${M3_str}"_${M3_tag}_"${M4_str}"_${M4_tag}
else
    my_anly_dir="${my_col_dir}"/${analysis_header}_"${multi_ensemble}"_"${M1_X_Y}"_${M1_tag}_"${M2_X_Y}"_${M2_tag}_"${M3_X_Y}"_${M3_tag}_"${M4_X_Y}"_${M4_tag}
fi
# mkdir -p "${my_anly_dir}"

if [ "$mode" = "parallel" ]; then

    run_parallel_batch_Miriam \
        "$batch_size" "$program" \
        label[@] train[@] \
        "$location" "$multi_ensemble" "${ensembles}" \
        "$M1_str" "$M1_X_joined" "$M1_Y" "$M1_tag" "$M1_model" "$M1_read_column_X" "$M1_read_column_Y" "$M1_index_column" "${M1_X_Y}" \
        "$M2_str" "$M2_X_joined" "$M2_Y" "$M2_tag" "$M2_model" "$M2_read_column_X" "$M2_read_column_Y" "$M2_index_column" "${M2_X_Y}" \
        "$M3_str" "$M3_X_joined" "$M3_Y" "$M3_tag" "$M3_model" "$M3_read_column_X" "$M3_read_column_Y" "$M3_index_column" "${M3_X_Y}" \
        "$M4_str" "$M4_X_joined" "$M4_Y" "$M4_tag" "$M4_model" "$M4_read_column_X" "$M4_read_column_Y" "$M4_index_column" "${M4_X_Y}" \
        "$dump_original" \
        "$ns" "$nt" "$nf" "$beta" "$csw" \
        "${kappa_list}" \
        "$maxiter" "$eps" \
        "$bin_size" "$ranseed" "$N_bs" "$blk_size" \
        "$nkappaT" \
        "${analysis_header}" "${IDX_shift}" "${dump_X}" \
        "$my_anly_dir" \
        "$use_abbreviation" "$bootstrap_method"

    label_joined=$(printf '"%s",' "${label[@]}")
    label_joined="[${label_joined%,}]" 

    train_joined=$(printf '"%s",' "${train[@]}")
    train_joined="[${train_joined%,}]" 

    run_parallel_batch_MiriamDocument \
        "$batch_size" "$program" \
        "${label_joined}" "${train_joined}" \
        "$location" "$multi_ensemble" "${ensembles}" \
        "$M1_str" "$M1_X_joined" "$M1_Y" "$M1_tag" "$M1_model" "$M1_read_column_X" "$M1_read_column_Y" "$M1_index_column" "${M1_X_Y}" \
        "$M2_str" "$M2_X_joined" "$M2_Y" "$M2_tag" "$M2_model" "$M2_read_column_X" "$M2_read_column_Y" "$M2_index_column" "${M2_X_Y}" \
        "$M3_str" "$M3_X_joined" "$M3_Y" "$M3_tag" "$M3_model" "$M3_read_column_X" "$M3_read_column_Y" "$M3_index_column" "${M3_X_Y}" \
        "$M4_str" "$M4_X_joined" "$M4_Y" "$M4_tag" "$M4_model" "$M4_read_column_X" "$M4_read_column_Y" "$M4_index_column" "${M4_X_Y}" \
        "$dump_original" \
        "$ns" "$nt" "$nf" "$beta" "$csw" \
        "${kappa_list}" \
        "$maxiter" "$eps" \
        "$bin_size" "$ranseed" "$N_bs" "$blk_size" \
        "$nkappaT" \
        "${analysis_header}" "${IDX_shift}" "${dump_X}" \
        "$my_anly_dir" \
        "$use_abbreviation" "$bootstrap_method"

elif [ "$mode" = "single" ]; then

    run_single_job_Miriam \
        "$batch_size" "$program" \
        label[@] train[@] \
        "$location" "$multi_ensemble" "${ensembles}" \
        "$M1_str" "$M1_X_joined" "$M1_Y" "$M1_tag" "$M1_model" "$M1_read_column_X" "$M1_read_column_Y" "$M1_index_column" "${M1_X_Y}" \
        "$M2_str" "$M2_X_joined" "$M2_Y" "$M2_tag" "$M2_model" "$M2_read_column_X" "$M2_read_column_Y" "$M2_index_column" "${M2_X_Y}" \
        "$M3_str" "$M3_X_joined" "$M3_Y" "$M3_tag" "$M3_model" "$M3_read_column_X" "$M3_read_column_Y" "$M3_index_column" "${M3_X_Y}" \
        "$M4_str" "$M4_X_joined" "$M4_Y" "$M4_tag" "$M4_model" "$M4_read_column_X" "$M4_read_column_Y" "$M4_index_column" "${M4_X_Y}" \
        "$dump_original" \
        "$ns" "$nt" "$nf" "$beta" "$csw" \
        "${kappa_list}" \
        "$maxiter" "$eps" \
        "$bin_size" "$ranseed" "$N_bs" "$blk_size" \
        "$nkappaT" \
        "${analysis_header}" "${IDX_shift}" "${dump_X}" \
        "$my_anly_dir" \
        "$use_abbreviation" \
        "$manual_label" "$manual_train" "$bootstrap_method"

else
    echo "Error: unknown mode: $mode"
    exit 1
fi