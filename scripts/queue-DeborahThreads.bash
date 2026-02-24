#!/bin/bash
program=DeborahThreadsEntry.jl
batch_size=55

location=sample/nf4_clover_wilson_finiteT
ranseed="850528"
N_bs="1000"
bootstrap_method="nonoverlapping"

# label=("5"  "10" "15" "20" "25" 
#        "30" "35" "40" "45" "50" 
#        "55" "60" "65" "70" "75")
label=("5"  "10" "15" "20" "25" 
       "30" "35" "40" "45" "50")
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

[ -f "./utils-Deborah.shlib" ] && source "./utils-Deborah.shlib"
[ -f "${SCRIPT_DIR}/utils-Deborah.shlib" ] && source "${SCRIPT_DIR}/utils-Deborah.shlib"

if [[ -z $1 ]]; then
    echo "Error: Missing required argument."
    print_usage_DeborahThreads
    exit 1
fi

io_str="$1"
model="$2"
fix_ens="$3"
binning="$4"

blk_size=$binning
bin_size=$binning

declare -a io_X
io_Y=""

parse_abbreviation "$io_str" io_X io_Y

io_X_joined=$(printf '"%s",' "${io_X[@]}")
io_X_joined="[${io_X_joined%,}]" 

io_X_underbar=$(printf '%s_' "${io_X[@]}")
io_X_underbar="${io_X_underbar%_}"

io_X_Y="${io_X_underbar}_${io_Y}"

model_tag=$(parse_model_tag "$model")

read_column_X=$(yes "$read_column" | head -n "${#io_X[@]}" | paste -sd, -)
read_column_X="[$read_column_X]"

read_column_Y="${read_column}"

# ----------------------------------------
# Setup directory and file layout
# ----------------------------------------
HERE=$(pwd -P)
HERE_location=${HERE}/${location}

parse_ensemble_info_zero_idx "${HERE_location}"

# shellcheck disable=SC2154
ensemble="L${L[$fix_ens]}T${T[$fix_ens]}b${b[$fix_ens]}k${k[$fix_ens]}"

# ----------------------------------------
# Partition definitions
# ----------------------------------------

if [ "$use_abbreviation" == "true" ]; then
    my_anly_dir="${HERE_location}/${analysis_header}_${ensemble}/${analysis_header}_${ensemble}_${io_str}_${model_tag}"
else
    my_anly_dir="${HERE_location}/${analysis_header}_${ensemble}/${analysis_header}_${ensemble}_${io_X_Y}_${model_tag}"
fi
# mkdir -p "${my_anly_dir}"

label_joined=$(printf '"%s",' "${label[@]}")
label_joined="[${label_joined%,}]" 

train_joined=$(printf '"%s",' "${train[@]}")
train_joined="[${train_joined%,}]" 

run_parallel_batch_DeborahThreads \
    "$batch_size" "$program" \
    "${label_joined}" "${train_joined}" \
    "$location" "$ensemble" "$analysis_header" \
    "$io_str" "$io_X_joined" "$io_Y" "$model_tag" "$model" \
    "$IDX_shift" "$read_column_X" "$read_column_Y" "$dump_X" \
    "$ranseed" "$N_bs" "$blk_size" "$bin_size" \
    "$my_anly_dir" "$use_abbreviation" "${io_X_Y}" "$index_column" "$bootstrap_method"