#!/bin/sh
# Attempt to re-run with modern bash if available
if [ -z "$BASH_VERSION" ] || [ "${BASH_VERSINFO:-0}" -lt 4 ]; then
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    else
        echo "Error: modern bash (v4+) is required." >&2
        exit 1
    fi
fi
# Now in real bash 4+

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

# shellcheck disable=SC2034
location=./sample/nf4_clover_wilson_finiteT

# shellcheck disable=SC2034
declare -A group_map
# shellcheck disable=SC2034
declare -A prefix_seen
# shellcheck disable=SC2034
prefix_list=()

main() {
    collect_and_sort_ensembles
    group_ensembles_by_prefix
    rewrite_ids_by_group
}

main "$@"