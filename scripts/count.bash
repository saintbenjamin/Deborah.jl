#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Count Julia source stats under ../src:
# - Number of .jl files
# - Total lines
# - Comment lines (lines starting with optional whitespace + #)
# - Estimated code lines (total - comment)
# - Files with module declaration (at least one `module` or `baremodule`)
# - Nested module declarations (module declarations beyond the first per file)
# - Function count (block-form `function ...` + one-liner `name(...) = ...`)
#
# Notes:
# - This is a best-effort heuristic using grep/awk; it is not AST-precise.
# - "Files with module declaration" is a FILE count.
# - "Nested module declarations" is a DECLARATION count (extra `module` lines
#   beyond the first in each file that contains at least one module).
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"

echo "Analyzing .jl files under $SRC_DIR..."

# Find all .jl files
mapfile -t jl_files < <(find "$SRC_DIR" -type f -name "*.jl")

num_files="${#jl_files[@]}"
if [[ "$num_files" -eq 0 ]]; then
  echo "-------------------------------------"
  echo "Number of .jl files                : 0"
  echo "Total lines (all)                  : 0"
  echo "Comment lines (with #)             : 0"
  echo "Code lines (estimated)             : 0"
  echo "Files with module declaration      : 0"
  echo "Nested module declarations         : 0"
  echo "Functions (total, est)             : 0"
  echo "  - block 'function'               : 0"
  echo "  - one-liner 'f(...)= '           : 0"
  echo "-------------------------------------"
  exit 0
fi

# Count total lines across all files
total_lines=$(wc -l "${jl_files[@]}" | awk 'END {print $1}')

# Count comment lines across all files (lines starting with optional whitespace + #)
comment_lines=$(grep -chE '^[[:space:]]*#' "${jl_files[@]}" | paste -sd+ - | bc)

# Estimate code lines as total - comments
code_lines=$((total_lines - comment_lines))

# Regex for module declarations
module_decl_re='^[[:space:]]*(baremodule|module)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\b'

# Files with at least one module/baremodule declaration (FILE count)
files_with_module=$(
  grep -R --include="*.jl" -lE "$module_decl_re" "$SRC_DIR" \
  | wc -l
)

# Total module declarations (DECLARATION count)
total_module_decls=$(
  grep -R --include="*.jl" -hE "$module_decl_re" "$SRC_DIR" \
  | wc -l
)

# Nested module declarations = declarations beyond the first per module-bearing file
# (DECLARATION count; not a file count)
nested_module_decls=$((total_module_decls - files_with_module))
if [[ "$nested_module_decls" -lt 0 ]]; then
  nested_module_decls=0
fi

# Functions: block-form `function name...` (heuristic)
function_block=$(
  grep -R --include="*.jl" -hE '^[[:space:]]*function[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\b' "$SRC_DIR" \
  | wc -l
)

# Functions: one-liners `name(args) = ...` (heuristic)
function_oneline=$(
  grep -R --include="*.jl" -hE '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\([^)]*\)[[:space:]]*=' "$SRC_DIR" \
  | wc -l
)

function_total_est=$((function_block + function_oneline))

# Output
echo "-------------------------------------"
echo "Number of .jl files                : $num_files"
echo "Total lines (all)                  : $total_lines"
echo "Comment lines (with #)             : $comment_lines"
echo "Code lines (estimated)             : $code_lines"
echo "-------------------------------------"
echo "Files with module declaration      : $files_with_module"
echo "Nested module declarations         : $nested_module_decls"
echo "-------------------------------------"
echo "Functions (total, est)             : $function_total_est"
echo "  - block 'function'               : $function_block"
echo "  - one-liner 'f(...)= '           : $function_oneline"
echo "-------------------------------------"