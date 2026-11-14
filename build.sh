#!/usr/bin/env bash
# Linux native extensions (Python.h + NumPy C API), including WSL2 and Colab.
# Run inside the Linux Python environment used for research:
#   bash build.sh [ops] [backtest] [search]     (no arguments = all three)
# CUDA kernels are compiled by CuPy at explicit runtime preparation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Derive the ABI and headers from the active interpreter. Windows Python is not
# a Linux build target, including when Windows executables appear on WSL's PATH.
PYTHON_CONFIG=$(python3 - <<'PY'
import sys
import sysconfig

if sys.platform != 'linux':
    raise SystemExit('Build requires Linux Python; on Windows use a WSL2 Linux environment.')
import numpy
print(sysconfig.get_config_var('EXT_SUFFIX'))
print(sysconfig.get_paths()['include'])
print(numpy.get_include())
PY
)
mapfile -t PYTHON_PATHS <<< "$PYTHON_CONFIG"
EXT="${PYTHON_PATHS[0]}"
INCLUDES=("-I${PYTHON_PATHS[1]}" "-I${PYTHON_PATHS[2]}" -Ifoundry/include)
WARN=(-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
# EXTRA_CFLAGS is a build-only hook used by the sanitizer CI job.
read -r -a EXTRA <<< "${EXTRA_CFLAGS:-}"
TARGETS=("$@")
if [ "$#" -eq 0 ]; then
    TARGETS=(ops backtest search)
fi

for target in "${TARGETS[@]}"; do
    case "$target" in
        ops)
            output=foundry/ops/_ops
            flags=(-O3 -march=native -fno-fast-math -ffp-contract=off -fno-strict-aliasing -fopenmp)
            sources=(foundry/ops/src/ops_ts.cpp foundry/ops/src/ops_xs.cpp
                     foundry/ops/src/ops_inject_ts.cpp foundry/ops/src/module.cpp)
            ;;
        backtest)
            output=foundry/backtest/_bt
            flags=(-O3 -march=native -ffast-math -fno-finite-math-only -fno-strict-aliasing -fopenmp)
            sources=(foundry/backtest/src/kernel.cpp foundry/backtest/src/perpetual.cpp
                     foundry/backtest/src/module.cpp)
            ;;
        search)
            output=foundry/search/_gfn
            flags=(-O3)
            sources=(foundry/search/src/gfn.cpp)
            ;;
        *)
            printf 'Unknown build target: %s\nUsage: bash build.sh [ops] [backtest] [search]\n' "$target" >&2
            exit 2
            ;;
    esac
    g++ -std=c++17 "${flags[@]}" "${EXTRA[@]}" -fPIC "${WARN[@]}" -shared \
        "-Ifoundry/$target/src" "${INCLUDES[@]}" "${sources[@]}" -o "$output$EXT" -lgomp -lm
    printf 'Built %s\n' "$ROOT/$output$EXT"
done
