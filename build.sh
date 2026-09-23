#!/usr/bin/env bash
# Build the native Python extensions (Python.h + numpy C API) next to their packages:
#   foundry/ops/_ops         formula operators, CPU reference backend
#   foundry/backtest/_bt     execution kernel, CPU reference
#   foundry/search/_gfn      portfolio-conditioned GFlowNet policy and CEM
#     ./build.sh [ops] [backtest] [search]      (no argument = all three)
# Linux -> .so; WSL or msys2 -> Windows .pyd (msys2 ucrt64 g++). Everything is derived from the target interpreter;
# the only knobs are PYTHON (target interpreter), GXX (Windows compiler) and EXTRA_CFLAGS (CI sanitizers).
# CUDA kernels are compiled at runtime by cupy (NVRTC) and have no build step.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
F="$ROOT/foundry"

# The numeric kernels keep IEEE NaN semantics (the backtest relies on NaN propagation); the policy network is built
# without -march/-ffast-math so that its sampling is reproducible across machines.
FAST="-O3 -march=native -ffast-math -fno-finite-math-only -fno-strict-aliasing -fopenmp ${EXTRA_CFLAGS:-}"
PLAIN="-O3 ${EXTRA_CFLAGS:-}"
WARN="-Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers"
TARGETS=(
  "ops/_ops|$FAST|$F/ops/src|$F/ops/src/ops_ts.cpp $F/ops/src/ops_xs.cpp $F/ops/src/ops_inject_ts.cpp $F/ops/src/module.cpp"
  "backtest/_bt|$FAST|$F/backtest/src|$F/backtest/src/kernel.cpp $F/backtest/src/module.cpp"
  "search/_gfn|$PLAIN|$F/search/src|$F/search/src/gfn.cpp"
)
if [ $# -gt 0 ]; then
    selected=()
    for t in "${TARGETS[@]}"; do
        for want in "$@"; do [ "${t%%/*}" = "$want" ] && selected+=("$t"); done
    done
    TARGETS=("${selected[@]}")
fi

IS_WSL=0
grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null && IS_WSL=1

if [ "$(uname -s)" = "Linux" ] && [ "$IS_WSL" = "0" ]; then
    PY="${PYTHON:-python3}"
    EXT="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    INC="-I$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])') -I$($PY -c 'import numpy; print(numpy.get_include())') -I$F/include"
    for t in "${TARGETS[@]}"; do
        IFS='|' read -r out flags srcdir srcs <<< "$t"
        g++ -std=c++17 $flags -fPIC $WARN -shared -I"$srcdir" $INC $srcs -o "$F/$out$EXT" -lgomp -lm
        echo "Built $F/$out$EXT"
    done
else
    case "$(uname -s)" in
    MINGW*|MSYS*)
        GXX="${GXX:-g++}"
        PYRUN="$(cygpath -u "${PYTHON:?msys2 needs PYTHON=<Windows python.exe>}")"
        w() { cygpath -m "$1"; }
        ;;
    *)  # WSL
        GXX="${GXX:-/mnt/c/msys64/ucrt64/bin/g++.exe}"
        PYRUN="${PYTHON:-/mnt/d/03learn/machine/.venv/Scripts/python.exe}"
        w() { wslpath -w "$1"; }
        ;;
    esac
    q() { "$PYRUN" -c "$1" | tr -d '\r' | tr '\\' '/'; }
    EXT="$(q 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    INC="-I$(q 'import sysconfig;print(sysconfig.get_paths()["include"])') -I$(q 'import numpy;print(numpy.get_include())') -I$(w "$F/include")"
    LIB="-L$(q 'import sys;print(sys.base_prefix)')/libs -l$(q 'import sysconfig;print("python"+sysconfig.get_config_var("py_version_nodot"))')"
    for t in "${TARGETS[@]}"; do
        IFS='|' read -r out flags srcdir srcs <<< "$t"
        files=""; for s in $srcs; do files="$files $(w "$s")"; done
        "$GXX" -std=c++17 $flags -static -static-libgcc -static-libstdc++ $WARN -shared -I"$(w "$srcdir")" $INC $files \
            $LIB -o "$(w "$F/$out$EXT")" -lm
        echo "Built $F/$out$EXT"
    done
fi
