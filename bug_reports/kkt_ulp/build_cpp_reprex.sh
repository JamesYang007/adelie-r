#!/usr/bin/env bash
# Builds cpp_reprex.cpp with the EXACT compile command R uses for the adelie
# package's .cpp files (extracted from `R CMD INSTALL adelie-r` output), so
# the standalone binary's IEEE-754 roundoff matches R's adelie.so as closely
# as possible.
set -euo pipefail
cd "$(dirname "$0")"

ADELIE_INC="$HOME/GitHub/adelie-r/inst/adelie/adelie/src/include"
EIGEN_INC="$(Rscript -e 'cat(system.file("include", package = "RcppEigen"))')"
RCPP_INC="$(Rscript -e 'cat(system.file("include", package = "Rcpp"))')"
R_INC="$(R RHOME)/include"

CXX="clang++"
FLAGS=(
    -arch arm64
    -std=gnu++17
    -I"$R_INC"
    -DNDEBUG
    -I"$ADELIE_INC"
    -DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
    -I"$RCPP_INC"
    -I"$EIGEN_INC"
    -I/opt/R/arm64/include
    -fPIC
    -falign-functions=64
    -Wall -g -O2
)

CMD=( "$CXX" "${FLAGS[@]}" -o cpp_reprex cpp_reprex.cpp -L/opt/R/arm64/lib )
echo "${CMD[@]}"
"${CMD[@]}"
