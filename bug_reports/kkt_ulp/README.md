# `kkt`/`screen` 1-ULP infinite-loop reprex

Files supporting the bug fixed by [JamesYang007/adelie#159](https://github.com/JamesYang007/adelie/pull/159).
On unfixed `adelie_core` (CRAN `adelie` 1.0.8 / 1.0.9) the calls below hang
at 100% CPU because `kkt()` and the `screen` escape valve in
`adelie_core/solver/solver_base.hpp` use bit-different multiplication
orderings to compute the same threshold; `abs_grad[k=4]` lands in their
1-ULP gap and the BASIL `while(1)` cannot exit.

## R reprex

```r
library(adelie)
load("reprex_data.rda")  # X_train (84x100), y_train, group_starts, foldid
cv.grpnet(X_train, glm.gaussian(y_train),
          min_ratio = 0.05, foldid = foldid,
          groups = group_starts, alpha = 0.7)
```

Run via `Rscript reprex.R`.  Hangs on unfixed adelie; completes on fixed.

## Pure-C++ reprex

No R, Python, or wrapper involvement at runtime: links only `adelie_core`
header templates and reads bit-exact state inputs (dumped from R through
the library's own `X$mul` / `X$var` functions, so every double matches
what `grpnet()` constructs internally).

```bash
Rscript dump_state.R       # writes cpp_data/*.bin
./build_cpp_reprex.sh      # uses R's exact compile flags
ADELIE_TRACE=1 ADELIE_MAX_OUTER=3 ./cpp_reprex
```

On unfixed source the env-gated trace prints the same group `k=4`,
`abs_grad`, `cut_kkt`, `cut_scr` triple every iteration until the
diagnostic cap fires; on fixed source the harness completes in
milliseconds with no other change to the inputs or binary.

## Files

| file | purpose |
|---|---|
| `reprex.R`            | R reprex (`cv.grpnet`)                                            |
| `reprex_data.rda`     | bundled inputs (`X_train`, `y_train`, `group_starts`, `foldid`)   |
| `dump_state.R`        | dumps StateGaussianNaive inputs to `cpp_data/*.bin`               |
| `cpp_reprex.cpp`      | standalone C++ harness using `adelie_core` templates              |
| `build_cpp_reprex.sh` | compile script (uses R's exact flags from `R CMD config`)         |
