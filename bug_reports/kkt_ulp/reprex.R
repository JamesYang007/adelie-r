## Minimal reproducer for the adelie infinite-loop bug in adelie_core <= 1.0.9.
## Tested on macOS arm64 / Apple clang.  Hangs at 100% CPU; no progress.
##
## Root cause (see PR text): in
##   inst/adelie/adelie/src/include/adelie_core/solver/solver_base.hpp
## the screen escape valve at line 369 and kkt() at line 429 compute the same
## threshold with different floating-point multiplication orderings,
##     line 369: lmda_next * penalty[i] * alpha
##     line 429: lmda      * alpha      * pk
## which clang produces as 1-ULP-different doubles for some inputs.  When
## abs_grad lands exactly between them, kkt() reports "violator" forever and
## screen() refuses to add the group, infinite-looping the BASIL while(1) at
## solver_base.hpp:611.

library(adelie)
load("reprex_data.rda")
cv.grpnet(X_train, glm.gaussian(y_train),
          min_ratio = 0.05, foldid = foldid,
          groups = group_starts, alpha = 0.7)
