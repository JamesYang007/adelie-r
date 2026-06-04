# adelie (development version)
# adelie 1.0.10

* Fixed an issue with standardization. Variables with zero variance
  get turned into Inf. What we do instead is change the variance to
  Inf, and those variables become all 0 and are given zero
  coefficients. This problem can exhibit during CV. 

* Fixed the dense-matrix `sp_tmul` so that its serial path honors
  `n_threads`. Its `n_threads = 1` branch ran a raw Eigen product that
  ignored the requested thread count and fell back to Eigen's global
  default (all cores), inflating example CPU time to `user/elapsed`
  ~10x with no wall-clock gain on a many-core machine (the source of
  CRAN's gcc-16 R-devel "CPU time > 2.5 times elapsed" NOTE). The
  `inst/adelie` submodule points at the one-line fix on a forked branch
  (`bnaras/adelie@zero-variance`). Investigation notes and a Docker
  reprex live in `debug/` (excluded from the package tarball).
  
# adelie 1.0.9

* Fixed an infinite loop in `grpnet()` and `cv.grpnet()` triggered by
  inputs where the BASIL `kkt` and `screen` checks disagreed by a
  1-ULP gap due to inconsistent multiplication ordering in the bundled
  `adelie_core` solver. As a temporary workaround, the `inst/adelie`
  submodule points at the fix on a forked branch
  (`bnaras/adelie@fix-kkt-ulp`) until upstream pull request
  <https://github.com/JamesYang007/adelie/pull/159> is merged. A
  reprex and root-cause analysis live in `bug_reports/kkt_ulp/`
  (excluded from the package tarball).

# adelie 1.0.1

* Added fixes to UBSAN issues in `IOSNPUnphased` and `IOSNPPhasedAncestry` classes.
* Added fixes to UBSAN issues in exporting `RStateMultiGlm64`.

# adelie 1.0.0

* Added a `NEWS.md` file to track changes to the package.
