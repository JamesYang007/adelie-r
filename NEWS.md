# adelie (development version)

# adelie 1.0.10

* Fixed an infinite loop in `grpnet()` and `cv.grpnet()` triggered by
  inputs where the BASIL `kkt` and `screen` checks landed
  `abs_grad[k]` in a 1-ULP gap caused by inconsistent multiplication
  ordering in the bundled `adelie_core` solver. Fix lives in the
  `adelie_core` submodule (see
  <https://github.com/JamesYang007/adelie/pull/159>); this release
  bumps the submodule pointer to that fix.

# adelie 1.0.1

* Added fixes to UBSAN issues in `IOSNPUnphased` and `IOSNPPhasedAncestry` classes.
* Added fixes to UBSAN issues in exporting `RStateMultiGlm64`.

# adelie 1.0.0

* Added a `NEWS.md` file to track changes to the package.
