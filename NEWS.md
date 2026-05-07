# adelie (development version)

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
