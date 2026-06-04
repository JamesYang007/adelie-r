# Isolate the source of the "CPU time > 2.5 x elapsed" NOTE for grpnet/cv.grpnet.
#
# Hypothesis: with the example default n_threads = 1, adelie's own OpenMP loops
# run single-threaded, but Eigen's built-in OpenMP GEMM parallelization is
# uncapped and defaults to omp_get_max_threads() = all cores. If so, capping
# OpenMP threads should collapse the user/elapsed ratio.
#
# Run from the shell under three conditions:
#   Rscript ...                          # OMP_NUM_THREADS unset  -> expect ratio ~#cores
#   OMP_NUM_THREADS=1   Rscript ...      # expect ratio ~1
#   OMP_THREAD_LIMIT=2  Rscript ...      # CRAN's check cap; expect ratio <~2

suppressPackageStartupMessages(library(adelie))

cat("OMP_NUM_THREADS =", Sys.getenv("OMP_NUM_THREADS", "<unset>"),
    " OMP_THREAD_LIMIT =", Sys.getenv("OMP_THREAD_LIMIT", "<unset>"),
    " detectCores =", parallel::detectCores(), "\n\n")

# Eval the expression unevaluated (avoid R promise-caching), warm up once, then
# time the median of several fresh evaluations so a ~0.3s fit is measured cleanly.
run <- function(label, expr, reps = 5) {
  e  <- substitute(expr)
  pf <- parent.frame()
  eval(e, pf)                                   # warm up
  tt <- system.time(for (i in seq_len(reps)) eval(e, pf))
  u <- tt[["user.self"]] / reps
  s <- tt[["sys.self"]]  / reps
  el <- tt[["elapsed"]]  / reps
  cat(sprintf("%-12s  user=%7.3f  sys=%6.3f  elapsed=%7.3f  ratio=%6.2f\n",
              label, u, s, el, if (el > 0) u / el else NA))
}

## --- grpnet.Rd example data ---
set.seed(0)
n <- 100; p <- 200
X <- matrix(rnorm(n * p), n, p)
y <- X[, 1] * rnorm(1) + rnorm(n)
groups <- sort(c(1, sample(2:199, 60, replace = FALSE)))

run("grpnet",    grpnet(X, glm.gaussian(y), groups = groups))
run("cv.grpnet", cv.grpnet(X, glm.gaussian(y), groups = groups))
