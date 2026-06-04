# Confirm matrix_naive_dense::sp_tmul honors n_threads on the PATCHED build.
# Build the dense matrix with a chosen n_threads (as predict() does), then call
# sp_tmul on a sparse v large enough that row-parallelism is real work.
#   n_threads=1 -> serial branch (my guard): ratio ~1, elapsed = baseline
#   n_threads=8 -> else branch, omp_parallel_for over rows with 8 threads:
#                  elapsed should DROP and user/elapsed ratio should climb to ~8
suppressPackageStartupMessages({library(adelie); library(Matrix)})
set.seed(0)
n <- 2000; p <- 2000; nlam <- 200
X <- matrix(rnorm(n * p), n, p)

## sparse v: nlam x p, ~30% nonzero (rows = lambdas, like an interpolated path)
B <- matrix(rnorm(nlam * p) * (matrix(runif(nlam * p), nlam, p) < 0.30), nlam, p)
betas <- as(as(B, "CsparseMatrix"), "RsparseMatrix")

cat("adelie", as.character(packageVersion("adelie")),
    "| detectCores=", parallel::detectCores(),
    "| X =", n, "x", p, "| v =", nlam, "x", p, "(sparse)\n")

reps <- 20
for (T in c(1, 8, 16)) {
  md <- matrix.dense(X, method = "naive", n_threads = T)   # object carries _n_threads = T
  invisible(md$sp_tmul(betas))                              # warm up
  tt <- system.time(for (i in seq_len(reps)) invisible(md$sp_tmul(betas)))
  u <- tt[["user.self"]]; el <- tt[["elapsed"]]
  cat(sprintf("n_threads=%2d  sp_tmul x%d  user=%7.3f  elapsed=%7.3f  ratio=%5.2f  speedup_vs_T1=%s\n",
              T, reps, u, el, u/el,
              if (T == 1) "1.00x(ref)" else sprintf("%.2fx", get0("el1", ifnotfound=el)/el)))
  if (T == 1) el1 <- el
}
