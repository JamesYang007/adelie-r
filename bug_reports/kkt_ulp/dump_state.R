## Dumps the StateGaussianNaive constructor inputs for the weights[5]=0
## reprex as raw little-endian binary files in cpp_data/, for bit-exact
## ingestion by cpp_reprex.cpp.
##
## Important: centers/scales/X_means/grad are computed by calling adelie's
## library functions (X$mul / X$mean / X$var) so the doubles go through the
## same MatrixNaiveStandardize::mul code path that grpnet() uses.  Computing
## them with pure-R crossprod produces ULP-different values and the C++
## harness will *not* land in the kkt/screen 1-ULP gap.

library(adelie)
load("reprex_data.rda")

dir.create("cpp_data", showWarnings = FALSE)
W <- function(x, name, type = "double") {
  con <- file(file.path("cpp_data", name), "wb")
  on.exit(close(con))
  if (type == "double") writeBin(as.double(x), con, size = 8L, endian = "little")
  else if (type == "int32") writeBin(as.integer(x), con, size = 4L, endian = "little")
  else stop("?")
}

n <- nrow(X_train); p <- ncol(X_train)
weights <- rep(1, n); weights[5] <- 0; weights <- weights / sum(weights)

## Build matrix tree exactly as solver.R does, then read centers/scales from
## the wrapper attributes and compute X_means/grad through the library.
X_dense   <- matrix.dense(X_train, method = "naive", n_threads = 1)
X         <- matrix.standardize(X_dense, centers = NULL, weights = weights, n_threads = 1)
centers   <- as.numeric(attr(X, "_centers"))
scales    <- as.numeric(attr(X, "_scales"))
ones      <- rep(1.0, n)
X_means   <- as.numeric(X$mul(ones, weights))
y_off     <- y_train
y_mean    <- sum(y_off * weights)
resid     <- y_off - y_mean                     # intercept = TRUE
y_var     <- sum(weights * resid^2)
resid_sum <- sum(weights * resid)
grad      <- as.numeric(X$mul(resid, weights))

groups_0    <- as.integer(group_starts - 1L)
group_sizes <- as.integer(diff(c(group_starts, p + 1L)))
G           <- length(groups_0)
penalty     <- sqrt(as.double(group_sizes))

W(X_train,         "X_train.bin")
W(weights,         "weights.bin")
W(centers,         "centers.bin")
W(scales,          "scales.bin")
W(X_means,         "X_means.bin")
W(y_mean,          "y_mean.bin")
W(y_var,           "y_var.bin")
W(resid,           "resid.bin")
W(resid_sum,       "resid_sum.bin")
W(grad,            "grad.bin")
W(groups_0,        "groups.bin",      "int32")
W(group_sizes,     "group_sizes.bin", "int32")
W(integer(G),      "dual_groups.bin", "int32")  # zeros, length G (no constraints)
W(penalty,         "penalty.bin")
W(integer(0),      "screen_set.bin",  "int32")
W(double(0),       "screen_beta.bin")
W(integer(0),      "screen_is_active.bin", "int32")
W(integer(G),      "active_set.bin",  "int32")
W(c(n, p, G),      "shape.bin",       "int32")
cat("dumped to cpp_data/  (n=", n, " p=", p, " G=", G, ")\n", sep = "")
