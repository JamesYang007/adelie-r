# Run the ACTUAL Rd example blocks for grpnet and cv.grpnet (verbatim, including
# the embedded plot() calls, which R CMD check also times), and report the
# user/system/elapsed/ratio exactly as the CRAN "examples" NOTE does.
#
# Conditions are set by the caller via env vars:
#   (unset)              -> OMP at default (all cores)
#   OMP_THREAD_LIMIT=2   -> CRAN's incoming-check cap
#
# Plots go to a throwaway pdf device so plotting cost is included but no window
# is needed (this is what R CMD check does).

suppressPackageStartupMessages(library(adelie))
cat("adelie", as.character(packageVersion("adelie")),
    "| OMP_NUM_THREADS=", Sys.getenv("OMP_NUM_THREADS","<unset>"),
    "| OMP_THREAD_LIMIT=", Sys.getenv("OMP_THREAD_LIMIT","<unset>"),
    "| detectCores=", parallel::detectCores(), "\n", sep=" ")

pdf(tempfile(fileext = ".pdf"))   # absorb plot() output, like R CMD check
on.exit(dev.off())

## ---- grpnet.Rd example (verbatim) ----
grpnet_example <- function() {
  set.seed(0)
  n <- 100
  p <- 200
  X <- matrix(rnorm(n * p), n, p)
  y <- X[,1] * rnorm(1) + rnorm(n)
  groups <- c(1, sample(2:199, 60, replace = FALSE))
  groups <- sort(groups)
  print(groups)
  fit <- grpnet(X, glm.gaussian(y), groups = groups)
  print(fit)
  plot(fit)
  coef(fit)
  cvfit  <- cv.grpnet(X, glm.gaussian(y), groups = groups)
  print(cvfit)
  plot(cvfit)
  predict(cvfit,newx=X[1:5,], lambda="lambda.min")
}

## ---- cv.grpnet.Rd example (verbatim) ----
cvgrpnet_example <- function() {
  set.seed(0)
  n <- 100
  p <- 200
  X <- matrix(rnorm(n * p), n, p)
  y <- X[,1:25] %*% rnorm(25)/4 + rnorm(n)
  groups <- c(1, sample(2:199, 60, replace = FALSE))
  groups <- sort(groups)
  cvfit <- cv.grpnet(X, glm.gaussian(y), groups = groups)
  print(cvfit)
  plot(cvfit)
  predict(cvfit, newx = X[1:5,])
  predict(cvfit, type = "nonzero")
}

time_once <- function(label, fn) {
  invisible(capture.output(fn()))                # warm up, swallow prints
  for (rep in 1:3) {
    tt <- system.time(invisible(capture.output(fn())))
    u <- tt[["user.self"]]; s <- tt[["sys.self"]]; el <- tt[["elapsed"]]
    cat(sprintf("%-12s rep%d  user=%7.3f  sys=%6.3f  elapsed=%7.3f  ratio=%6.2f\n",
                label, rep, u, s, el, if (el > 0) u/el else NA))
  }
}

time_once("grpnet",    grpnet_example)
time_once("cv.grpnet", cvgrpnet_example)
