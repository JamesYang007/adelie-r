# ==================================================================
# TEST GLM
# ==================================================================

test_that("glm.gaussian", {
    n <- 10
    y <- rnorm(n)
    expect_error(glm.gaussian(y), NA)
})

test_that("glm.multigaussian", {
    n <- 10
    K <- 2
    y <- matrix(rnorm(n * K), n, K)
    expect_error(glm.multigaussian(y), NA)
})

test_that("glm.binomial", {
    n <- 10
    y <- rbinom(n, 1, 0.5)
    expect_error(glm.binomial(y), NA)
})

test_that("glm.cox", {
    n <- 10
    start <- sample.int(20, size=n, replace=TRUE)
    stop <- start + 1 + sample.int(5, size=n, replace=TRUE)
    status <- rbinom(n, 1, 0.5)
    expect_error(glm.cox(start, stop, status), NA)
})

test_that("glm.multinomial", {
    n <- 10
    K <- 2
    y <- t(rmultinom(n, 1, rep_len(1/K, K)))
    expect_error(glm.multinomial(y), NA)
})

test_that("glm.poisson", {
    n <- 10
    y <- rpois(n, 1)
    expect_error(glm.poisson(y), NA)
})