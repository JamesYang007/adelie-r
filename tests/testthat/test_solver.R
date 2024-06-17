# ==================================================================
# TEST solver
# ==================================================================

test_that("solver.gaussian_cov", {
    set.seed(0)
    n <- 100
    p <- 200
    X <- matrix(rnorm(n * p), n, p)
    y <- X[,1] * rnorm(1) + rnorm(n)
    A <- t(X) %*% X / n
    v <- t(X) %*% y / n
    state <- gaussian_cov(A, v)
})

test_that("solver.gaussian_naive", {
    set.seed(0)
    n <- 100
    p <- 20
    X <- matrix(rnorm(n * p), n, p)
    y <- X[, 1] + rnorm(n)
    glmo <- glm.gaussian(y)
    expect_error(state <- grpnet(X, glmo), NA)
})

test_that("solver.multigaussian_naive", {
    set.seed(0)
    n <- 100
    p <- 20
    K <- 3
    X <- matrix(rnorm(n * p), n, p)
    y <- X[, 1] + matrix(rnorm(n * K), n, K)
    glmo <- glm.multigaussian(y)
    expect_error(state <- grpnet(X, glmo), NA)
})

test_that("solver.glm_naive", {
    set.seed(0)
    n <- 100
    p <- 20
    X <- matrix(rnorm(n * p), n, p)
    eta <- X[, 1] + rnorm(n)
    mu <- 1 / (1 + exp(-eta))
    y <- sapply(mu, function(m) { rbinom(1, 1, m) })
    glmo <- glm.binomial(y)
    expect_error(state <- grpnet(X, glmo), NA)
})

test_that("solver.multiglm_naive", {
    set.seed(0)
    n <- 100
    p <- 20
    K <- 3
    X <- matrix(rnorm(n * p), n, p)
    eta <- X[, 1] + matrix(rnorm(n * K), n, K)
    exp_eta <- exp(eta)
    sum_exp_eta <- as.double(rowSums(exp_eta))
    mu <- exp_eta / sum_exp_eta
    y <- t(sapply(1:nrow(mu), function(i) { rmultinom(1, 1, mu[i,]) }))
    glmo <- glm.multinomial(y)
    expect_error(state <- grpnet(X, glmo), NA)
})