# ==================================================================
# TEST matrix
# ==================================================================

test_that("matrix.block_diag", {
    n <- 100
    ps <- c(10, 20, 30)
    mats <- lapply(ps, function(p) {
        X <- matrix(rnorm(n * p), n, p)
        matrix.dense(t(X) %*% X, method="cov")
    })
    expect_error(matrix.block_diag(mats, method="cov"), NA)
    mats <- lapply(ps, function(p) {
        matrix(rnorm(n * p), n, p)
    })
    expect_error(matrix.block_diag(mats, method="naive"), NA)
})

test_that("matrix.concatenate", {
    n <- 100
    ps <- c(10, 20, 30)
    mats <- lapply(ps, function(p) { 
        matrix.dense(matrix(rnorm(n * p), n, p))
    })
    expect_error(matrix.concatenate(mats, axis=2), NA)

    ns <- c(10, 20, 30)
    p <- 100
    mats <- lapply(ns, function(n) { 
        matrix.dense(matrix(rnorm(n * p), n, p))
    })
    expect_error(matrix.concatenate(mats, axis=1), NA)
})

test_that("matrix.convex_relu", {
    n <- 100
    p <- 20
    m <- 10
    Z <- matrix(rnorm(n * p), n, p)
    mask <- matrix(rbinom(n * m, 1, 0.5), n, m)
    expect_error(matrix.convex_relu(Z, mask), NA)
    Z_sp <- as(Z, "dgCMatrix")
    expect_error(matrix.convex_relu(Z_sp, mask), NA)
})

test_that("matrix.dense", {
    n <- 100
    p <- 20
    X_dense <- matrix(rnorm(n * p), n, p)
    expect_error(matrix.dense(X_dense, method="naive"), NA)
    A_dense <- t(X_dense) %*% X_dense
    expect_error(matrix.dense(A_dense, method="cov"), NA)
    expect_error(matrix.dense(X_dense, method="constraint"), NA)
})

test_that("matrix.interaction", {
    n <- 10
    p <- 20
    X_dense <- matrix(rnorm(n * p), n, p)
    X_dense[,1] <- rbinom(n, 4, 0.5)
    intr_keys <- c(1, 2)
    intr_values <- list(NULL, c(1, 3))
    levels <- c(c(5), rep(1, p-1))
    expect_error(matrix.interaction(X_dense, intr_keys, intr_values, levels), NA)
})

test_that("matrix.kronecker_eye", {
    n <- 100
    p <- 20
    K <- 2
    mat <- matrix(rnorm(n * p), n, p)
    expect_error(matrix.kronecker_eye(mat, K), NA)
    mat <- matrix.dense(mat)
    expect_error(matrix.kronecker_eye(mat, K), NA)
})

test_that("matrix.one_hot", {
    n <- 100
    p <- 20
    mat <- matrix(rnorm(n * p), n, p)
    expect_error(matrix.one_hot(mat), NA)
})

test_that("matrix.lazy_cov", {
    n <- 100
    p <- 20
    mat <- matrix(rnorm(n * p), n, p)
    expect_error(matrix.lazy_cov(mat), NA)
})

test_that("matrix.snp_phased_ancestry", {
    n <- 123
    s <- 423
    A <- 8
    filename <- paste(tempdir(), "snp_phased_ancestry_dummy.snpdat", sep="/")
    handle <- io.snp_phased_ancestry(filename)
    calldata <- matrix(
        as.integer(sample.int(
            2, n * s * 2,
            replace=TRUE,
            prob=c(0.7, 0.3)
        ) - 1),
        n, s * 2
    )
    ancestries <- matrix(
        as.integer(sample.int(
            A, n * s * 2,
            replace=TRUE,
            prob=rep_len(1/A, A)
        ) - 1),
        n, s * 2
    )
    handle$write(calldata, ancestries, A, 1)
    expect_error(matrix.snp_phased_ancestry(handle), NA)
    file.remove(filename)
})

test_that("matrix.snp_unphased", {
    n <- 123
    s <- 423
    filename <- paste(tempdir(), "snp_unphased_dummy.snpdat", sep="/")
    handle <- io.snp_unphased(filename)
    mat <- matrix(
        as.integer(sample.int(
            3, n * s, 
            replace=TRUE, 
            prob=c(0.7, 0.2, 0.1)
        ) - 1),
        n, s
    )
    impute <- double(s)
    handle$write(mat, "mean", impute, 1)
    expect_error(matrix.snp_unphased(handle), NA)
    file.remove(filename)
})

test_that("matrix.sparse", {
    n <- 100
    p <- 20
    X_dense <- matrix(rnorm(n * p), n, p)
    X_sp <- as(X_dense, "dgCMatrix")
    expect_error(matrix.sparse(X_sp, method="naive"), NA)
    A_dense <- t(X_dense) %*% X_dense
    A_sp <- as(A_dense, "dgCMatrix")
    expect_error(matrix.sparse(A_sp, method="cov"), NA)
    expect_error(matrix.sparse(X_sp, method="constraint"), NA)
})

test_that("matrix.standardize", {
    n <- 100
    p <- 20
    X <- matrix(rnorm(n * p), n, p)
    expect_error(matrix.standardize(matrix.dense(X)), NA)
})

test_that("matrix.subset", {
    n <- 100
    p <- 20
    X <- matrix.dense(matrix(rnorm(n * p), n, p))
    indices <- c(1, 3, 10)
    expect_error(matrix.subset(X, indices, axis=1), NA)
    expect_error(matrix.subset(X, indices, axis=2), NA)
})