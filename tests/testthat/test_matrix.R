# ==================================================================
# TEST matrix
# ==================================================================

test_that("matrix.concatenate", {
    n <- 100
    ps <- c(10, 20, 30)
    mats <- lapply(ps, function(p) { 
        matrix.dense(matrix(rnorm(n * p), n, p))
    })
    expect_error(matrix.concatenate(mats), NA)
})

test_that("matrix.dense", {
    n <- 100
    p <- 20
    X_dense <- matrix(rnorm(n * p), n, p)
    expect_error(matrix.dense(X_dense, method="naive"), NA)
    A_dense <- t(X_dense) %*% X_dense
    expect_error(matrix.dense(A_dense, method="cov"), NA)
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

test_that("matrix.snp_unphased", {
    n <- 123
    s <- 423
    filename <- "/tmp/snp_unphased_dummy.snpdat"
    handle <- io.snp_unphased(filename)
    mat <- matrix(
        as.integer(sample.int(
            3, n * s, 
            replace=TRUE, 
            prob=c(0.7, 0.2, 0.1)
        ) - 1),
        n, s
    )
    handle$write(mat, 1)
    expect_error(matrix.snp_unphased("/tmp/snp_unphased_dummy.snpdat"), NA)
    file.remove(filename)
})

test_that("matrix.snp_phased_ancestry", {
    n <- 123
    s <- 423
    A <- 8
    filename <- "/tmp/snp_phased_ancestry_dummy.snpdat"
    handle <- io.snp_phased_ancestry(filename)
    calldata <- matrix(
        as.integer(sample.int(
            3, n * s * 2,
            replace=TRUE,
            prob=c(0.7, 0.2, 0.1)
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
    expect_error(matrix.snp_phased_ancestry("/tmp/snp_phased_ancestry_dummy.snpdat"), NA)
})

test_that("matrix.cov_lazy", {
    n <- 100
    p <- 20
    mat <- matrix(rnorm(n * p), n, p)
    expect_error(matrix.cov_lazy(mat), NA)
})