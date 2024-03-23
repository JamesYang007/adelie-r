# ==================================================================
# TEST io
# ==================================================================

test_that("io.snp_unphased", {
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
    handle$read()
    expect_equal(handle$rows(), n)
    expect_equal(handle$snps(), s)
    expect_equal(handle$cols(), s)
    file.remove(filename)
})

test_that("io.snp_phased_ancestry", {
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
    handle$read()
    expect_equal(handle$rows(), n)
    expect_equal(handle$snps(), s)
    expect_equal(handle$cols(), s * A)
    file.remove(filename)
})