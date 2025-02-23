# ==================================================================
# TEST constraint
# ==================================================================

test_that("constraint.box", {
    lower <- double(10)
    upper <- 5 + lower
    expect_error(constraint.box(lower, upper), NA)
})
