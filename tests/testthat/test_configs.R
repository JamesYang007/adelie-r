# ==================================================================
# TEST configs
# ==================================================================

test_that("check_global_change", {
    configs <- new(Configs)
    curr <- configs$hessian_min
    set_configs("hessian_min", 23)
    expect_equal(configs$hessian_min, 23)

    configs <- new(Configs)
    expect_equal(configs$hessian_min, 23)
})

test_that("check_default_change", {
    configs <- new(Configs)
    curr <- configs$hessian_min_def
    set_configs("hessian_min", 23)
    expect_equal(configs$hessian_min, 23)
    set_configs("hessian_min")
    expect_equal(configs$hessian_min, curr)
})