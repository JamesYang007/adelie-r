render_constraints_ <- function(
    n_groups,
    constraints
)
{
    if (is.null(constraints)) {
        constraints <- replicate(n_groups, NULL, FALSE)
    }
    constraints
}

render_dual_groups <- function(
    constraints
)
{
    G <- length(constraints)
    counts <- sapply(1:G, function(i) {
        c <- constraints[[i]]
        if (is.null(c)) return(0)
        c$dual_sizes
    })
    as.integer(cumsum(c(0, counts))[1:G])
}

render_gaussian_inputs_ <- function(
    groups,
    lmda_max,
    lmda_path,
    lmda_path_size,
    max_screen_size,
    max_active_size
)
{
    if (is.null(max_screen_size)) {
        max_screen_size <- length(groups)
    }
    if (is.null(max_active_size)) {
        max_active_size <- length(groups)
    }
    max_screen_size <- min(max_screen_size, length(groups))
    max_active_size <- min(max_active_size, length(groups))

    lmda_path_size <- as.integer(ifelse(is.null(lmda_path), lmda_path_size, length(lmda_path)))

    setup_lmda_max <- is.null(lmda_max)
    setup_lmda_path <- is.null(lmda_path)

    if (setup_lmda_max) lmda_max <- -1.0
    if (setup_lmda_path) lmda_path <- double(0)

    list(
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        lmda_path_size=lmda_path_size,
        setup_lmda_max=setup_lmda_max,
        setup_lmda_path=setup_lmda_path,
        lmda_max=lmda_max,
        lmda_path=lmda_path
    )
}

render_gaussian_cov_inputs_ <- function(A, ...)
{
    render_gaussian_inputs_(...)
}

render_gaussian_naive_inputs_ <- function(
    X, ...
)
{
    render_gaussian_inputs_(...)
}

render_multi_inputs_ <- function(
    X,
    offsets,
    intercept,
    n_threads
)
{
    n <- nrow(offsets)
    n_classes <- ncol(offsets)
    offsets <- matrix(as.double(offsets), n, n_classes)
    X <- matrix.kronecker_eye(X, n_classes, n_threads)
    if (intercept) {
        ones_kron <- matrix.kronecker_eye(matrix(rep_len(1.0, n), n, 1), n_classes, n_threads)
        X <- matrix.concatenate(
            list(ones_kron, X),
            axis=2,
            n_threads
        )
    }

    list(
        X=X,
        offsets=offsets
    )
}

state.create_from_core <- function(
    state,
    core_state
)
{
    attrs <- attributes(state)

    for (i in 1:length(attrs)) {
        key <- attributes(attrs[i])[[1]]
        if (startsWith(key[1], "_")) {
            attr(core_state, key) <- attrs[[i]]
        }
    }

    core_state
}

state.gaussian_cov <- function(
    A,
    v,
    constraints,
    groups,
    group_sizes,
    alpha,
    penalty,
    screen_set,
    screen_beta,
    screen_is_active,
    active_set_size,
    active_set,
    rsq,
    lmda,
    grad,
    lmda_path=NULL,
    lmda_max=NULL,
    max_iters=as.integer(1e5),
    tol=1e-7,
    rdev_tol=1e-4,
    newton_tol=1e-12,
    newton_max_iters=1000,
    n_threads=1,
    early_exit=TRUE,
    screen_rule="pivot",
    min_ratio=1e-2,
    lmda_path_size=100,
    max_screen_size=NULL,
    max_active_size=NULL,
    pivot_subset_ratio=0.1,
    pivot_subset_min=1,
    pivot_slack_ratio=1.25
)
{
    inputs <- render_gaussian_cov_inputs_(
        A=A,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size
    )
    max_screen_size <- inputs[["max_screen_size"]]
    max_active_size <- inputs[["max_active_size"]]
    lmda_path_size <- inputs[["lmda_path_size"]]
    setup_lmda_max <- inputs[["setup_lmda_max"]]
    setup_lmda_path <- inputs[["setup_lmda_path"]]
    lmda_max <- inputs[["lmda_max"]]
    lmda_path <- inputs[["lmda_path"]]

    if (is.matrix(A) || is.array(A) || is.data.frame(A)) {
        A <- matrix.dense(A, method="cov", n_threads=n_threads)
    }

    constraints <- render_constraints_(length(groups), constraints)
    dual_groups <- render_dual_groups(constraints)

    input <- list(
        "A"=A,
        "v"=v,
        "constraints"=constraints,
        "groups"=groups,
        "group_sizes"=group_sizes,
        "dual_groups"=dual_groups,
        "alpha"=alpha,
        "penalty"=penalty,
        "lmda_path"=lmda_path,
        "lmda_max"=lmda_max,
        "min_ratio"=min_ratio,
        "lmda_path_size"=lmda_path_size,
        "max_screen_size"=max_screen_size,
        "max_active_size"=max_active_size,
        "pivot_subset_ratio"=pivot_subset_ratio,
        "pivot_subset_min"=pivot_subset_min,
        "pivot_slack_ratio"=pivot_slack_ratio,
        "screen_rule"=screen_rule,
        "max_iters"=max_iters,
        "tol"=tol,
        "rdev_tol"=rdev_tol,
        "newton_tol"=newton_tol,
        "newton_max_iters"=newton_max_iters,
        "early_exit"=early_exit,
        "setup_lmda_max"=setup_lmda_max,
        "setup_lmda_path"=setup_lmda_path,
        "n_threads"=n_threads,
        "screen_set"=screen_set,
        "screen_beta"=screen_beta,
        "screen_is_active"=screen_is_active,
        "active_set_size"=active_set_size,
        "active_set"=active_set,
        "rsq"=rsq,
        "lmda"=lmda,
        "grad"=grad
    )
    out <- new(RStateGaussianCov64, input)
    attr(out, "_A") <- A
    attr(out, "_v") <- v
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_dual_groups") <- dual_groups
    attr(out, "_penalty") <- penalty
    out
}

state.gaussian_naive <- function(
    X,
    y,
    X_means,
    y_mean,
    y_var,
    resid,
    resid_sum,
    constraints,
    groups,
    group_sizes,
    alpha,
    penalty,
    weights,
    offsets,
    screen_set,
    screen_beta,
    screen_is_active,
    active_set_size,
    active_set,
    rsq,
    lmda,
    grad,
    lmda_path=NULL,
    lmda_max=NULL,
    max_iters=as.integer(1e5),
    tol=1e-7,
    adev_tol=0.9,
    ddev_tol=0,
    newton_tol=1e-12,
    newton_max_iters=1000,
    n_threads=1,
    early_exit=TRUE,
    intercept=TRUE,
    screen_rule="pivot",
    min_ratio=1e-2,
    lmda_path_size=100,
    max_screen_size=NULL,
    max_active_size=NULL,
    pivot_subset_ratio=0.1,
    pivot_subset_min=1,
    pivot_slack_ratio=1.25
)
{
    inputs <- render_gaussian_naive_inputs_(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size
    )
    max_screen_size <- inputs[["max_screen_size"]]
    max_active_size <- inputs[["max_active_size"]]
    lmda_path_size <- inputs[["lmda_path_size"]]
    setup_lmda_max <- inputs[["setup_lmda_max"]]
    setup_lmda_path <- inputs[["setup_lmda_path"]]
    lmda_max <- inputs[["lmda_max"]]
    lmda_path <- inputs[["lmda_path"]]

    if (is.matrix(X) || is.array(X) || is.data.frame(X)) {
        X <- matrix.dense(X, method="naive", n_threads=n_threads)
    }

    glm <- glm.gaussian(y=y, weights=weights)
    constraints <- render_constraints_(length(groups), constraints)
    dual_groups <- render_dual_groups(constraints)
    input <- list(
        "X"=X,
        "X_means"=X_means,
        "y_mean"=y_mean,
        "y_var"=y_var,
        "resid"=resid,
        "resid_sum"=resid_sum,
        "constraints"=constraints,
        "groups"=groups,
        "group_sizes"=group_sizes,
        "dual_groups"=dual_groups,
        "alpha"=alpha,
        "penalty"=penalty,
        "weights"=weights,
        "lmda_path"=lmda_path,
        "lmda_max"=lmda_max,
        "min_ratio"=min_ratio,
        "lmda_path_size"=lmda_path_size,
        "max_screen_size"=max_screen_size,
        "max_active_size"=max_active_size,
        "pivot_subset_ratio"=pivot_subset_ratio,
        "pivot_subset_min"=pivot_subset_min,
        "pivot_slack_ratio"=pivot_slack_ratio,
        "screen_rule"=screen_rule,
        "max_iters"=max_iters,
        "tol"=tol,
        "adev_tol"=adev_tol,
        "ddev_tol"=ddev_tol,
        "newton_tol"=newton_tol,
        "newton_max_iters"=newton_max_iters,
        "early_exit"=early_exit,
        "setup_lmda_max"=setup_lmda_max,
        "setup_lmda_path"=setup_lmda_path,
        "intercept"=intercept,
        "n_threads"=n_threads,
        "screen_set"=screen_set,
        "screen_beta"=screen_beta,
        "screen_is_active"=screen_is_active,
        "active_set_size"=active_set_size,
        "active_set"=active_set,
        "rsq"=rsq,
        "lmda"=lmda,
        "grad"=grad
    )
    out <- new(RStateGaussianNaive64, input)
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X
    attr(out, "_X_means") <- X_means
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_dual_groups") <- dual_groups
    attr(out, "_penalty") <- penalty
    attr(out, "_offsets") <- offsets
    out
}

state.multigaussian_naive <- function(
    X,
    y,
    X_means,
    y_var,
    resid,
    resid_sum,
    constraints,
    groups,
    group_sizes,
    alpha,
    penalty,
    weights,
    offsets,
    screen_set,
    screen_beta,
    screen_is_active,
    active_set_size,
    active_set,
    rsq,
    lmda,
    grad,
    lmda_path=NULL,
    lmda_max=NULL,
    max_iters=1e5L,
    tol=1e-7,
    adev_tol=0.9,
    ddev_tol=0,
    newton_tol=1e-12,
    newton_max_iters=1000,
    n_threads=1,
    early_exit=TRUE,
    intercept=TRUE,
    screen_rule="pivot",
    min_ratio=1e-2,
    lmda_path_size=100,
    max_screen_size=NULL,
    max_active_size=NULL,
    pivot_subset_ratio=0.1,
    pivot_subset_min=1,
    pivot_slack_ratio=1.25
)
{
    inputs <- render_gaussian_naive_inputs_(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size
    )
    max_screen_size <- inputs[["max_screen_size"]]
    max_active_size <- inputs[["max_active_size"]]
    lmda_path_size <- inputs[["lmda_path_size"]]
    setup_lmda_max <- inputs[["setup_lmda_max"]]
    setup_lmda_path <- inputs[["setup_lmda_path"]]
    lmda_max <- inputs[["lmda_max"]]
    lmda_path <- inputs[["lmda_path"]]

    X_raw <- X
    n_classes <- ncol(y)
    inputs <- render_multi_inputs_(
        X=X,
        offsets=offsets,
        intercept=intercept,
        n_threads=n_threads
    )
    X <- inputs[["X"]]
    offsets <- inputs[["offsets"]]

    glm <- glm.multigaussian(y=y, weights=weights)
    X_expanded <- X
    weights_expanded <- rep(weights, each=n_classes) / n_classes
    constraints <- render_constraints_(length(groups), constraints)
    dual_groups <- render_dual_groups(constraints)

    input <- list(
        "n_classes"=n_classes,
        "multi_intercept"=intercept,
        "X"=X,
        "X_means"=X_means,
        # y_mean is not used in the solver since global intercept is turned off,
        # but it is used to compute loss_null and loss_full.
        # This is not the actual y_mean, but it is a value that will result in correct
        # calculation of loss_null and loss_full.
        "y_mean"=sqrt(sum((rowSums(weights * (y - offsets)) / n_classes) ** 2)),
        "y_var"=y_var,
        "resid"=resid,
        "resid_sum"=resid_sum,
        "constraints"=constraints,
        "groups"=groups,
        "group_sizes"=group_sizes,
        "dual_groups"=dual_groups,
        "alpha"=alpha,
        "penalty"=penalty,
        "weights"=weights_expanded,
        "lmda_path"=lmda_path,
        "lmda_max"=lmda_max,
        "min_ratio"=min_ratio,
        "lmda_path_size"=lmda_path_size,
        "max_screen_size"=max_screen_size,
        "max_active_size"=max_active_size,
        "pivot_subset_ratio"=pivot_subset_ratio,
        "pivot_subset_min"=pivot_subset_min,
        "pivot_slack_ratio"=pivot_slack_ratio,
        "screen_rule"=screen_rule,
        "max_iters"=max_iters,
        "tol"=tol,
        "adev_tol"=adev_tol,
        "ddev_tol"=ddev_tol,
        "newton_tol"=newton_tol,
        "newton_max_iters"=newton_max_iters,
        "early_exit"=early_exit,
        "setup_lmda_max"=setup_lmda_max,
        "setup_lmda_path"=setup_lmda_path,
        "intercept"=FALSE,
        "n_threads"=n_threads,
        "screen_set"=screen_set,
        "screen_beta"=screen_beta,
        "screen_is_active"=screen_is_active,
        "active_set_size"=active_set_size,
        "active_set"=active_set,
        "rsq"=rsq,
        "lmda"=lmda,
        "grad"=grad
    )
    out <- new(RStateMultiGaussianNaive64, input)
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X_raw
    attr(out, "_X_expanded") <- X_expanded
    attr(out, "_X_means") <- X_means
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_dual_groups") <- dual_groups
    attr(out, "_penalty") <- penalty
    attr(out, "_weights_expanded") <- weights_expanded
    attr(out, "_offsets") <- offsets
    out
}

render_glm_naive_inputs_ <- function(
    loss_null, ...
)
{
    out <- render_gaussian_naive_inputs_(...)

    setup_loss_null <- is.null(loss_null)
    if (setup_loss_null) loss_null <- Inf

    out[["setup_loss_null"]] <- setup_loss_null
    out[["loss_null"]] <- loss_null
    out
}

state.glm_naive <- function(
    X,
    glm,
    constraints,
    groups,
    group_sizes,
    alpha,
    penalty,
    offsets,
    screen_set,
    screen_beta,
    screen_is_active,
    active_set_size,
    active_set,
    beta0,
    lmda,
    grad,
    eta,
    resid,
    loss_full,
    loss_null=NULL,
    lmda_path=NULL,
    lmda_max=NULL,
    irls_max_iters=as.integer(1e4),
    irls_tol=1e-7,
    max_iters=as.integer(1e5),
    tol=1e-7,
    adev_tol=0.9,
    ddev_tol=0,
    newton_tol=1e-12,
    newton_max_iters=1000,
    n_threads=1,
    early_exit=TRUE,
    intercept=TRUE,
    screen_rule="pivot",
    min_ratio=1e-2,
    lmda_path_size=100,
    max_screen_size=NULL,
    max_active_size=NULL,
    pivot_subset_ratio=0.1,
    pivot_subset_min=1,
    pivot_slack_ratio=1.25
)
{
    inputs <- render_glm_naive_inputs_(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        loss_null=loss_null
    )
    max_screen_size <- inputs[["max_screen_size"]]
    max_active_size <- inputs[["max_active_size"]]
    lmda_path_size <- inputs[["lmda_path_size"]]
    setup_lmda_max <- inputs[["setup_lmda_max"]]
    setup_lmda_path <- inputs[["setup_lmda_path"]]
    lmda_max <- inputs[["lmda_max"]]
    lmda_path <- inputs[["lmda_path"]]
    setup_loss_null <- inputs[["setup_loss_null"]]
    loss_null <- inputs[["loss_null"]]

    if (is.matrix(X)) {
        X <- matrix.dense(X, method="naive", n_threads=n_threads)
    }
    constraints <- render_constraints_(length(groups), constraints)
    dual_groups <- render_dual_groups(constraints)

    input <- list(
        "X"=X,
        "eta"=eta,
        "resid"=resid,
        "constraints"=constraints,
        "groups"=groups,
        "group_sizes"=group_sizes,
        "dual_groups"=dual_groups,
        "alpha"=alpha,
        "penalty"=penalty,
        "offsets"=offsets,
        "lmda_path"=lmda_path,
        "loss_null"=loss_null,
        "loss_full"=loss_full,
        "lmda_max"=lmda_max,
        "min_ratio"=min_ratio,
        "lmda_path_size"=lmda_path_size,
        "max_screen_size"=max_screen_size,
        "max_active_size"=max_active_size,
        "pivot_subset_ratio"=pivot_subset_ratio,
        "pivot_subset_min"=pivot_subset_min,
        "pivot_slack_ratio"=pivot_slack_ratio,
        "screen_rule"=screen_rule,
        "irls_max_iters"=irls_max_iters,
        "irls_tol"=irls_tol,
        "max_iters"=max_iters,
        "tol"=tol,
        "adev_tol"=adev_tol,
        "ddev_tol"=ddev_tol,
        "newton_tol"=newton_tol,
        "newton_max_iters"=newton_max_iters,
        "early_exit"=early_exit,
        "setup_loss_null"=setup_loss_null,
        "setup_lmda_max"=setup_lmda_max,
        "setup_lmda_path"=setup_lmda_path,
        "intercept"=intercept,
        "n_threads"=n_threads,
        "screen_set"=screen_set,
        "screen_beta"=screen_beta,
        "screen_is_active"=screen_is_active,
        "active_set_size"=active_set_size,
        "active_set"=active_set,
        "beta0"=beta0,
        "lmda"=lmda,
        "grad"=grad
    )
    out <- new(RStateGlmNaive64, input)
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_dual_groups") <- dual_groups
    attr(out, "_penalty") <- penalty
    attr(out, "_offsets") <- offsets
    out
}

state.multiglm_naive <- function(
    X,
    glm,
    constraints,
    groups,
    group_sizes,
    alpha,
    penalty,
    offsets,
    screen_set,
    screen_beta,
    screen_is_active,
    active_set_size,
    active_set,
    lmda,
    grad,
    eta,
    resid,
    loss_full,
    loss_null=NULL,
    lmda_path=NULL,
    lmda_max=NULL,
    irls_max_iters=as.integer(1e4),
    irls_tol=1e-7,
    max_iters=as.integer(1e5),
    tol=1e-7,
    adev_tol=0.9,
    ddev_tol=0,
    newton_tol=1e-12,
    newton_max_iters=1000,
    n_threads=1,
    early_exit=TRUE,
    intercept=TRUE,
    screen_rule="pivot",
    min_ratio=1e-2,
    lmda_path_size=100,
    max_screen_size=NULL,
    max_active_size=NULL,
    pivot_subset_ratio=0.1,
    pivot_subset_min=1,
    pivot_slack_ratio=1.25
)
{
    inputs <- render_glm_naive_inputs_(
        X=X,
        groups=groups,
        lmda_max=lmda_max,
        lmda_path=lmda_path,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        loss_null=loss_null
    )
    max_screen_size <- inputs[["max_screen_size"]]
    max_active_size <- inputs[["max_active_size"]]
    lmda_path_size <- inputs[["lmda_path_size"]]
    setup_lmda_max <- inputs[["setup_lmda_max"]]
    setup_lmda_path <- inputs[["setup_lmda_path"]]
    lmda_max <- inputs[["lmda_max"]]
    lmda_path <- inputs[["lmda_path"]]
    setup_loss_null <- inputs[["setup_loss_null"]]
    loss_null <- inputs[["loss_null"]]

    X_raw <- X
    n_classes <- ncol(glm$y)
    inputs <- render_multi_inputs_(
        X=X,
        offsets=offsets,
        intercept=intercept,
        n_threads=n_threads
    )
    X <- inputs[["X"]]
    offsets <- inputs[["offsets"]]
    constraints <- render_constraints_(length(groups), constraints)
    dual_groups <- render_dual_groups(constraints)

    X_expanded <- X
    input <- list(
        "n_classes"=n_classes,
        "multi_intercept"=intercept,
        "X"=X_expanded,
        "eta"=eta,
        "resid"=resid,
        "constraints"=constraints,
        "groups"=groups,
        "group_sizes"=group_sizes,
        "dual_groups"=dual_groups,
        "alpha"=alpha,
        "penalty"=penalty,
        "offsets"=as.double(t(offsets)),
        "lmda_path"=lmda_path,
        "loss_null"=loss_null,
        "loss_full"=loss_full,
        "lmda_max"=lmda_max,
        "min_ratio"=min_ratio,
        "lmda_path_size"=lmda_path_size,
        "max_screen_size"=max_screen_size,
        "max_active_size"=max_active_size,
        "pivot_subset_ratio"=pivot_subset_ratio,
        "pivot_subset_min"=pivot_subset_min,
        "pivot_slack_ratio"=pivot_slack_ratio,
        "screen_rule"=screen_rule,
        "irls_max_iters"=irls_max_iters,
        "irls_tol"=irls_tol,
        "max_iters"=max_iters,
        "tol"=tol,
        "adev_tol"=adev_tol,
        "ddev_tol"=ddev_tol,
        "newton_tol"=newton_tol,
        "newton_max_iters"=newton_max_iters,
        "early_exit"=early_exit,
        "setup_loss_null"=setup_loss_null,
        "setup_lmda_max"=setup_lmda_max,
        "setup_lmda_path"=setup_lmda_path,
        "intercept"=FALSE,
        "n_threads"=n_threads,
        "screen_set"=screen_set,
        "screen_beta"=screen_beta,
        "screen_is_active"=screen_is_active,
        "active_set_size"=active_set_size,
        "active_set"=active_set,
        "beta0"=0.0,
        "lmda"=lmda,
        "grad"=grad
    )
    out <- new(RStateMultiGlmNaive64, input)
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X_raw
    attr(out, "_X_expanded") <- X_expanded
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_dual_groups") <- dual_groups
    attr(out, "_penalty") <- penalty
    attr(out, "_offsets") <- offsets
    out
}
