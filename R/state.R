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

render_gaussian_naive_inputs_ <- function(
    X, ...
)
{
    render_gaussian_inputs_(...)
}

render_multi_inputs_ <- function(
    X,
    groups,
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
            axis=1,
            n_threads
        )
    }

    if (length(groups) == X$cols()) {
        group_type <- "ungrouped"
    } else {
        if (length(groups) != X$cols() / n_classes + (n_classes-1) * intercept) {
            stop("groups must be of the \"grouped\" or \"ungrouped\" type.")
        }
        group_type <- "grouped"
    }

    list(
        X=X,
        offsets=offsets,
        group_type=group_type
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

#' @export
state.gaussian_naive <- function(
    X,
    y,
    X_means,
    y_mean,
    y_var,
    resid,
    resid_sum,
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
    ddev_tol=1e-4,
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

    if (is.matrix(X) || is.data.frame(X)) {
        X <- matrix.dense(X, method="naive", n_threads=n_threads)
    }

    glm <- glm.gaussian(y=y, weights=weights)

    out <- make_state_gaussian_naive_64(
        X=X,
        X_means=X_means,
        y_mean=y_mean,
        y_var=y_var,
        resid=resid,
        resid_sum=resid_sum,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        weights=weights,
        lmda_path=lmda_path,
        lmda_max=lmda_max,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        pivot_subset_ratio=pivot_subset_ratio,
        pivot_subset_min=pivot_subset_min,
        pivot_slack_ratio=pivot_slack_ratio,
        screen_rule=screen_rule,
        max_iters=max_iters,
        tol=tol,
        adev_tol=adev_tol,
        ddev_tol=ddev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        early_exit=early_exit,
        setup_lmda_max=setup_lmda_max,
        setup_lmda_path=setup_lmda_path,
        intercept=intercept,
        n_threads=n_threads,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
        rsq=rsq,
        lmda=lmda,
        grad=grad
    )
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X
    attr(out, "_X_means") <- X_means
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_penalty") <- penalty
    attr(out, "_offsets") <- offsets
    attr(out, "_lmda_path") <- lmda_path
    attr(out, "_screen_set") <- screen_set
    attr(out, "_screen_beta") <- screen_beta
    attr(out, "_screen_is_active") <- screen_is_active
    attr(out, "_active_set") <- active_set
    attr(out, "_grad") <- grad
    out
}

#' @export
state.multigaussian_naive <- function(
    X,
    y,
    X_means,
    y_var,
    resid,
    resid_sum,
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
    max_iters=int(1e5),
    tol=1e-7,
    adev_tol=0.9,
    ddev_tol=1e-4,
    newton_tol=1e-12,
    newton_max_iters=1000,
    n_threads=1,
    early_exit=True,
    intercept=True,
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
        groups=groups,
        offsets=offsets,
        intercept=intercept,
        n_threads=n_threads
    )
    X <- inputs[["X"]]
    offsets <- inputs[["offsets"]]
    group_type <- inputs[["group_type"]]

    glm <- glm.multigaussian(y=y, weights=weights)
    X_expanded <- X
    weights_expanded <- rep(weights, each=n_classes) / n_classes

    out <- make_state_multigaussian_naive_64(
        group_type=group_type,
        n_classes=n_classes,
        multi_intercept=intercept,
        X=X,
        X_means=X_means,
        # y_mean is not used in the solver since global intercept is turned off,
        # but it is used to compute loss_null and loss_full.
        # This is not the actual y_mean, but it is a value that will result in correct
        # calculation of loss_null and loss_full.
        y_mean=sqrt(sum((rowSums(weights * (y - offsets)) / n_classes) ** 2)),
        y_var=y_var,
        resid=resid,
        resid_sum=resid_sum,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        weights=weights_expanded,
        lmda_path=lmda_path,
        lmda_max=lmda_max,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        pivot_subset_ratio=pivot_subset_ratio,
        pivot_subset_min=pivot_subset_min,
        pivot_slack_ratio=pivot_slack_ratio,
        screen_rule=screen_rule,
        max_iters=max_iters,
        tol=tol,
        adev_tol=adev_tol,
        ddev_tol=ddev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        early_exit=early_exit,
        setup_lmda_max=setup_lmda_max,
        setup_lmda_path=setup_lmda_path,
        intercept=FALSE,
        n_threads=n_threads,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
        rsq=rsq,
        lmda=lmda,
        grad=grad
    )
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X_raw
    attr(out, "_X_expanded") <- X_expanded
    attr(out, "_X_means") <- X_means
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_penalty") <- penalty
    attr(out, "_weights_expanded") <- weights_expanded
    attr(out, "_offsets") <- offsets
    attr(out, "_lmda_path") <- lmda_path
    attr(out, "_screen_set") <- screen_set
    attr(out, "_screen_beta") <- screen_beta
    attr(out, "_screen_is_active") <- screen_is_active
    attr(out, "_active_set") <- active_set
    attr(out, "_grad") <- grad
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

#' @export
state.glm_naive <- function(
    X,
    glm,
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
    ddev_tol=1e-4,
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

    out <- make_state_glm_naive_64(
        X=X,
        eta=eta,
        resid=resid,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        offsets=offsets,
        lmda_path=lmda_path,
        loss_null=loss_null,
        loss_full=loss_full,
        lmda_max=lmda_max,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        pivot_subset_ratio=pivot_subset_ratio,
        pivot_subset_min=pivot_subset_min,
        pivot_slack_ratio=pivot_slack_ratio,
        screen_rule=screen_rule,
        irls_max_iters=irls_max_iters,
        irls_tol=irls_tol,
        max_iters=max_iters,
        tol=tol,
        adev_tol=adev_tol,
        ddev_tol=ddev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        early_exit=early_exit,
        setup_loss_null=setup_loss_null,
        setup_lmda_max=setup_lmda_max,
        setup_lmda_path=setup_lmda_path,
        intercept=intercept,
        n_threads=n_threads,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
        beta0=beta0,
        lmda=lmda,
        grad=grad
    )
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_penalty") <- penalty
    attr(out, "_offsets") <- offsets
    attr(out, "_lmda_path") <- lmda_path
    attr(out, "_screen_set") <- screen_set
    attr(out, "_screen_beta") <- screen_beta
    attr(out, "_screen_is_active") <- screen_is_active
    attr(out, "_active_set") <- active_set
    attr(out, "_grad") <- grad
    out
}

#' @export
state.multiglm_naive <- function(
    X,
    glm,
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
    ddev_tol=1e-4,
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
        groups=groups,
        offsets=offsets,
        intercept=intercept,
        n_threads=n_threads
    )
    X <- inputs[["X"]]
    offsets <- inputs[["offsets"]]
    group_type <- inputs[["group_type"]]

    X_expanded <- X
    out <- make_state_multiglm_naive_64(
        group_type=group_type,
        n_classes=n_classes,
        multi_intercept=intercept,
        X=X_expanded,
        eta=eta,
        resid=resid,
        groups=groups,
        group_sizes=group_sizes,
        alpha=alpha,
        penalty=penalty,
        offsets=as.double(t(offsets)),
        lmda_path=lmda_path,
        loss_null=loss_null,
        loss_full=loss_full,
        lmda_max=lmda_max,
        min_ratio=min_ratio,
        lmda_path_size=lmda_path_size,
        max_screen_size=max_screen_size,
        max_active_size=max_active_size,
        pivot_subset_ratio=pivot_subset_ratio,
        pivot_subset_min=pivot_subset_min,
        pivot_slack_ratio=pivot_slack_ratio,
        screen_rule=screen_rule,
        irls_max_iters=irls_max_iters,
        irls_tol=irls_tol,
        max_iters=max_iters,
        tol=tol,
        adev_tol=adev_tol,
        ddev_tol=ddev_tol,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        early_exit=early_exit,
        setup_loss_null=setup_loss_null,
        setup_lmda_max=setup_lmda_max,
        setup_lmda_path=setup_lmda_path,
        intercept=FALSE,
        n_threads=n_threads,
        screen_set=screen_set,
        screen_beta=screen_beta,
        screen_is_active=screen_is_active,
        active_set_size=active_set_size,
        active_set=active_set,
        beta0=0.0,
        lmda=lmda,
        grad=grad
    )
    attr(out, "_glm") <- glm
    attr(out, "_X") <- X_raw
    attr(out, "_X_expanded") <- X_expanded
    attr(out, "_groups") <- groups
    attr(out, "_group_sizes") <- group_sizes
    attr(out, "_penalty") <- penalty
    attr(out, "_offsets") <- offsets
    attr(out, "_lmda_path") <- lmda_path
    attr(out, "_screen_set") <- screen_set
    attr(out, "_screen_beta") <- screen_beta
    attr(out, "_screen_is_active") <- screen_is_active
    attr(out, "_active_set") <- active_set
    attr(out, "_grad") <- grad
    out
}