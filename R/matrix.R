#' @export
matrix.block_diag <- function(
    mats,
    n_threads =1
)
{
    mats_wrap <- list()
    for (i in 1:length(mats)) {
        mat <- mats[[i]]
        if (is.matrix(mat) || is.array((mat)) || is.data.frame((mat))) {
            mat <- matrix.dense(mat, method="cov", n_threads=n_threads)
        }
        mats_wrap[[i]] <- mat
    }
    mats <- mats_wrap
    out <- make_r_matrix_cov_block_diag_64(mats, n_threads)
    attr(out, "_mats") <- mats
    out
}

#' @export
matrix.concatenate <- function(
    mats, 
    axis =0,
    n_threads =1
)
{
    mats_wrap <- list()
    for (i in 1:length(mats)) {
        mat <- mats[[i]]
        if (is.matrix(mat) || is.array((mat)) || is.data.frame((mat))) {
            mat <- matrix.dense(mat, method="naive", n_threads=n_threads)
        }
        mats_wrap[[i]] <- mat
    }
    mats <- mats_wrap
    dispatcher <- c(
        make_r_matrix_naive_rconcatenate_64,
        make_r_matrix_naive_cconcatenate_64
    )
    out <- dispatcher[[axis+1]](mats)
    attr(out, "_mats") <- mats
    out
}

#' @export
matrix.dense <- function(
    mat,
    method ="naive",
    n_threads =1
)
{
    mat <- as.matrix(mat)
    dispatcher <- c(
        "naive" = make_r_matrix_naive_dense_64F,
        "cov" = make_r_matrix_cov_dense_64F
    )
    out <- dispatcher[[method]](mat, n_threads)
    attr(out, "_mat") <- mat
    out
}

#' @export
matrix.interaction <- function(
    mat,
    intr_keys,
    intr_values,
    levels =NULL,
    n_threads =1
)
{   
    mat <- as.matrix(mat)
    d <- ncol(mat)

    if (is.null(levels)) {
        levels <- integer(d)
    }

    stopifnot(length(intr_keys) == length(intr_values))
    stopifnot(length(intr_keys) > 0)

    arange_d <- as.integer((1:d) - 1)
    keys <- sort(unique(as.integer(intr_keys)))
    pairs_seen <- hashset()
    pairs <- c()
    for (i in 1:length(keys)) {
        key <- keys[i]
        if (key < 0 || key >= d) {
            warning("key not in range [0, d).")
        }
        value_lst <- intr_values[[i]]
        if (is.null(value_lst)) {
            value_lst <- arange_d
        } else {
            value_lst <- sort(unique(as.integer(value_lst)))
        }

        for (val in value_lst) {
            if (
                query(pairs_seen, c(key, val)) ||
                query(pairs_seen, c(val, key)) ||
                (key == val)
            ) {
                next
            }
            if (val < 0 || val >= d) {
                warning("value not in range [0, d).")
            }
            pairs <- c(pairs, key, val)
            insert(pairs_seen, c(key, val))
        }
    }
    stopifnot(length(pairs) > 0)

    pairsT <- matrix(pairs, nrow=2)
    mode(pairsT) <- "integer"
    levels <- as.integer(levels)

    out <- make_r_matrix_naive_interaction_dense_64F(mat, pairsT, levels, n_threads)
    attr(out, "_mat") <- mat
    attr(out, "_pairs") <- t(pairsT)
    attr(out, "_levels") <- levels
    out
}

#' @export
matrix.kronecker_eye <- function(
    mat,
    K,
    n_threads =1
)
{
    if (is.matrix(mat) || is.array((mat)) || is.data.frame((mat))) {
        mat <- as.matrix(mat)
        out <- make_r_matrix_naive_kronecker_eye_dense_64F(mat, K, n_threads)
    } else {
        out <- make_r_matrix_naive_kronecker_eye_64(mat, K, n_threads)
    }
    attr(out, "_mat") <- mat
    out
}

#' @export
matrix.one_hot <- function(
    mat,
    levels =NULL,
    n_threads =1
)
{
    d <- ncol(mat)
    if (is.null(levels)) {
        levels <- integer(d)
    }
    levels <- as.integer(levels)
    out <- make_r_matrix_naive_one_hot_dense_64F(mat, levels, n_threads)
    attr(out, "_mat") <- mat
    attr(out, "_levels") <- levels
    out
}

#' @export
matrix.lazy_cov <- function(
    mat,
    n_threads =1
)
{
    mat <- as.matrix(mat)
    out <- make_r_matrix_cov_lazy_cov_64F(mat, n_threads)
    attr(out, "_mat") <- mat
    out
}

#' @export
matrix.snp_phased_ancestry <- function(
    io,
    n_threads =1
)
{
    if (!io$is_read) { io$read() }
    out <- make_r_matrix_naive_snp_phased_ancestry_64(io, n_threads)
    attr(out, "_io") <- io
    out
}

#' @export
matrix.snp_unphased <- function(
    io,
    n_threads =1
)
{
    if (!io$is_read) { io$read() }
    out <- make_r_matrix_naive_snp_unphased_64(io, n_threads)
    attr(out, "_io") <- io
    out
}

#' @export
matrix.sparse <- function(
    mat,
    method ="naive",
    n_threads =1
)
{  
    mat <- as(mat, "dgCMatrix")
    dispatcher <- c(
        "naive" = make_r_matrix_naive_sparse_64F,
        "cov" = make_r_matrix_cov_sparse_64F
    )
    out <- dispatcher[[method]](
        nrow(mat),
        ncol(mat), 
        length(mat@i),
        mat@p,
        mat@i,
        mat@x,
        n_threads
    )
    attr(out, "mat") <- mat
    out
}

#' @export
matrix.standardize <- function(
    mat,
    centers =NULL,
    scales =NULL,
    ddof =0,
    n_threads =1
)
{
    n <- mat$rows
    p <- mat$cols
    sqrt_weights <- as.double(rep(1 / sqrt(n), n))
    is_centers_none <- is.null(centers)

    if (is_centers_none) {
        centers <- double(p)
        mat$mul(sqrt_weights, sqrt_weights, centers)
    }

    if (is.null(scales)) {
        if (is_centers_none) {
            means <- centers
        } else {
            means <- double(p)
            mat$mul(sqrt_weights, sqrt_weights, means)
        }

        vars <- double(p)
        buffer <- matrix(0, n, 1)
        for (j in 1:p) {
            var_j <- 0
            mat$cov(j-1, 1, sqrt_weights, var_j, buffer)
            vars[j] <- var_j
        }
        vars <- vars + centers * (centers - 2 * means)
        scales <- sqrt((n / (n - ddof)) * vars)
    }

    centers <- as.double(centers)
    scales <- as.double(scales)
    out <- make_r_matrix_naive_standardize_64(mat, centers, scales, n_threads)
    attr(out, "_mat") <- mat
    attr(out, "_centers") <- centers
    attr(out, "_scales") <- scales
    out
}

#' @export
matrix.subset <- function(
    mat,
    indices,
    axis =0,
    n_threads =1
)
{
    dispatcher <- c(
        make_r_matrix_naive_rsubset_64,
        make_r_matrix_naive_csubset_64
    )
    indices <- as.integer(indices)
    out <- dispatcher[[axis+1]](mat, indices, n_threads)
    attr(out, "_mat") <- mat
    attr(out, "_indices") <- indices
    out
}