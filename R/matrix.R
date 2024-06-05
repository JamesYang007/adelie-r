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
    if (axis == 0) {
        out <- make_matrix_naive_rconcatenate_64(mats, n_threads)
    } else {
        out <- make_matrix_naive_cconcatenate_64(mats, n_threads)
    }
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
        "naive" = make_matrix_naive_dense_64F,
        "cov" = make_matrix_cov_dense_64F
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
    centers =NULL,
    scales =NULL,
    n_threads =1
)
{   
    mat <- as.matrix(mat)
    d <- ncol(mat)

    if (is.null(levels)) {
        levels <- integer(d)
    }
    if (is.null(centers)) {
        centers <- double(0)
    }
    if (is.null(scales)) {
        scales <- double(0)
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
    centers <- as.double(centers)
    scales <- as.double(scales)

    out <- make_matrix_naive_interaction_dense_64F(mat, pairsT, levels, centers, scales, n_threads)
    attr(out, "_mat") <- mat
    attr(out, "_pairs") <- t(pairsT)
    attr(out, "_levels") <- levels
    attr(out, "_centers") <- centers
    attr(out, "_scales") <- scales
    out
}

#' @export
matrix.kronecker_eye <- function(
    mat,
    K,
    n_threads =1
)
{
    if (is.matrix(mat)) {
        out <- make_matrix_naive_kronecker_eye_dense_64F(mat, K, n_threads)
    } else {
        out <- make_matrix_naive_kronecker_eye_64(mat, K, n_threads)
    }
    attr(out, "_mat") <- mat
    out
}

#' @export
matrix.snp_unphased <- function(
    filename,
    read_mode ="file",
    n_threads =1
)
{
    make_matrix_naive_snp_unphased_64(filename, read_mode, n_threads)
}

#' @export
matrix.snp_phased_ancestry <- function(
    filename,
    read_mode ="file",
    n_threads =1
)
{
    make_matrix_naive_snp_phased_ancestry_64(filename, read_mode, n_threads)
}

#' @export
matrix.lazy_cov <- function(
    mat,
    n_threads =1
)
{
    mat <- as.matrix(mat)
    out <- make_matrix_cov_lazy_cov_64F(mat, n_threads)
    attr(out, "_mat") <- mat
    out
}