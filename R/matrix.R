#' @export
matrix.concatenate <- function(
    mats, 
    axis =0,
    n_threads =1
)
{
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
    dispatcher <- c(
        "naive" = make_matrix_naive_dense_64F,
        "cov" = make_matrix_cov_dense_64F
    )
    out <- dispatcher[[method]](mat, n_threads)
    attr(out, "_mat") <- mat
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
    out <- make_matrix_cov_lazy_cov_64F(mat, n_threads)
    attr(out, "_mat") <- mat
    out
}