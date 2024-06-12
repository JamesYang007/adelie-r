#' Creates a block-diagonal matrix.
#' 
#' @param   mats    List of matrices.
#' @param   n_threads   Number of threads.
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
    input <- list(
        "mats"=mats, 
        "n_threads"=n_threads
    )
    out <- new(RMatrixCovBlockDiag64, input)
    attr(out, "_mats") <- mats
    out
}

#' Creates a concatenation of the matrices.
#' 
#' @param   mats    List of matrices.
#' @param   axis    The axis along which the matrices will be joined.
#' @param   n_threads   Number of threads.
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
        RMatrixNaiveRConcatenate64,
        RMatrixNaiveCConcatenate64
    )
    input <- list(
        "mats"=mats
    )
    out <- new(dispatcher[[axis+1]], input)
    attr(out, "_mats") <- mats
    out
}

#' Creates a viewer of a dense matrix.
#' 
#' @param   mat     The dense matrix.
#' @param   method  Method type.
#' @param   n_threads   Number of threads.
#' @export
matrix.dense <- function(
    mat,
    method ="naive",
    n_threads =1
)
{
    mat <- as.matrix(mat)
    dispatcher <- c(
        "naive" = RMatrixNaiveDense64F,
        "cov" = RMatrixCovDense64F
    )
    input <- list(
        "mat"=mat, 
        "n_threads"=n_threads
    )
    out <- new(dispatcher[[method]], input)
    attr(out, "_mat") <- mat
    out
}

#' Creates a matrix with pairwise interactions.
#' 
#' @param   mat     The dense matrix.
#' @param   intr_keys   List of feature indices.
#' @param   intr_values List of list of feature indices.
#' @param   n_threads   Number of threads.
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

    input <- list(
        "mat"=mat, 
        "pairsT"=pairsT, 
        "levels"=levels, 
        "n_threads"=n_threads
    )
    out <- new(RMatrixNaiveInteractionDense64F, input)
    attr(out, "_mat") <- mat
    attr(out, "_pairs") <- t(pairsT)
    attr(out, "_levels") <- levels
    out
}

#' Creates a Kronecker product with identity matrix.
#'
#' @param   mat     The matrix to view as a Kronecker product.
#' @param   K       Dimension of the identity matrix.
#' @param   n_threads   Number of threads.
#' @export
matrix.kronecker_eye <- function(
    mat,
    K,
    n_threads =1
)
{
    if (is.matrix(mat) || is.array((mat)) || is.data.frame((mat))) {
        mat <- as.matrix(mat)
        input <- list(
            "mat"=mat,
            "K"=K,
            "n_threads"=n_threads
        )
        out <- new(RMatrixNaiveKroneckerEyeDense64F, input)
    } else {
        input <- list(
            "mat"=mat,
            "K"=K,
            "n_threads"=n_threads
        )
        out <- new(RMatrixNaiveKroneckerEye64, input)
    }
    attr(out, "_mat") <- mat
    out
}

#' Creates a lazy covariance matrix.
#' 
#' @param   mat     The data matrix.
#' @param   n_threads   Number of threads.
#' @export
matrix.lazy_cov <- function(
    mat,
    n_threads =1
)
{
    mat <- as.matrix(mat)
    input <- list(
        "mat"=mat, 
        "n_threads"=n_threads
    )
    out <- new(RMatrixCovLazyCov64F, input)
    attr(out, "_mat") <- mat
    out
}

#' Creates a one-hot encoded matrix.
#' 
#' @param   mat     The dense matrix.
#' @param   levels      Number of levels.
#' @param   n_threads   Number of threads.
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
    input <- list(
        "mat"=mat, 
        "levels"=levels, 
        "n_threads"=n_threads
    )
    out <- new(RMatrixNaiveOneHotDense64F, input)
    attr(out, "_mat") <- mat
    attr(out, "_levels") <- levels
    out
}

#' Creates a SNP phased, ancestry matrix.
#' 
#' @param   io      IO handler.
#' @param   n_threads   Number of threads.
#' @export
matrix.snp_phased_ancestry <- function(
    io,
    n_threads =1
)
{
    if (!io$is_read) { io$read() }
    input <- list(
        "io"=io, 
        "n_threads"=n_threads
    )
    out <- new(RMatrixNaiveSNPPhasedAncestry64, input)
    attr(out, "_io") <- io
    out
}

#' Creates a SNP unphased matrix.
#' 
#' @param   io      IO handler.
#' @param   n_threads   Number of threads.
#' @export
matrix.snp_unphased <- function(
    io,
    n_threads =1
)
{
    if (!io$is_read) { io$read() }
    input <- list(
        "io"=io, 
        "n_threads"=n_threads
    )
    out <- new(RMatrixNaiveSNPUnphased64, input)
    attr(out, "_io") <- io
    out
}

#' Creates a viewer of a sparse matrix.
#' 
#' @param   mat     The sparse matrix to view.
#' @param   method  Method type.
#' @param   n_threads   Number of threads.
#' @export
matrix.sparse <- function(
    mat,
    method ="naive",
    n_threads =1
)
{  
    mat <- as(mat, "dgCMatrix")
    dispatcher <- c(
        "naive" = RMatrixNaiveSparse64F,
        "cov" = RMatrixCovSparse64F
    )
    input <- list(
        "rows"=nrow(mat),
        "cols"=ncol(mat), 
        "nnz"=length(mat@i),
        "outer"=mat@p,
        "inner"=mat@i,
        "value"=mat@x,
        "n_threads"=n_threads
    )
    out <- new(dispatcher[[method]], input)
    attr(out, "mat") <- mat
    out
}

#' Creates a standardized matrix.
#' 
#' @param   mat     The underlying matrix.
#' @param   centers     The center values.
#' @param   scales     The scale values.
#' @param   ddof        Degrees of freedom.
#' @param   n_threads   Number of threads.
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
        centers <- mat$mul(sqrt_weights, sqrt_weights)
    }

    if (is.null(scales)) {
        if (is_centers_none) {
            means <- centers
        } else {
            means <- mat$mul(sqrt_weights, sqrt_weights)
        }

        vars <- sapply(1:p, function(j) mat$cov(j-1, 1, sqrt_weights))
        vars <- vars + centers * (centers - 2 * means)
        scales <- sqrt((n / (n - ddof)) * vars)
    }

    centers <- as.numeric(centers)
    scales <- as.numeric(scales)
    input <- list(
        "mat"=mat, 
        "centers"=centers, 
        "scales"=scales, 
        "n_threads"=n_threads
    )
    out <- new(RMatrixNaiveStandardize64, input)
    attr(out, "_mat") <- mat
    attr(out, "_centers") <- centers
    attr(out, "_scales") <- scales
    out
}

#' Creates a subset of the matrix along an axis.
#' 
#' @param   mat     The matrix to subset.
#' @param   indices     Array of indices to subset the matrix.
#' @param   axis        The axis along which to subset.
#' @param   n_threads   Number of threads.
#' @export
matrix.subset <- function(
    mat,
    indices,
    axis =0,
    n_threads =1
)
{
    dispatcher <- c(
        RMatrixNaiveRSubset64,
        RMatrixNaiveCSubset64
    )
    indices <- as.integer(indices)
    input <- list(
        "mat"=mat, 
        "subset"=indices, 
        "n_threads"=n_threads
    )
    out <- new(dispatcher[[axis+1]], input)
    attr(out, "_mat") <- mat
    attr(out, "_indices") <- indices
    out
}