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