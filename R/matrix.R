#' Creates a block-diagonal matrix.
#'
#' @param   mats    List of matrices.
#' @param   method  Method type, with  default \code{method="naive"}.
#' @param   n_threads   Number of threads.
#' @return Block-diagonal matrix.
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' ps <- c(10, 20, 30)
#' mats <- lapply(ps, function(p) {
#'     X <- matrix(rnorm(n * p), n, p)
#'     matrix.dense(t(X) %*% X, method="cov")
#' })
#' out <- matrix.block_diag(mats, method="cov")
#' mats <- lapply(ps, function(p) {
#'     X <- matrix(rnorm(n * p), n, p)
#'     matrix.dense(X, method="naive")
#' })
#' out <- matrix.block_diag(mats, method="naive")
#' @export
matrix.block_diag <- function(
    mats,
    method =c("naive", "cov"),
    n_threads =1
)
{
    method <- match.arg(method)
    mats_wrap <- list()
    for (i in 1:length(mats)) {
        mat <- mats[[i]]
        if (is.matrix(mat) || is.array((mat)) || is.data.frame((mat))) {
            mat <- matrix.dense(mat, method=method, n_threads=1)
        }
        mats_wrap[[i]] <- mat
    }
    mats <- mats_wrap
    input <- list(
        "mats"=mats,
        "n_threads"=n_threads
    )
    dispatcher <- c(
        "cov"=RMatrixCovBlockDiag64,
        "naive"=RMatrixNaiveBlockDiag64
    )
    out <- new(dispatcher[[method]], input)
    attr(out, "_mats") <- mats
    out
}

#' Creates a concatenation of the matrices.
#'
#' @param   mats    List of matrices.
#' @param   axis    The axis along which the matrices will be joined. With axis = 2 (default) this function is equivalent to \code{cbind()} and axis = 1 is equivalent to \code{rbind()}.
#' @param   n_threads   Number of threads.
#' @return Concatenation of matrices.
#' The object is an S4 class with methods for efficient computation in C++ by adelie. Note that for the object itself axis is represented with base 0 (so 1 less than the argument here).
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' ps <- c(10, 20, 30)
#' n <- 100
#' mats <- lapply(ps, function(p) {
#'     matrix.dense(matrix(rnorm(n * p), n, p))
#' })
#' out <- matrix.concatenate(mats, axis=2)
#' @export
matrix.concatenate <- function(
    mats,
    axis = 2,
    n_threads =1
)
{
    if(axis %in% c(1,2))axis = axis -1 # C++ base 0
    else stop("axis can take values 1 (row-bind) or 2 (column-bind) only")
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

#' Creates a feature matrix for the convex relu problem.
#'
#' @param   mat    Base feature matrix. It is either a dense or sparse matrix.
#' @param   mask   Boolean mask matrix.
#' @param   gated  Flag to indicate whether to use the convex gated relu feature matrix.
#' @param   n_threads   Number of threads.
#' @return Convex relu feature matrix.
#' The object is an S4 class with methods for efficient computation in C++ by adelie.
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' p <- 20
#' m <- 10
#' Z_dense <- matrix(rnorm(n * p), n, p)
#' mask <- matrix(rbinom(n * m, 1, 0.5), n, m)
#' out <- matrix.convex_relu(Z_dense, mask)
#' Z_sparse <- as(Z_dense, "dgCMatrix")
#' out <- matrix.convex_relu(Z_sparse, mask)
#' @export
matrix.convex_relu <- function(
    mat,
    mask,
    gated =FALSE,
    n_threads =1
)
{
    if(inherits(mat,"sparseMatrix")){
        if(!inherits(mat,"dgCMatrix"))
            mat=as(as(as(mat, "generalMatrix"), "CsparseMatrix"), "dMatrix")

        dispatcher <- c(
            RMatrixNaiveConvexReluSparse64F,
            RMatrixNaiveConvexGatedReluSparse64F
        )[[gated+1]]
        input <- list(
            "rows"=nrow(mat),
            "cols"=ncol(mat),
            "nnz"=length(mat@i),
            "outer"=mat@p,
            "inner"=mat@i,
            "value"=mat@x,
            "mask"=mask,
            "n_threads"=n_threads
        )
    } else {
        dispatcher <- c(
            RMatrixNaiveConvexReluDense64F,
            RMatrixNaiveConvexGatedReluDense64F
        )[[gated+1]]
        input <- list(
            "mat"=mat,
            "mask"=mask,
            "n_threads"=n_threads
        )
    }
    out <- new(dispatcher, input)
    attr(out, "_mats") <- mat
    attr(out, "_mask") <- mask
    out
}

#' Creates a dense matrix object.
#'
#' @param   mat     The dense matrix.
#' @param   method  Method type, with  default \code{method="naive"}.
#' If \code{method="cov"}, the matrix is used with the solver \code{gaussian_cov()}.
#' Used for \code{glm.gaussian()} and \code{glm.multigaussian()} families. Generally "naive" is used for wide matrices, and "cov" for tall matrices.
#' If \code{method="constraint"}, the matrix is used as input to the constraint objects.
#' @param   n_threads   Number of threads.
#' @return Dense matrix.
#' The object is an S4 class with methods for efficient computation by adelie.
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' p <- 20
#' X_dense <- matrix(rnorm(n * p), n, p)
#' out <- matrix.dense(X_dense, method="naive")
#' A_dense <- t(X_dense) %*% X_dense
#' out <- matrix.dense(A_dense, method="cov")
#' out <- matrix.dense(X_dense, method="constraint")
#' @export
matrix.dense <- function(
    mat,
    method = c("naive","cov","constraint"),
    n_threads =1
)
{
    method=match.arg(method)
    mat <- as.matrix(mat)
    dispatcher <- c(
        "naive" = RMatrixNaiveDense64F,
        "cov" = RMatrixCovDense64F,
        "constraint" = RMatrixConstraintDense64F
    )
    input <- list(
        "n_threads"=n_threads
    )
    if (method == "constraint") {
        mat <- t(mat)
        input[["matT"]] <- mat
    } else {
        input[["mat"]] <- mat
    }
    out <- new(dispatcher[[method]], input)
    attr(out, "_mat") <- mat
    out
}

#' Creates an eager covariance matrix.
#'
#' @param   mat     A dense matrix to be used with the \code{gaussian_cov()} solver.
#' @param   n_threads   Number of threads.
#' @return The dense covariance matrix. This matrix is exactly \code{t(mat)%*%mat}, computed with some efficiency.
#' @examples
#' n <- 100
#' p <- 20
#' mat <- matrix(rnorm(n * p), n, p)
#' out <- matrix.eager_cov(mat)
#' @export
matrix.eager_cov <- function(
    mat,
    n_threads =1
)
{
    dgemtm(mat, n_threads)
}

#' Creates a matrix with pairwise interactions.
#'
#' @param   mat     The dense matrix, which can include factors with levels coded as non-negative integers.
#' @param   intr_keys   List of feature indices. This is a list of all features with which interactions can be formed. Default is \code{1:p} where \code{p} is the number of columns in \code{mat}.
#' @param   intr_values List of integer vectors of feature indices. For each of the \code{m <= p} indices listed in \code{intr_keys}, there is a vector of indices indicating which columns are candidates for interaction with that feature. If a list is \code{list(NULL)}, that means all other features are candidates for interactions.  The default is a list of length \code{m} where each element is \code{list(NULL)}; that is \code{rep(list(NULL), m}.
#' @param   levels Number of levels for each of the columns of \code{mat}, with \code{1} representing a quantitative feature. A factor with \code{K} levels should be represented by the numbers \code{0,1,...,K-1}.
#' @param   n_threads   Number of threads.
#' @return Pairwise interaction matrix. Logic is used to avoid repetitions. For each factor variable, the column is one-hot-encoded to form a basis for that feature.
#' The object is an S4 class with methods for efficient computation by adelie. Note that some of the arguments are transformed to C++ base 0 for internal use, and if the object is examined, it will reflect that.
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 10
#' p <- 20
#' X_dense <- matrix(rnorm(n * p), n, p)
#' X_dense[,1] <- rbinom(n, 4, 0.5)
#' intr_keys <- c(1, 2)
#' intr_values <- list(NULL, c(1, 3))
#' levels <- c(c(5), rep(1, p-1))
#' out <- matrix.interaction(X_dense, intr_keys, intr_values, levels)
#' @export
matrix.interaction <- function(
    mat,
    intr_keys = NULL,
    intr_values,
    levels =NULL,
    n_threads =1
)
{
    mat <- as.matrix(mat)
    d <- ncol(mat)

    if (is.null(levels)) {# levels of 1 are translated to 0
        levels <- integer(d)
    }
    else levels[levels==1] <- 0
    levels=as.integer(levels)
    if(is.null(intr_keys))intr_keys = 1:d
    intr_keys = intr_keys-1 # base 0 for C++
    if(missing(intr_values))
        intr_values = rep(list(NULL),length(intr_keys))
    else{
        if(length(intr_values) != length(intr_keys))
            stop("the length of intr_values should be the same as length of intr_keys")
        intr_values = lapply(intr_values,# base 0 for C++
                             function(x)if(!is.null(x))x-1 else x)
        }
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

#' Creates a Kronecker product with an identity matrix.
#'
#' @param   mat     The matrix to view as a Kronecker product.
#' @param   K       Dimension of the identity matrix (default is 1, which does essentially nothing).
#' @param   n_threads   Number of threads.
#' @return Kronecker product with identity matrix. If \code{mat} is n x p, the the resulting matrix will be nK x np.
#' The object is an S4 class with methods for efficient computation by adelie.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' p <- 20
#' K <- 2
#' mat <- matrix(rnorm(n * p), n, p)
#' out <- matrix.kronecker_eye(mat, K)
#' mat <- matrix.dense(mat)
#' out <- matrix.kronecker_eye(mat, K)
#' @export
matrix.kronecker_eye <- function(
    mat,
    K=1,
    n_threads =1
)
{
    if (is.matrix(mat) || is.array((mat)) || is.data.frame((mat))) {
        mat <- as.matrix(mat)
        dispatcher <- RMatrixNaiveKroneckerEyeDense64F
    } else {
        dispatcher <- RMatrixNaiveKroneckerEye64
    }
    input <- list(
        "mat"=mat,
        "K"=K,
        "n_threads"=n_threads
    )
    out <- new(dispatcher, input)
    attr(out, "_mat") <- mat
    out
}

#' Creates a lazy covariance matrix.
#'
#' @param   mat     A dense  data matrix to be used with the \code{gaussian_cov()} solver.
#' @param   n_threads   Number of threads.
#' @return Lazy covariance matrix. This is essentially the same matrix, but with a setup to create covariance terms as needed on the fly.
#' The object is an S4 class with methods for efficient computation by adelie.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' p <- 20
#' mat <- matrix(rnorm(n * p), n, p)
#' out <- matrix.lazy_cov(mat)
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
#' @param   mat     A dense matrix, which can include factors with levels coded as non-negative integers.
#' @param   levels      Number of levels for each of the columns of \code{mat}, with \code{1} representing a quantitative feature. A factor with \code{K} levels should be represented by the numbers \code{0,1,...,K-1}.
#' @param   n_threads   Number of threads.
#' @return One-hot encoded matrix. All the factor columns, with levels>1, are replaced by a collection of one-hot encoded versions (dummy matrices). The resulting matrix has \code{sum(levels)} columns.
#' The object is an S4 class with methods for efficient computation by adelie. Note that some of the arguments are transformed to C++ base 0 for internal use, and if the object is examined, it will reflect that.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' p <- 20
#' mat <- matrix(rnorm(n * p), n, p)
#' fac <- sample(0:5, n, replace = TRUE)
#' mat=cbind(fac,mat)
#' levels <- c(6, rep(1,p))
#' out <- matrix.one_hot(mat, levels = levels)
#' @export
matrix.one_hot <- function(
    mat,
    levels =NULL,
    n_threads =1
)
{
    d <- ncol(mat)
    if (is.null(levels)) {# levels of 1 are translated to 0
        levels <- integer(d)
    }
    else levels[levels==1] <- 0
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
#' @param   io  IO handler.
#' @param   n_threads   Number of threads.
#' @return SNP phased, ancestry matrix.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 123
#' s <- 423
#' A <- 8
#' filename <- paste(tempdir(), "snp_phased_ancestry_dummy.snpdat", sep="/")
#' handle <- io.snp_phased_ancestry(filename)
#' calldata <- matrix(
#'     as.integer(sample.int(
#'         2, n * s * 2,
#'         replace=TRUE,
#'         prob=c(0.7, 0.3)
#'     ) - 1),
#'     n, s * 2
#' )
#' ancestries <- matrix(
#'     as.integer(sample.int(
#'         A, n * s * 2,
#'         replace=TRUE,
#'         prob=rep_len(1/A, A)
#'     ) - 1),
#'     n, s * 2
#' )
#' handle$write(calldata, ancestries, A, 1)
#' out <- matrix.snp_phased_ancestry(handle)
#' file.remove(filename)
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
#' @return SNP unphased matrix.
#' @examples
#' n <- 123
#' s <- 423
#' filename <- paste(tempdir(), "snp_unphased_dummy.snpdat", sep="/")
#' handle <- io.snp_unphased(filename)
#' mat <- matrix(
#'     as.integer(sample.int(
#'         3, n * s,
#'         replace=TRUE,
#'         prob=c(0.7, 0.2, 0.1)
#'     ) - 1),
#'     n, s
#' )
#' impute <- double(s)
#' handle$write(mat, "mean", impute, 1)
#' out <- matrix.snp_unphased(handle)
#' file.remove(filename)
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

#' Creates a sparse matrix object.
#'
#' @param   mat     A sparse matrix.
#' @param   method  Method type, with  default \code{method="naive"}.
#' If \code{method="cov"}, the matrix is used with the solver \code{gaussian_cov()}.
#' Used for \code{glm.gaussian()} and \code{glm.multigaussian()} families. Generally "naive" is used for wide matrices, and "cov" for tall matrices.
#' If \code{method="constraint"}, the matrix is used as input to the constraint objects.
#' @param   n_threads   Number of threads.
#' @return Sparse matrix object.
#' The object is an S4 class with methods for efficient computation by adelie.
#' @examples
#' n <- 100
#' p <- 20
#' X_dense <- matrix(rnorm(n * p), n, p)
#' X_sp <- as(X_dense, "dgCMatrix")
#' out <- matrix.sparse(X_sp, method="naive")
#' A_dense <- t(X_dense) %*% X_dense
#' A_sp <- as(A_dense, "dgCMatrix")
#' out <- matrix.sparse(A_sp, method="cov")
#' out <- matrix.sparse(X_sp, method="constraint")
#' @export
matrix.sparse <- function(
    mat,
    method = c("naive","cov","constraint"),
    n_threads =1
    )
{
    method = match.arg(method)
    if(inherits(mat,"sparseMatrix")){
        if(!inherits(mat,"dgCMatrix"))
            mat=as(as(as(mat, "generalMatrix"), "CsparseMatrix"), "dMatrix")
    }
    else stop("matrix is not a 'sparseMatrix'")
    if (method == "constraint") {
        mat <- t(mat)
    }
    dispatcher <- c(
        "naive" = RMatrixNaiveSparse64F,
        "cov" = RMatrixCovSparse64F,
        "constraint" = RMatrixConstraintSparse64F
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
#' @param   mat     An \code{adelie} matrix.
#' @param   centers     The center values. Default is to use the column means.
#' @param   scales     The scale values. Default is to use the sample standard deviations.
#' @param   weights  Observation weight vector, which defaults to 1/n per observation.
#' @param   ddof        Degrees of freedom for standard deviations, with default 0 (1/n). The alternative is 1 leading to 1/(n-1).
#' @param   n_threads   Number of threads.
#' @return Standardized matrix.
#' The object is an S4 class with methods for efficient computation by adelie.
#' Conventions depend on the matrix class. For example, if a matrix is constructed using `matrix.onehot()`, only the quantitative variables are standardized.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' p <- 20
#' X <- matrix(rnorm(n * p), n, p)
#' out <- matrix.standardize(matrix.dense(X))
#' @export

matrix.standardize <- function(
    mat,
    centers =NULL,
    scales =NULL,
    weights=NULL,
    ddof =0,
    n_threads =1
)
{
    n <- mat$rows
    p <- mat$cols
    if (is.null(weights)) {
        weights <- rep(1/n, n)
    } else {
        weights <- weights / sum(weights)
    }
    is_centers_none <- is.null(centers)

    if (is_centers_none) {
        centers <- mat$mean(weights)
    }
    if (is.null(scales)) {
        vars <- mat$var(centers, weights)
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
#' @param   mat     The \code{adelie} matrix to subset.
#' @param   indices     Vector of indices to subset the matrix.
#' @param   axis        The axis along which to subset (2 is columns, 1 is rows).
#' @param   n_threads   Number of threads.
#' @return Matrix subsetted along the appropriate axis.
#' The object is an S4 class with methods for efficient computation by adelie.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' n <- 100
#' p <- 20
#' X <- matrix.dense(matrix(rnorm(n * p), n, p))
#' indices <- c(1, 3, 10)
#' out <- matrix.subset(X, indices, axis=1)
#' out <- matrix.subset(X, indices, axis=2)
#' @export
matrix.subset <- function(
    mat,
    indices,
    axis =1,
    n_threads =1
)
{
    if(axis %in% c(1,2))axis = axis -1 # C++ base 0
    else stop("axis can take values 1 (rows) or 2 (columns)  only")

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
