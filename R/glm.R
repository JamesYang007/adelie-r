render_inputs_ <- function(y, weights)
{
    if (is.matrix(y)) {
        n <- nrow(y)
        K <- ncol(y)
        y <- matrix(as.double(y), n, K)
    } else {
        n <- length(y)
        y <- as.double(y)
    }

    if (is.null(weights)) {
        weights <- as.double(rep_len(1/n, n))
    } else {
        weights_sum <- sum(weights)
        if (abs(weights_sum - 1) > 1e-14) {
            weights <- as.double(weights / weights_sum)
        }
    }

    list(
        y=y,
        weights=weights
    )
}

#' Creates a Binomial GLM family object.
#'
#' A GLM family object specifies the type of model fit, provides the appropriate response object and makes sure it is represented in the right form for the model family, and allows for optional parameters such as a weight vector.
#'
#' @param   y     Binary response vector, with values 0 or 1, or a logical vector. Alternatively, if data are represented by a two-column matrix of proportions (with row-sums = 1), then one can provide one of the columns as the response. This is useful for grouped binomial data, where each observation represents the result of \code{m[i]} successes out of \code{n[i]} trials. Then the response is provided as \code{y[i] = m[i]/n[i]} and the corresponding element of the weight vector as \code{w[i]=n[i]}. Alternatively can use \code{glm.multinomial()} instead.
#' @param   weights Observation weight vector, with default \code{NULL}.
#' @param   link    The link function type, with choice \code{"logit"} (default) or \code{"probit"}).
#' @return Binomial GLM object.
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{glm.gaussian}, \code{glm.binomial}, \code{glm.poisson},  \code{glm.multinomial}, \code{glm.multigaussian}, \code{glm.cox}.
#' @examples
#' n <- 100
#' y <- rbinom(n, 1, 0.5)
#' obj <- glm.binomial(y)
#' @export
glm.binomial <- function(y, weights=NULL, link="logit")
{

    input <- render_inputs_(y, weights)
    y <- input[["y"]]
    weights <- input[["weights"]]
    dispatcher <- c(
        "logit" = RGlmBinomialLogit64,
        "probit" = RGlmBinomialProbit64
    )
    out <- new(dispatcher[[link]], input)
    attr(out, "_y") <- y
    attr(out, "_weights") <- weights
    out
}

#' Creates a Cox GLM family object.
#'
#' A GLM family object specifies the type of model fit, provides the appropriate response object and makes sure it is represented in the right form for the model family, and allows for optional parameters such as a weight vector.
#'
#' @param   stop     Stop time vector.
#' @param   status     Binary status vector of same length as \code{stop}, with 1 a "death", and 0 censored.
#' @param   strata      TODO
#' @param   start     Start time vector. Default is a vector of \code{-Inf} of same length as \code{stop}.
#' @param   weights Observation weights, with default \code{NULL}.
#' @param tie_method    The tie-breaking method - one of  \code{"efron"} (default) or \code{"breslow"}.
#' @return Cox GLM object.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{glm.gaussian}, \code{glm.binomial}, \code{glm.poisson},  \code{glm.multinomial}, \code{glm.multigaussian}, \code{glm.cox}.
#' @examples
#' n <- 100
#' start <- sample.int(20, size=n, replace=TRUE)
#' stop <- start + 1 + sample.int(5, size=n, replace=TRUE)
#' # TODO: add strata?
#' status <- rbinom(n, 1, 0.5)
#' obj <- glm.cox(start, stop, status)
#' @export
glm.cox <- function(stop, status, start = -Inf, strata=NULL, weights=NULL, tie_method=c("efron","breslow"))
{
    tie_method=match.arg(tie_method)
    input <- render_inputs_(status, weights)
    n <- length(stop)
    start <- rep(as.double(start), length.out = n)
    stop <- as.double(stop)
    # C++ is 0-indexed
    if (is.null(strata)) {
        strata <- integer(n)
    } else {
        strata <- as.integer(strata - 1) 
    }
    status <- input[["y"]]
    weights <- input[["weights"]]
    input <- list(
        "start"=start,
        "stop"=stop,
        "status"=status,
        "strata"=strata,
        "weights"=weights,
        "tie_method"=tie_method
    )
    out <- new(RGlmCox64, input)
    attr(out, "_start") <- start
    attr(out, "_stop") <- stop
    attr(out, "_status") <- status
    attr(out, "_strata") <- strata
    attr(out, "_weights") <- weights
    attr(out, "_tie_method") <- tie_method
    out
}

#' Creates a Gaussian GLM family object.
#'
#' A GLM family object specifies the type of model fit, provides the appropriate response object and makes sure it is represented in the right form for the model family, and allows for optional parameters such as a weight vector.
#'
#' @param   y     Response vector.
#' @param   weights Observation weight vector, with default \code{NULL}.
#' @param   opt     If \code{TRUE} (default), an optimized routine is run.
#' @return Gaussian GLM
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{glm.gaussian}, \code{glm.binomial}, \code{glm.poisson},  \code{glm.multinomial}, \code{glm.multigaussian}, \code{glm.cox}.
#' @examples
#' n <- 100
#' y <- rnorm(n)
#' obj <- glm.gaussian(y)
#' @export
glm.gaussian <- function(y, weights=NULL, opt=TRUE)
{
    input <- render_inputs_(y, weights)
    y <- input[["y"]]
    weights <- input[["weights"]]
    out <- new(RGlmGaussian64, input)
    attr(out, "_y") <- y
    attr(out, "_weights") <- weights
    attr(out, "opt") <- opt
    out
}

#' Creates a MultiGaussian GLM family object.
#'
#' A GLM family object specifies the type of model fit, provides the appropriate response object and makes sure it is represented in the right form for the model family, and allows for optional parameters such as a weight vector.
#'
#' @param   y     Response matrix, with two or more columns.
#' @param   weights Observation weight vector, with default \code{NULL}.
#' @param   opt     If \code{TRUE} (default), an optimized routine is run.
#' @return MultiGaussian GLM object.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{glm.gaussian}, \code{glm.binomial}, \code{glm.poisson},  \code{glm.multinomial}, \code{glm.multigaussian}, \code{glm.cox}.
#' @examples
#' n <- 100
#' K <- 5
#' y <- matrix(rnorm(n*K), n, K)
#' obj <- glm.multigaussian(y)
#' @export
glm.multigaussian <- function(y, weights=NULL, opt=TRUE)
{
    input <- render_inputs_(y, weights)
    y <- input[["y"]]
    weights <- input[["weights"]]
    yT <- t(y)
    input <- list("yT"=yT, "weights"=weights)
    out <- new(RGlmMultiGaussian64, input)
    attr(out, "_yT") <- yT
    attr(out, "_weights") <- weights
    attr(out, "opt") <- opt
    out
}

#' Creates a Multinomial GLM family object.
#'
#' A GLM family object specifies the type of model fit, provides the appropriate response object and makes sure it is represented in the right form for the model family, and allows for optional parameters such as a weight vector.
#'
#' @param   y     Response matrix with \code{K>1} columns, and row sums equal to 1. This can either be a "one-hot" encoded version of a K-category factor variable, or else a matrix of proportions. This is useful for grouped multinomial data, where column \code{y[i, k]} represents the proportion of outcomes in category k in \code{n[i]} trials. Then the corresponding element of the weight vector is \code{w[i]=n[i]}.
#' @param   weights Observation weights.
#' @return Multinomial GLM object.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{glm.gaussian}, \code{glm.binomial}, \code{glm.poisson},  \code{glm.multinomial}, \code{glm.multigaussian}, \code{glm.cox}.
#' @examples
#' n <- 100
#' K <- 5
#' y <- t(rmultinom(n, 1, rep(1/K, K)))
#' obj <- glm.multinomial(y)
#' @export
glm.multinomial <- function(y, weights=NULL)
{
    input <- render_inputs_(y, weights)
    y <- input[["y"]]
    weights <- input[["weights"]]
    yT <- t(y)
    input <- list("yT"=yT, "weights"=weights)
    out <- new(RGlmMultinomial64, input)
    attr(out, "_yT") <- yT
    attr(out, "_weights") <- weights
    out
}

#' Creates a Poisson GLM family object.
#'
#' A GLM family object specifies the type of model fit, provides the appropriate response object and makes sure it is represented in the right form for the model family, and allows for optional parameters such as a weight vector.
#'
#' @param   y     Response vector of non-negative counts.
#' @param   weights Observation weight vector, with default \code{NULL}.
#' @return Poisson GLM object.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{glm.gaussian}, \code{glm.binomial}, \code{glm.poisson},  \code{glm.multinomial}, \code{glm.multigaussian}, \code{glm.cox}.
#' @examples
#' n <- 100
#' y <- rpois(n, 1)
#' obj <- glm.poisson(y)
#' @export
glm.poisson <- function(y, weights=NULL)
{
    input <- render_inputs_(y, weights)
    y <- input[["y"]]
    weights <- input[["weights"]]
    out <- new(RGlmPoisson64, input)
    attr(out, "_y") <- y
    attr(out, "_weights") <- weights
    out
}
