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
#' @param   y     Binary response vector, with values 0 or 1, or a logical vector.
#' @param   weights Observation weight vector, with default \code{NULL}.
#' @param   link    The link function type, with choice \code{"logit"} (default) or \code{"probit"}).
#' @returns Binomial GLM object.
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
#' @param   start     Start time vector. Default is a vector of \code{-Inf} of same length as \code{stop}.
#' @param   stop     Stop time vector.
#' @param   status     Binary status vector of same length ast \code{stop}, with 1 a "death", and 0 censored.
#' @param   weights Observation weights, with default \code{NULL}.
#' @param tie_method    The tie-breaking method - one of  \code{"efron"} (default) or \code{"breslow"}.
#' @returns Cox GLM object.
#' @examples
#' n <- 100
#' start <- sample.int(20, size=n, replace=TRUE)
#' stop <- start + 1 + sample.int(5, size=n, replace=TRUE)
#' status <- rbinom(n, 1, 0.5)
#' obj <- glm.cox(start, stop, status)
#' @export
glm.cox <- function(start = -Inf, stop, status, weights=NULL, tie_method=c("efron","breslow"))
{
    tie_method=match.arg(tie_method)
    input <- render_inputs_(status, weights)
    n <- length(stop)
    start <- rep(as.double(start), length = n)
    stop <- as.double(stop)
    status <- input[["y"]]
    weights <- input[["weights"]]
    input <- list(
        "start"=start,
        "stop"=stop,
        "status"=status,
        "weights"=weights,
        "tie_method"=tie_method
    )
    out <- new(RGlmCox64, input)
    attr(out, "_start") <- start
    attr(out, "_stop") <- stop
    attr(out, "_status") <- status
    attr(out, "_weights") <- weights
    attr(out, "_tie_method") <- tie_method
    out
}

#' Creates a Gaussian GLM family object.
#'
#' @param   y     Response vector.
#' @param   weights Observation weight vector, with default \code{NULL}.
#' @param   opt     If \code{TRUE}, an optimized routine is run.
#' @returns Gaussian GLM
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
#' @param   y     Response matrix, with two or more columns.
#' @param   weights Observation weight vector, with default \code{NULL}.
#' @param   opt     If \code{TRUE}, an optimized routine is run.
#' @returns MultiGaussian GLM object.
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
#' @param   y     Response vector.
#' @param   weights Observation weights.
#' @returns Multinomial GLM object.
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
#' @param   y     Response vector.
#' @param   weights Observation weights.
#' @returns Poisson GLM object.
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
