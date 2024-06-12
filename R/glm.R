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
            warning("Normalizing weights to sum to 1.")
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
#' @param   y     Response vector.
#' @param   weights Observation weights. 
#' @param   link    The link function type.
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
#' @param   start     Start time vector.
#' @param   stop     Stop time vector.
#' @param   status     Status vector.
#' @param   weights Observation weights. 
#' @param tie_method    The tie-breaking method.
#' @export
glm.cox <- function(start, stop, status, weights=NULL, tie_method="efron")
{
    input <- render_inputs_(status, weights)
    start <- as.double(start)
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
#' @param   weights Observation weights. 
#' @param   opt     If \code{TRUE}, an optimized routine is run.
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
#' @param   y     Response vector.
#' @param   weights Observation weights. 
#' @param   opt     If \code{TRUE}, an optimized routine is run.
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