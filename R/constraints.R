#' Create a box constraint for a group.
#'
#' A box constraint sets upper and lower bounds for coefficients in a model.
#' This is done per group, and this function is used separately to set the bounds for each group in the model. The constraints are returned as a list, with number of elements the number of groups. List entries can be `NULL`, which means no constraints for that group. Currently works with single-response models (so `glm.multinomial` and `glm.multigaussian` are excluded).
#'
#'
#' @param   lower       lower bound for each coefficient in the group. If the group has `m` variables, this should be a vector of length `m`. Values can be `-Inf`.
#' @param   upper        upper bound for each coefficient in the group. If the group has `m` variables, this should be a vector of length `m`. Values can be `Inf`.
#' @param   max_iters   maximum number of proximal Newton iterations; default is 100.
#' @param   tol         convergence tolerance for proximal Newton; default is 1e-9.
#' @return Box constraint object.
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#'
#' # Group of length 10, with positivity constraint on all the coefficients.
#' lower <- rep(0,10)
#' upper <- rep(Inf,10)
#' cont <- constraint.box(
#'                        lower=lower,
#'                        upper=upper
#'                        )
#'
#' # No groups, 10 variables, and positivity constraints on all parameters.
#' box_cont <- constraint.box(
#'                            upper=Inf,
#'                            lower=0
#'                            )
#' cont <- rep(list(box_cont),10)
#'
#' # Same, but only positivity on first 5 coefficients
#' cont <- c(rep(list(box_cont), 5), rep(list(NULL), 5))
#'
#' @export
constraint.box <- function(
    lower,
    upper,
    max_iters=100,
    tol=1e-9
)
{
    pinball_max_iters = 10000
    pinball_tol = 1e-7
    slack = 1e-4
    lower = -lower
    input <- list(
        "l"=lower,
        "u"=upper,
        max_iters=max_iters,
        tol=tol,
        pinball_max_iters=pinball_max_iters,
        pinball_tol=pinball_tol,
        slack=slack
    )
    out <- new(RConstraintBox64, input)
    attr(out, "_lower") <- lower
    attr(out, "_upper") <- upper
    out
}
