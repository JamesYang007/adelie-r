#' Creates a box constraint. 
#'
#' @param   lower       Lower bound.
#' @param   upper       Upper bound.  
#' @param   max_iters   Maximum number of proximal Newton iterations.
#' @param   tol         Convergence tolerance for proximal Newton.
#' @param   n_threads   Number of threads.
#' @return Box constraint. 
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @examples
#' 
#' lower <- double(10)
#' upper <- lower + 5
#' c <- constraint.box(
#'     lower=lower,
#'     upper=upper
#' )
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
