#' Set configuration settings.
#' 
#' @param name  Configuration variable name.
#' @param value Value to assign to the configuration variable.
#' @returns Assigned value.
#' @examples
#' set_configs("hessian_min", 1e-6)
#' set_configs("hessian_min")
#' @export
set_configs <- function(name, value=NULL)
{
    configs <- new(RConfigs)
    if (is.null(value)) {
        value <- configs[[paste(name, "_def", sep="")]]
    }
    configs[[name]] <- value
    value
}