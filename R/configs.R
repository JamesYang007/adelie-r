#' Set configuration settings.
#' 
#' @param name  Configuration variable name.
#' @param value Value to assign to the configuration variable.
#' @export
set_configs <- function(name, value=NULL)
{
    configs <- new(RConfigs)
    if (is.null(value)) {
        value <- configs[[paste(name, "_def", sep="")]]
    }
    configs[[name]] <- value
}