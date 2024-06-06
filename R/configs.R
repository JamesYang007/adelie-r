#' @export
set_configs <- function(name, value=NULL)
{
    configs <- new(RConfigs)
    if (is.null(value)) {
        value <- configs[[paste(name, "_def", sep="")]]
    }
    configs[[name]] <- value
}