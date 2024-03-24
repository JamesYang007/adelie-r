#' @export
set_configs <- function(name, value=NULL)
{
    configs <- new(Configs)
    if (is.null(value)) {
        value <- configs[[paste(name, "_def", sep="")]]
    }
    configs[[name]] <- value
}