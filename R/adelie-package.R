#' @import Rcpp
#' @useDynLib adelie, .registration = TRUE
NULL
Rcpp::loadModule("adelie_core_configs", TRUE)
Rcpp::loadModule("adelie_core_glm", TRUE)
Rcpp::loadModule("adelie_core_io", TRUE)
Rcpp::loadModule("adelie_core_matrix", TRUE)
Rcpp::loadModule("adelie_core_solver", TRUE)
Rcpp::loadModule("adelie_core_state", TRUE)