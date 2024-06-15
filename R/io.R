#' IO handler for SNP phased, ancestry matrix.
#' 
#' @param   filename    File name.
#' @param   read_mode   Reading mode.
#' @returns IO handler for SNP phased, ancestry data.
#' @examples
#' n <- 123
#' s <- 423
#' A <- 8
#' filename <- paste(tempdir(), "snp_phased_ancestry_dummy.snpdat", sep="/")
#' handle <- io.snp_phased_ancestry(filename)
#' calldata <- matrix(
#'     as.integer(sample.int(
#'         2, n * s * 2,
#'         replace=TRUE,
#'         prob=c(0.7, 0.3)
#'     ) - 1),
#'     n, s * 2
#' )
#' ancestries <- matrix(
#'     as.integer(sample.int(
#'         A, n * s * 2,
#'         replace=TRUE,
#'         prob=rep_len(1/A, A)
#'     ) - 1),
#'     n, s * 2
#' )
#' handle$write(calldata, ancestries, A, 1)
#' handle$read()
#' file.remove(filename)
#' @export
io.snp_phased_ancestry <- function(filename, read_mode="file")
{
    new(RIOSNPPhasedAncestry, filename, read_mode)
}

#' IO handler for SNP unphased matrix.
#' 
#' @param   filename    File name.
#' @param   read_mode   Reading mode.
#' @returns IO handler for SNP unphased data.
#' @examples
#' n <- 123
#' s <- 423
#' filename <- paste(tempdir(), "snp_unphased_dummy.snpdat", sep="/")
#' handle <- io.snp_unphased(filename)
#' mat <- matrix(
#'     as.integer(sample.int(
#'         3, n * s, 
#'         replace=TRUE, 
#'         prob=c(0.7, 0.2, 0.1)
#'     ) - 1),
#'     n, s
#' )
#' impute <- double(s)
#' handle$write(mat, "mean", impute, 1)
#' handle$read()
#' file.remove(filename)
#' @export
io.snp_unphased <- function(filename, read_mode="file")
{
    new(RIOSNPUnphased, filename, read_mode)
}