#' IO handler for SNP phased, ancestry matrix.
#' 
#' @param   filename    File name.
#' @param   read_mode   Reading mode.
#' @export
io.snp_phased_ancestry <- function(filename, read_mode="file")
{
    new(RIOSNPPhasedAncestry, filename, read_mode)
}

#' IO handler for SNP unphased matrix.
#' 
#' @param   filename    File name.
#' @param   read_mode   Reading mode.
#' @export
io.snp_unphased <- function(filename, read_mode="file")
{
    new(RIOSNPUnphased, filename, read_mode)
}