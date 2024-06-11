#' @export
io.snp_phased_ancestry <- function(filename, read_mode="file")
{
    new(RIOSNPPhasedAncestry, filename, read_mode)
}

#' @export
io.snp_unphased <- function(filename, read_mode="file")
{
    new(RIOSNPUnphased, filename, read_mode)
}