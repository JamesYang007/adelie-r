#include "rcpp_io.hpp"

/* Factory functions */

auto make_r_io_snp_unphased(
    const std::string& filename,
    const std::string& read_mode
)
{
    return r_io_snp_unphased_t(filename, read_mode);
}

auto make_r_io_snp_phased_ancestry(
    const std::string& filename,
    const std::string& read_mode
)
{
    return r_io_snp_phased_ancestry_t(filename, read_mode);
}

RCPP_MODULE(adelie_core_io)
{
    Rcpp::class_<io_snp_base_t>("IOSNPBase")
        .constructor<std::string, std::string>()
        .property("endian", &io_snp_base_t::endian, "")
        .property("is_read", &io_snp_base_t::is_read, "")
        .method("read", &io_snp_base_t::read, "")
        ;
    Rcpp::class_<io_snp_unphased_t>("IOSNPUnphased")
        .derives<io_snp_base_t>("IOSNPBase")
        .constructor<std::string, std::string>()
        .property("rows", &r_io_snp_unphased_t::rows, "")
        .property("snps", &r_io_snp_unphased_t::snps, "")
        .property("cols", &r_io_snp_unphased_t::cols, "")
        ;
    Rcpp::class_<io_snp_phased_ancestry_t>("IOSNPPhasedAncestry")
        .derives<io_snp_base_t>("IOSNPBase")
        .constructor<std::string, std::string>()
        .property("rows", &io_snp_phased_ancestry_t::rows, "")
        .property("snps", &io_snp_phased_ancestry_t::snps, "")
        .property("cols", &io_snp_phased_ancestry_t::cols, "")
        ;
    Rcpp::class_<r_io_snp_unphased_t>("RIOSNPUnphased")
        .derives<io_snp_unphased_t>("IOSNPUnphased")
        .constructor<std::string, std::string>()
        .method("write", &r_io_snp_unphased_t::write)
        ;
    Rcpp::class_<r_io_snp_phased_ancestry_t>("RIOSNPPhasedAncestry")
        .derives<io_snp_phased_ancestry_t>("IOSNPPhasedAncestry")
        .constructor<std::string, std::string>()
        .method("write", &r_io_snp_phased_ancestry_t::write)
        ;

    Rcpp::function(
        "make_r_io_snp_unphased",
        &make_r_io_snp_unphased
    );
    Rcpp::function(
        "make_r_io_snp_phased_ancestry",
        &make_r_io_snp_phased_ancestry
    );
}