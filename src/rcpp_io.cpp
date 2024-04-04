#include <Rcpp.h>
#include <RcppEigen.h>
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>

namespace ad = adelie_core;

using value_t = int;
using colarr_value_t = ad::util::colarr_type<value_t>;
using io_snp_base_t = ad::io::IOSNPBase;
using io_snp_unphased_t = ad::io::IOSNPUnphased;
using io_snp_phased_ancestry_t = ad::io::IOSNPPhasedAncestry;

auto make_io_snp_unphased(const std::string& filename)
{
    // TODO: generalize
    return io_snp_unphased_t(filename, "file");
}

auto make_io_snp_phased_ancestry(const std::string& filename)
{
    // TODO: generalize
    return io_snp_phased_ancestry_t(filename, "file");
}

auto write(
    io_snp_unphased_t* io, 
    const Eigen::Map<colarr_value_t>& calldata, 
    size_t n_threads
)
{
    ad::util::colarr_type<int8_t> calldata8(calldata.rows(), calldata.cols());
    calldata8 = calldata.template cast<int8_t>();
    return io->write(calldata8, n_threads);
}

auto write(
    io_snp_phased_ancestry_t* io, 
    const Eigen::Map<colarr_value_t>& calldata, 
    const Eigen::Map<colarr_value_t>& ancestries, 
    size_t A,
    size_t n_threads
)
{
    ad::util::colarr_type<int8_t> calldata8(calldata.rows(), calldata.cols());
    ad::util::colarr_type<int8_t> ancestries8(ancestries.rows(), ancestries.cols());
    calldata8 = calldata.template cast<int8_t>();
    ancestries8 = ancestries.template cast<int8_t>();
    return io->write(calldata8, ancestries8, A, n_threads);
}

RCPP_EXPOSED_AS(io_snp_base_t)
RCPP_EXPOSED_WRAP(io_snp_unphased_t)
RCPP_EXPOSED_WRAP(io_snp_phased_ancestry_t)

RCPP_MODULE(adelie_core_io)
{
    //Rcpp::class_<io_snp_base_t>("IOSNPBase")
    //    .constructor<std::string, std::string>()
    //    .method("endian", &io_snp_base_t::endian)
    //    .method("read", &io_snp_base_t::read)
    //    ;
    //Rcpp::class_<io_snp_unphased_t>("IOSNPUnphased")
    //    .constructor<std::string, std::string>()
    //    .derives<io_snp_base_t>("IOSNPBase")
    //    .method("rows", &io_snp_unphased_t::rows)
    //    .method("snps", &io_snp_unphased_t::snps)
    //    .method("cols", &io_snp_unphased_t::cols)
    //    .method("write", &write)
    //    ;
    //Rcpp::class_<io_snp_phased_ancestry_t>("IOSNPPhasedAncestry")
    //    .constructor<std::string, std::string>()
    //    .derives<io_snp_base_t>("IOSNPBase")
    //    .method("rows", &io_snp_phased_ancestry_t::rows)
    //    .method("snps", &io_snp_phased_ancestry_t::snps)
    //    .method("cols", &io_snp_phased_ancestry_t::cols)
    //    .method("write", &write)
    //    ;

    //Rcpp::function(
    //    "make_io_snp_unphased",
    //    &make_io_snp_unphased
    //);
    //Rcpp::function(
    //    "make_io_snp_phased_ancestry",
    //    &make_io_snp_phased_ancestry
    //);
}