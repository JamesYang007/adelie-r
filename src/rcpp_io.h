#pragma once
#include "decl.h"
#include <adelie_core/io/io_snp_base.ipp>
#include <adelie_core/io/io_snp_unphased.ipp>
#include <adelie_core/io/io_snp_phased_ancestry.ipp>

using io_snp_base_t = ad::io::IOSNPBase<std::shared_ptr<char>>;
using io_snp_unphased_t = ad::io::IOSNPUnphased<std::shared_ptr<char>>;
using io_snp_phased_ancestry_t = ad::io::IOSNPPhasedAncestry<std::shared_ptr<char>>;

class RIOSNPUnphased: public io_snp_unphased_t
{
    using base_t = io_snp_unphased_t;
public:
    using value_t = int;
    using vec_value_t = ad::util::colvec_type<double>;
    using colarr_value_t = ad::util::colarr_type<value_t>;

    using base_t::base_t;
    using base_t::rows;
    using base_t::snps;
    using base_t::cols;

    auto write(
        const Eigen::Map<colarr_value_t>& calldata, 
        const std::string& impute_method,
        Eigen::Map<vec_value_t> impute,
        size_t n_threads
    )
    {
        ad::util::colarr_type<int8_t> calldata8(calldata.rows(), calldata.cols());
        calldata8 = calldata.template cast<int8_t>();
        return std::get<0>(base_t::write(calldata8, impute_method, impute, n_threads));
    }
};

class RIOSNPPhasedAncestry: public io_snp_phased_ancestry_t
{
    using base_t = io_snp_phased_ancestry_t;
public:
    using value_t = int;
    using colarr_value_t = ad::util::colarr_type<value_t>;

    using base_t::base_t;
    using base_t::rows;
    using base_t::snps;
    using base_t::cols;

    auto write(
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
        return std::get<0>(base_t::write(calldata8, ancestries8, A, n_threads));
    }
};

RCPP_EXPOSED_CLASS(RIOSNPUnphased)
RCPP_EXPOSED_CLASS(RIOSNPPhasedAncestry)

using r_io_snp_unphased_t = RIOSNPUnphased;
using r_io_snp_phased_ancestry_t = RIOSNPPhasedAncestry;