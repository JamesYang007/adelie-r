#if defined(__APPLE__)
#ifdef ADELIE_CORE_EIGEN_USE_BLAS
#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#endif
#endif
#endif
#include "decl.h"
#include <adelie_core/matrix/utils.hpp>

namespace ad = adelie_core;

using value_t = double;
using colmat_value_t = ad::util::colmat_type<value_t>;
using map_colmat_value_t = Eigen::Map<colmat_value_t>;

colmat_value_t dgemtm(
    const map_colmat_value_t& mat, 
    size_t n_threads
)
{
    const auto p = mat.cols();
    colmat_value_t out(p, p);
    ad::matrix::dgemtm(mat, out, n_threads);
    return out;
}

RCPP_MODULE(adelie_core_matrix_utils_blas)
{
    Rcpp::function("dgemtm", dgemtm);
}