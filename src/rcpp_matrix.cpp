#include <Rcpp.h>
#include <RcppEigen.h>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_cov_dense.hpp>
#include <adelie_core/matrix/matrix_cov_lazy.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/matrix_naive_dense.hpp>
#include <adelie_core/matrix/matrix_naive_concatenate.hpp>
#include <adelie_core/matrix/matrix_naive_kronecker_eye.hpp>
#include <adelie_core/matrix/matrix_naive_snp_unphased.hpp>
#include <adelie_core/matrix/matrix_naive_snp_phased_ancestry.hpp>

namespace ad = adelie_core;

using value_t = double;
using vec_value_t = ad::util::colvec_type<value_t>;
using dense_64F_t = ad::util::colmat_type<value_t>;

using matrix_cov_base_64_t = ad::matrix::MatrixCovBase<value_t>;
using matrix_cov_dense_64F_t = ad::matrix::MatrixCovDense<dense_64F_t>;
using matrix_cov_lazy_64F_t = ad::matrix::MatrixCovLazy<dense_64F_t>;

using matrix_naive_base_64_t = ad::matrix::MatrixNaiveBase<value_t>;
using matrix_naive_dense_64F_t = ad::matrix::MatrixNaiveDense<dense_64F_t>;
using matrix_naive_concatenate_64_t = ad::matrix::MatrixNaiveConcatenate<value_t>;
using matrix_naive_kronecker_eye_64_t = ad::matrix::MatrixNaiveKroneckerEye<value_t>;
using matrix_naive_kronecker_eye_dense_64F_t = ad::matrix::MatrixNaiveKroneckerEyeDense<dense_64F_t>;
using matrix_naive_snp_unphased_64_t = ad::matrix::MatrixNaiveSNPUnphased<value_t>;
using matrix_naive_snp_phased_ancestry_64_t = ad::matrix::MatrixNaiveSNPPhasedAncestry<value_t>;

auto make_matrix_cov_dense_64F(
    const Eigen::Map<dense_64F_t>& mat,
    size_t n_threads
)
{
    return matrix_cov_dense_64F_t(mat, n_threads);
}

auto make_matrix_cov_lazy_64F(
    const Eigen::Map<dense_64F_t>& mat,
    size_t n_threads
)
{
    return matrix_cov_lazy_64F_t(mat, n_threads);
}

auto make_matrix_naive_concatenate_64(
    Rcpp::List mat_list_r,
    size_t n_threads
)
{
    std::vector<matrix_naive_base_64_t*> mat_list;
    for (auto obj : mat_list_r) {
        mat_list.push_back(Rcpp::as<matrix_naive_base_64_t*>(obj));
    }
    return matrix_naive_concatenate_64_t(mat_list, n_threads);
}

auto make_matrix_naive_dense_64F(
    const Eigen::Map<dense_64F_t>& dense,
    size_t n_threads
)
{
    return matrix_naive_dense_64F_t(dense, n_threads);
}

auto make_matrix_naive_kronecker_eye_64(
    matrix_naive_base_64_t& mat,
    size_t K,
    size_t n_threads
)
{
    return matrix_naive_kronecker_eye_64_t(mat, K, n_threads);
}

auto make_matrix_naive_kronecker_eye_dense_64F(
    const Eigen::Map<dense_64F_t>& mat,
    size_t K,
    size_t n_threads
)
{
    return matrix_naive_kronecker_eye_dense_64F_t(mat, K, n_threads);
}

auto make_matrix_naive_snp_unphased_64(
    const std::string& filename,
    size_t n_threads
)
{
    return matrix_naive_snp_unphased_64_t(filename, n_threads);
}

auto make_matrix_naive_snp_phased_ancestry_64(
    const std::string& filename,
    size_t n_threads
)
{
    return matrix_naive_snp_phased_ancestry_64_t(filename, n_threads);
}

void mul(
    matrix_naive_base_64_t* X,
    const Eigen::Map<vec_value_t>& v, 
    const Eigen::Map<vec_value_t>& weights,
    Eigen::Map<vec_value_t> out
)
{
    X->mul(v, weights, out);
}

RCPP_EXPOSED_AS(matrix_naive_base_64_t)
RCPP_EXPOSED_WRAP(matrix_cov_dense_64F_t)
RCPP_EXPOSED_WRAP(matrix_cov_lazy_64F_t)
RCPP_EXPOSED_WRAP(matrix_naive_concatenate_64_t)
RCPP_EXPOSED_WRAP(matrix_naive_dense_64F_t)
RCPP_EXPOSED_WRAP(matrix_naive_kronecker_eye_64_t)
RCPP_EXPOSED_WRAP(matrix_naive_kronecker_eye_dense_64F_t)
RCPP_EXPOSED_WRAP(matrix_naive_snp_unphased_64_t)
RCPP_EXPOSED_WRAP(matrix_naive_snp_phased_ancestry_64_t)

RCPP_MODULE(adelie_core_matrix)
{
    /* base matrices */
    Rcpp::class_<matrix_cov_base_64_t>("MatrixCovBase64")
        ;
    Rcpp::class_<matrix_naive_base_64_t>("MatrixNaiveBase64")
        .method("rows", &matrix_naive_base_64_t::rows)
        .method("cols", &matrix_naive_base_64_t::cols)
        .method("mul", &mul)
        ;

    /* cov matrices */
    Rcpp::class_<matrix_cov_dense_64F_t>("MatrixCovDense64F")
        .derives<matrix_cov_base_64_t>("MatrixCovBase64")
        ;
    Rcpp::class_<matrix_cov_lazy_64F_t>("MatrixCovLazy64F")
        .derives<matrix_cov_base_64_t>("MatrixCovBase64")
        ;

    /* naive matrices */
    Rcpp::class_<matrix_naive_concatenate_64_t>("MatrixNaiveConcatenate64")
        .derives<matrix_naive_base_64_t>("MatrixNaiveBase64")
        ;
    Rcpp::class_<matrix_naive_dense_64F_t>("MatrixNaiveDense64F")
        .derives<matrix_naive_base_64_t>("MatrixNaiveBase64")
        ;
    Rcpp::class_<matrix_naive_kronecker_eye_64_t>("MatrixNaiveKroneckerEye64")
        .derives<matrix_naive_base_64_t>("MatrixNaiveBase64")
        ;
    Rcpp::class_<matrix_naive_kronecker_eye_dense_64F_t>("MatrixNaiveKroneckerEyeDense64")
        .derives<matrix_naive_base_64_t>("MatrixNaiveBase64")
        ;
    Rcpp::class_<matrix_naive_snp_unphased_64_t>("MatrixNaiveSNPUnphased64")
        .derives<matrix_naive_base_64_t>("MatrixNaiveBase64")
        ;
    Rcpp::class_<matrix_naive_snp_phased_ancestry_64_t>("MatrixNaiveSNPPhasedAncestry64")
        .derives<matrix_naive_base_64_t>("MatrixNaiveBase64")
        ;

    /* factory functions */
    Rcpp::function(
        "make_matrix_cov_dense_64F", 
        &make_matrix_cov_dense_64F
    );
    Rcpp::function(
        "make_matrix_cov_lazy_64F", 
        &make_matrix_cov_lazy_64F
    );

    Rcpp::function(
        "make_matrix_naive_concatenate_64", 
        &make_matrix_naive_concatenate_64
    );
    Rcpp::function(
        "make_matrix_naive_dense_64F", 
        &make_matrix_naive_dense_64F
    );
    Rcpp::function(
        "make_matrix_naive_kronecker_eye_64", 
        &make_matrix_naive_kronecker_eye_64
    );
    Rcpp::function(
        "make_matrix_naive_kronecker_eye_dense_64F", 
        &make_matrix_naive_kronecker_eye_dense_64F
    );
    Rcpp::function(
        "make_matrix_naive_snp_unphased_64", 
        &make_matrix_naive_snp_unphased_64
    );
    Rcpp::function(
        "make_matrix_naive_snp_phased_ancestry_64", 
        &make_matrix_naive_snp_phased_ancestry_64
    );
}