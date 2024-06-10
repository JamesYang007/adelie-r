#include "rcpp_matrix.h"

using value_t = double;
using index_t = int;
using vec_value_t = ad::util::colvec_type<value_t>;
using vec_index_t = ad::util::colvec_type<index_t>;
using mat_index_t = ad::util::colmat_type<index_t>;
using dense_64F_t = ad::util::colmat_type<value_t>;

/* Factory functions */

auto make_r_matrix_cov_block_diag_64(
    Rcpp::List mat_list_r,
    size_t n_threads
)
{
    std::vector<matrix_cov_base_64_t*> mat_list;
    for (auto obj : mat_list_r) {
        mat_list.push_back(Rcpp::as<r_matrix_cov_base_64_t*>(obj)->ptr.get());
    }
    return r_matrix_cov_block_diag_64_t(mat_list, n_threads);
}

auto make_r_matrix_cov_dense_64F(
    const Eigen::Map<dense_64F_t>& mat,
    size_t n_threads
)
{
    return r_matrix_cov_dense_64F_t(mat, n_threads);
}

auto make_r_matrix_cov_lazy_cov_64F(
    const Eigen::Map<dense_64F_t>& mat,
    size_t n_threads
)
{
    return r_matrix_cov_lazy_cov_64F_t(mat, n_threads);
}

auto make_r_matrix_cov_sparse_64F(
    size_t rows,
    size_t cols,
    size_t nnz,
    const Eigen::Map<vec_index_t>& outer,
    const Eigen::Map<vec_index_t>& inner,
    const Eigen::Map<vec_value_t>& value,
    size_t n_threads
)
{
    return r_matrix_cov_sparse_64F_t(rows, cols, nnz, outer, inner, value, n_threads);
}

auto make_r_matrix_cov_s4_64(
    Rcpp::S4 mat
)
{
    return r_matrix_cov_s4_64_t(mat);
}

auto make_r_matrix_naive_cconcatenate_64(
    Rcpp::List mat_list_r
)
{
    std::vector<matrix_naive_base_64_t*> mat_list;
    for (auto obj : mat_list_r) {
        mat_list.push_back(Rcpp::as<r_matrix_naive_base_64_t*>(obj)->ptr.get());
    }
    return r_matrix_naive_cconcatenate_64_t(mat_list);
}

auto make_r_matrix_naive_rconcatenate_64(
    Rcpp::List mat_list_r
)
{
    std::vector<matrix_naive_base_64_t*> mat_list;
    for (auto obj : mat_list_r) {
        mat_list.push_back(Rcpp::as<r_matrix_naive_base_64_t*>(obj)->ptr.get());
    }
    return r_matrix_naive_rconcatenate_64_t(mat_list);
}

auto make_r_matrix_naive_dense_64F(
    const Eigen::Map<dense_64F_t>& dense,
    size_t n_threads
)
{
    return r_matrix_naive_dense_64F_t(dense, n_threads);
}

auto make_r_matrix_naive_interaction_dense_64F(
    const Eigen::Map<dense_64F_t>& dense,
    const Eigen::Map<mat_index_t>& pairsT,
    const Eigen::Map<vec_index_t>& levels,
    size_t n_threads
)
{
    using rowarr_index_t = typename ad::util::rowarr_type<index_t>;
    Eigen::Map<const rowarr_index_t> pairs(pairsT.data(), pairsT.cols(), pairsT.rows());
    return r_matrix_naive_interaction_dense_64F_t(dense, pairs, levels, n_threads);
}

auto make_r_matrix_naive_kronecker_eye_64(
    r_matrix_naive_base_64_t& mat,
    size_t K,
    size_t n_threads
)
{
    return r_matrix_naive_kronecker_eye_64_t(*mat.ptr, K, n_threads);
}

auto make_r_matrix_naive_kronecker_eye_dense_64F(
    const Eigen::Map<dense_64F_t>& mat,
    size_t K,
    size_t n_threads
)
{
    return r_matrix_naive_kronecker_eye_dense_64F_t(mat, K, n_threads);
}

auto make_r_matrix_naive_one_hot_dense_64F(
    const Eigen::Map<dense_64F_t>& mat,
    const Eigen::Map<vec_index_t>& levels,
    size_t n_threads
)
{
    return r_matrix_naive_one_hot_dense_64F_t(mat, levels, n_threads);
}

auto make_r_matrix_naive_snp_unphased_64(
    const r_io_snp_unphased_t& io,
    size_t n_threads
)
{
    return r_matrix_naive_snp_unphased_64_t(io, n_threads);
}

auto make_r_matrix_naive_snp_phased_ancestry_64(
    const r_io_snp_phased_ancestry_t& io,
    size_t n_threads
)
{
    return r_matrix_naive_snp_phased_ancestry_64_t(io, n_threads);
}

auto make_r_matrix_naive_sparse_64F(
    size_t rows,
    size_t cols,
    size_t nnz,
    const Eigen::Map<vec_index_t>& outer,
    const Eigen::Map<vec_index_t>& inner,
    const Eigen::Map<vec_value_t>& value,
    size_t n_threads
)
{
    return r_matrix_naive_sparse_64F_t(rows, cols, nnz, outer, inner, value, n_threads);
}

auto make_r_matrix_naive_standardize_64(
    r_matrix_naive_base_64_t& mat,
    const Eigen::Map<vec_value_t>& centers,
    const Eigen::Map<vec_value_t>& scales,
    size_t n_threads
)
{
    return r_matrix_naive_standardize_64_t(*mat.ptr, centers, scales, n_threads);
}

auto make_r_matrix_naive_csubset_64(
    r_matrix_naive_base_64_t& mat,
    const Eigen::Map<vec_index_t>& subset,
    size_t n_threads
)
{
    return r_matrix_naive_csubset_64_t(*mat.ptr, subset, n_threads);
}

auto make_r_matrix_naive_rsubset_64(
    r_matrix_naive_base_64_t& mat,
    const Eigen::Map<vec_index_t>& subset,
    size_t n_threads
)
{
    return r_matrix_naive_rsubset_64_t(*mat.ptr, subset, n_threads);
}

auto make_r_matrix_naive_s4_64(
    Rcpp::S4 mat
)
{
    return r_matrix_naive_s4_64_t(mat);
}

RCPP_MODULE(adelie_core_matrix)
{
    /* base matrices */
    Rcpp::class_<r_matrix_cov_base_64_t>("RMatrixCovBase64")
        ;
    Rcpp::class_<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        .property("rows", &r_matrix_naive_base_64_t::rows)
        .property("cols", &r_matrix_naive_base_64_t::cols)
        .method("mul", &r_matrix_naive_base_64_t::mul)
        .method("cov", &r_matrix_naive_base_64_t::cov)
        .method("sp_btmul", &r_matrix_naive_base_64_t::sp_btmul)
        ;

    /* cov matrices */
    Rcpp::class_<r_matrix_cov_block_diag_64_t>("RMatrixCovBlockDiag64")
        .derives<r_matrix_cov_base_64_t>("RMatrixCovBase64")
        ;
    Rcpp::class_<r_matrix_cov_dense_64F_t>("RMatrixCovDense64F")
        .derives<r_matrix_cov_base_64_t>("RMatrixCovBase64")
        ;
    Rcpp::class_<r_matrix_cov_lazy_cov_64F_t>("RMatrixCovLazyCov64F")
        .derives<r_matrix_cov_base_64_t>("RMatrixCovBase64")
        ;
    Rcpp::class_<r_matrix_cov_sparse_64F_t>("RMatrixCovSparse64F")
        .derives<r_matrix_cov_base_64_t>("RMatrixCovBase64")
        ;
    Rcpp::class_<r_matrix_cov_s4_64_t>("RMatrixCovS464")
        .derives<r_matrix_cov_base_64_t>("RMatrixCovBase64")
        ;

    /* naive matrices */
    Rcpp::class_<r_matrix_naive_cconcatenate_64_t>("RMatrixNaiveCConcatenate64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_rconcatenate_64_t>("RMatrixNaiveRConcatenate64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_dense_64F_t>("RMatrixNaiveDense64F")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_interaction_dense_64F_t>("RMatrixNaiveInteractionDense64F")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        .property("groups", &r_matrix_naive_interaction_dense_64F_t::groups)
        .property("group_sizes", &r_matrix_naive_interaction_dense_64F_t::group_sizes)
        ;
    Rcpp::class_<r_matrix_naive_kronecker_eye_64_t>("RMatrixNaiveKroneckerEye64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_kronecker_eye_dense_64F_t>("RMatrixNaiveKroneckerEyeDense64F")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_one_hot_dense_64F_t>("RMatrixNaiveOneHotDense64F")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        .property("groups", &r_matrix_naive_one_hot_dense_64F_t::groups)
        .property("group_sizes", &r_matrix_naive_one_hot_dense_64F_t::group_sizes)
        ;
    Rcpp::class_<r_matrix_naive_snp_phased_ancestry_64_t>("RMatrixNaiveSNPPhasedAncestry64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_snp_unphased_64_t>("RMatrixNaiveSNPUnphased64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_sparse_64F_t>("RMatrixNaiveSparse64F")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_standardize_64_t>("RMatrixNaiveStandardize64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_csubset_64_t>("RMatrixNaiveCSubset64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_rsubset_64_t>("RMatrixNaiveRSubset64")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;
    Rcpp::class_<r_matrix_naive_s4_64_t>("RMatrixNaiveS464")
        .derives<r_matrix_naive_base_64_t>("RMatrixNaiveBase64")
        ;

    /* factory functions */
    Rcpp::function(
        "make_r_matrix_cov_block_diag_64",
        &make_r_matrix_cov_block_diag_64
    );
    Rcpp::function(
        "make_r_matrix_cov_dense_64F", 
        &make_r_matrix_cov_dense_64F
    );
    Rcpp::function(
        "make_r_matrix_cov_lazy_cov_64F", 
        &make_r_matrix_cov_lazy_cov_64F
    );
    Rcpp::function(
        "make_r_matrix_cov_sparse_64F",
        &make_r_matrix_cov_sparse_64F
    );
    Rcpp::function(
        "make_r_matrix_cov_s4_64",
        &make_r_matrix_cov_s4_64
    );

    Rcpp::function(
        "make_r_matrix_naive_cconcatenate_64", 
        &make_r_matrix_naive_cconcatenate_64
    );
    Rcpp::function(
        "make_r_matrix_naive_rconcatenate_64", 
        &make_r_matrix_naive_rconcatenate_64
    );
    Rcpp::function(
        "make_r_matrix_naive_dense_64F", 
        &make_r_matrix_naive_dense_64F
    );
    Rcpp::function(
        "make_r_matrix_naive_interaction_dense_64F", 
        &make_r_matrix_naive_interaction_dense_64F
    );
    Rcpp::function(
        "make_r_matrix_naive_kronecker_eye_64", 
        &make_r_matrix_naive_kronecker_eye_64
    );
    Rcpp::function(
        "make_r_matrix_naive_kronecker_eye_dense_64F", 
        &make_r_matrix_naive_kronecker_eye_dense_64F
    );
    Rcpp::function(
        "make_r_matrix_naive_one_hot_dense_64F", 
        &make_r_matrix_naive_one_hot_dense_64F
    );
    Rcpp::function(
        "make_r_matrix_naive_snp_phased_ancestry_64", 
        &make_r_matrix_naive_snp_phased_ancestry_64
    );
    Rcpp::function(
        "make_r_matrix_naive_snp_unphased_64", 
        &make_r_matrix_naive_snp_unphased_64
    );
    Rcpp::function(
        "make_r_matrix_naive_sparse_64F",
        &make_r_matrix_naive_sparse_64F
    );
    Rcpp::function(
        "make_r_matrix_naive_standardize_64",
        &make_r_matrix_naive_standardize_64
    );
    Rcpp::function(
        "make_r_matrix_naive_csubset_64",
        &make_r_matrix_naive_csubset_64
    );
    Rcpp::function(
        "make_r_matrix_naive_rsubset_64",
        &make_r_matrix_naive_rsubset_64
    );
    Rcpp::function(
        "make_r_matrix_naive_s4_64",
        &make_r_matrix_naive_s4_64
    );
}