#pragma once
#include "decl.h"
#include "utils.h"
#include "rcpp_io.h"
#include <adelie_core/io/io_snp_unphased.hpp>
#include <adelie_core/io/io_snp_phased_ancestry.hpp>
#include <adelie_core/matrix/matrix_cov_base.hpp>
#include <adelie_core/matrix/matrix_cov_block_diag.hpp>
#include <adelie_core/matrix/matrix_cov_dense.hpp>
#include <adelie_core/matrix/matrix_cov_lazy_cov.hpp>
#include <adelie_core/matrix/matrix_cov_sparse.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/matrix/matrix_naive_dense.hpp>
#include <adelie_core/matrix/matrix_naive_concatenate.hpp>
#include <adelie_core/matrix/matrix_naive_interaction.hpp>
#include <adelie_core/matrix/matrix_naive_kronecker_eye.hpp>
#include <adelie_core/matrix/matrix_naive_one_hot.hpp>
#include <adelie_core/matrix/matrix_naive_snp_phased_ancestry.hpp>
#include <adelie_core/matrix/matrix_naive_snp_unphased.hpp>
#include <adelie_core/matrix/matrix_naive_sparse.hpp>
#include <adelie_core/matrix/matrix_naive_standardize.hpp>
#include <adelie_core/matrix/matrix_naive_subset.hpp>

using matrix_cov_base_64_t = ad::matrix::MatrixCovBase<double>;
using matrix_cov_block_diag_64_t = ad::matrix::MatrixCovBlockDiag<double>;
using matrix_cov_dense_64F_t = ad::matrix::MatrixCovDense<ad::util::colmat_type<double>>;
using matrix_cov_lazy_cov_64F_t = ad::matrix::MatrixCovLazyCov<ad::util::colmat_type<double>>;
using matrix_cov_sparse_64F_t = ad::matrix::MatrixCovSparse<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>;

using matrix_naive_base_64_t = ad::matrix::MatrixNaiveBase<double>;
using matrix_naive_cconcatenate_64_t = ad::matrix::MatrixNaiveCConcatenate<double>;
using matrix_naive_rconcatenate_64_t = ad::matrix::MatrixNaiveRConcatenate<double>;
using matrix_naive_dense_64F_t = ad::matrix::MatrixNaiveDense<ad::util::colmat_type<double>>;
using matrix_naive_interaction_dense_64F_t = ad::matrix::MatrixNaiveInteractionDense<ad::util::colmat_type<double>>;
using matrix_naive_kronecker_eye_64_t = ad::matrix::MatrixNaiveKroneckerEye<double>;
using matrix_naive_kronecker_eye_dense_64F_t = ad::matrix::MatrixNaiveKroneckerEyeDense<ad::util::colmat_type<double>>;
using matrix_naive_one_hot_dense_64F_t = ad::matrix::MatrixNaiveOneHotDense<ad::util::colmat_type<double>>;
using matrix_naive_snp_phased_ancestry_64_t = ad::matrix::MatrixNaiveSNPPhasedAncestry<double, std::shared_ptr<char>>;
using matrix_naive_snp_unphased_64_t = ad::matrix::MatrixNaiveSNPUnphased<double, std::shared_ptr<char>>;
using matrix_naive_sparse_64F_t = ad::matrix::MatrixNaiveSparse<Eigen::SparseMatrix<double, Eigen::ColMajor, int>>;
using matrix_naive_standardize_64_t = ad::matrix::MatrixNaiveStandardize<double>;
using matrix_naive_csubset_64_t = ad::matrix::MatrixNaiveCSubset<double>;
using matrix_naive_rsubset_64_t = ad::matrix::MatrixNaiveRSubset<double>;

class RMatrixCovBase64: public pimpl<matrix_cov_base_64_t>
{
    using base_t = pimpl<matrix_cov_base_64_t>;
public:
    using base_t::base_t;

    virtual ~RMatrixCovBase64() {}
};

class RMatrixNaiveBase64: public pimpl<matrix_naive_base_64_t>
{
    using base_t = pimpl<matrix_naive_base_64_t>;
public:
    using value_t = double;
    using index_t = int;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using dense_64F_t = ad::util::colmat_type<value_t>;
    using sp_mat_value_t = Eigen::SparseMatrix<value_t, Eigen::RowMajor, index_t>;

    using base_t::base_t;

    virtual ~RMatrixNaiveBase64() {}

    int rows() const { return ptr->rows(); }
    int cols() const { return ptr->cols(); }

    virtual void mul(
        const Eigen::Map<vec_value_t>& v, 
        const Eigen::Map<vec_value_t>& weights,
        Eigen::Map<vec_value_t> out
    )
    {
        ADELIE_CORE_OVERRIDE(mul, v, weights, out);
    }

    virtual void cov(
        int j, int q,
        const Eigen::Map<vec_value_t>& sqrt_weights,
        Eigen::Map<dense_64F_t> out,
        Eigen::Map<dense_64F_t> buffer
    )
    {
        ADELIE_CORE_OVERRIDE(cov, j, q, sqrt_weights, out, buffer);
    }

    virtual void sp_btmul(
        const sp_mat_value_t& v,
        Eigen::Map<dense_64F_t> outT
    )
    {
        using rowmat_value_t = ad::util::rowmat_type<value_t>;
        Eigen::Map<rowmat_value_t> out(outT.data(), outT.cols(), outT.rows());
        ADELIE_CORE_OVERRIDE(sp_btmul, v, out);
    }
};

ADELIE_CORE_PIMPL_DERIVED(RMatrixCovBlockDiag64, RMatrixCovBase64, matrix_cov_block_diag_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovDense64F, RMatrixCovBase64, matrix_cov_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovLazyCov64F, RMatrixCovBase64, matrix_cov_lazy_cov_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovSparse64F, RMatrixCovBase64, matrix_cov_sparse_64F_t,)

ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveCConcatenate64, RMatrixNaiveBase64, matrix_naive_cconcatenate_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveRConcatenate64, RMatrixNaiveBase64, matrix_naive_rconcatenate_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveDense64F, RMatrixNaiveBase64, matrix_naive_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveInteractionDense64F, RMatrixNaiveBase64, matrix_naive_interaction_dense_64F_t,
    auto groups() const { return dynamic_cast<matrix_naive_interaction_dense_64F_t&>(*ptr).groups(); }
    auto group_sizes() const { return dynamic_cast<matrix_naive_interaction_dense_64F_t&>(*ptr).group_sizes(); }
)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveKroneckerEye64, RMatrixNaiveBase64, matrix_naive_kronecker_eye_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveKroneckerEyeDense64F, RMatrixNaiveBase64, matrix_naive_kronecker_eye_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveOneHotDense64F, RMatrixNaiveBase64, matrix_naive_one_hot_dense_64F_t,
    auto groups() const { return dynamic_cast<matrix_naive_one_hot_dense_64F_t&>(*ptr).groups(); }
    auto group_sizes() const { return dynamic_cast<matrix_naive_one_hot_dense_64F_t&>(*ptr).group_sizes(); }
)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveSNPPhasedAncestry64, RMatrixNaiveBase64, matrix_naive_snp_phased_ancestry_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveSNPUnphased64, RMatrixNaiveBase64, matrix_naive_snp_unphased_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveSparse64F, RMatrixNaiveBase64, matrix_naive_sparse_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveStandardize64, RMatrixNaiveBase64, matrix_naive_standardize_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveCSubset64, RMatrixNaiveBase64, matrix_naive_csubset_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveRSubset64, RMatrixNaiveBase64, matrix_naive_rsubset_64_t,)

RCPP_EXPOSED_CLASS(RMatrixCovBase64)
RCPP_EXPOSED_CLASS(RMatrixCovBlockDiag64)
RCPP_EXPOSED_CLASS(RMatrixCovDense64F)
RCPP_EXPOSED_CLASS(RMatrixCovLazyCov64F)
RCPP_EXPOSED_CLASS(RMatrixCovSparse64F)

RCPP_EXPOSED_CLASS(RMatrixNaiveBase64)
RCPP_EXPOSED_CLASS(RMatrixNaiveCConcatenate64)
RCPP_EXPOSED_CLASS(RMatrixNaiveRConcatenate64)
RCPP_EXPOSED_CLASS(RMatrixNaiveDense64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveInteractionDense64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveKroneckerEye64)
RCPP_EXPOSED_CLASS(RMatrixNaiveKroneckerEyeDense64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveOneHotDense64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveSNPPhasedAncestry64)
RCPP_EXPOSED_CLASS(RMatrixNaiveSNPUnphased64)
RCPP_EXPOSED_CLASS(RMatrixNaiveSparse64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveStandardize64)
RCPP_EXPOSED_CLASS(RMatrixNaiveCSubset64)
RCPP_EXPOSED_CLASS(RMatrixNaiveRSubset64)

using r_matrix_cov_base_64_t = RMatrixCovBase64;
using r_matrix_cov_block_diag_64_t = RMatrixCovBlockDiag64;
using r_matrix_cov_dense_64F_t = RMatrixCovDense64F;
using r_matrix_cov_lazy_cov_64F_t = RMatrixCovLazyCov64F;
using r_matrix_cov_sparse_64F_t = RMatrixCovSparse64F;

using r_matrix_naive_base_64_t = RMatrixNaiveBase64;
using r_matrix_naive_cconcatenate_64_t = RMatrixNaiveCConcatenate64;
using r_matrix_naive_rconcatenate_64_t = RMatrixNaiveRConcatenate64;
using r_matrix_naive_dense_64F_t = RMatrixNaiveDense64F;
using r_matrix_naive_interaction_dense_64F_t = RMatrixNaiveInteractionDense64F;
using r_matrix_naive_kronecker_eye_64_t = RMatrixNaiveKroneckerEye64;
using r_matrix_naive_kronecker_eye_dense_64F_t = RMatrixNaiveKroneckerEyeDense64F;
using r_matrix_naive_one_hot_dense_64F_t = RMatrixNaiveOneHotDense64F;
using r_matrix_naive_snp_phased_ancestry_64_t = RMatrixNaiveSNPPhasedAncestry64;
using r_matrix_naive_snp_unphased_64_t = RMatrixNaiveSNPUnphased64;
using r_matrix_naive_sparse_64F_t = RMatrixNaiveSparse64F;
using r_matrix_naive_standardize_64_t = RMatrixNaiveStandardize64;
using r_matrix_naive_csubset_64_t = RMatrixNaiveCSubset64;
using r_matrix_naive_rsubset_64_t = RMatrixNaiveRSubset64;
