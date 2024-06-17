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

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixCovS4: public MatrixCovBase<ValueType, IndexType>
{
    Rcpp::S4 _mat;

public:
    using base_t = MatrixCovBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using colvec_value_t = util::colvec_type<value_t>;
    using colvec_index_t = util::colvec_type<index_t>;

    explicit MatrixCovS4(Rcpp::S4 mat): _mat(mat) {}

    void bmul(
        const Eigen::Ref<const vec_index_t>& subset,
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const Eigen::Map<colvec_index_t> subset_r(const_cast<index_t*>(subset.data()), subset.size());
        const Eigen::Map<colvec_index_t> indices_r(const_cast<index_t*>(indices.data()), indices.size());
        const Eigen::Map<colvec_value_t> values_r(const_cast<value_t*>(values.data()), values.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(bmul, _mat, subset_r, indices_r, values_r)
        );
    }

    void mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const Eigen::Map<colvec_index_t> indices_r(const_cast<index_t*>(indices.data()), indices.size());
        const Eigen::Map<colvec_value_t> values_r(const_cast<value_t*>(values.data()), values.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(bmul, _mat, indices_r, values_r)
        );
    }

    void to_dense(
        int i, int p, 
        Eigen::Ref<colmat_value_t> out
    ) override
    {
        out = Rcpp::as<Eigen::Map<colmat_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(to_dense, _mat, i, p)
        );
    }
    
    int cols() const override 
    {
        Rcpp::IntegerVector out_r = ADELIE_CORE_S4_PURE_OVERRIDE(cols, _mat,);
        return out_r[0]; 
    }
};

template <class ValueType, class IndexType=Eigen::Index>
class MatrixNaiveS4: public MatrixNaiveBase<ValueType, IndexType>
{
    Rcpp::S4 _mat;

public:
    using base_t = MatrixNaiveBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using typename base_t::rowmat_value_t;
    using typename base_t::sp_mat_value_t;
    using colvec_value_t = util::colvec_type<value_t>;

    explicit MatrixNaiveS4(Rcpp::S4 mat): _mat(mat) {}

    value_t cmul(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        Rcpp::NumericVector out_r = ADELIE_CORE_S4_PURE_OVERRIDE(cmul, _mat, j, v_r, weights_r);
        return out_r[0];
    }

    void ctmul(
        int j, 
        value_t v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out += Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(ctmul, _mat, j, v)
        );
    }

    void bmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    { 
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(bmul, _mat, j, q, v_r, weights_r)
        );
    }

    void btmul(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        out += Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(btmul, _mat, j, q, v_r)
        );
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(mul, _mat, v_r, weights_r)
        );
    }

    void cov(
        int j, int q,
        const Eigen::Ref<const vec_value_t>& sqrt_weights,
        Eigen::Ref<colmat_value_t> out,
        Eigen::Ref<colmat_value_t> 
    ) override
    {
        const Eigen::Map<colvec_value_t> sqrt_weights_r(const_cast<value_t*>(sqrt_weights.data()), sqrt_weights.size());
        out = Rcpp::as<Eigen::Map<colmat_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(cov, _mat, j, q, sqrt_weights_r)
        );
    }

    int rows() const override
    {
        Rcpp::IntegerVector out_r = ADELIE_CORE_S4_PURE_OVERRIDE(rows, _mat,);
        return out_r[0];
    }
    
    int cols() const override
    {
        Rcpp::IntegerVector out_r = ADELIE_CORE_S4_PURE_OVERRIDE(cols, _mat,);
        return out_r[0];
    }

    /* Non-speed critical routines */

    void sp_btmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) override
    {
        out = Rcpp::as<colmat_value_t>(
            ADELIE_CORE_S4_PURE_OVERRIDE(sp_btmul, _mat, v)
        ).transpose();
    }
};

} // namespace matrix
} // namespace adelie_core

using matrix_cov_base_64_t = ad::matrix::MatrixCovBase<double, int>;
using matrix_cov_block_diag_64_t = ad::matrix::MatrixCovBlockDiag<double, int>;
using matrix_cov_dense_64F_t = ad::matrix::MatrixCovDense<ad::util::colmat_type<double>, int>;
using matrix_cov_lazy_cov_64F_t = ad::matrix::MatrixCovLazyCov<ad::util::colmat_type<double>, int>;
using matrix_cov_sparse_64F_t = ad::matrix::MatrixCovSparse<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, int>;
using matrix_cov_s4_64_t = ad::matrix::MatrixCovS4<double, int>;

using matrix_naive_base_64_t = ad::matrix::MatrixNaiveBase<double, int>;
using matrix_naive_cconcatenate_64_t = ad::matrix::MatrixNaiveCConcatenate<double, int>;
using matrix_naive_rconcatenate_64_t = ad::matrix::MatrixNaiveRConcatenate<double, int>;
using matrix_naive_dense_64F_t = ad::matrix::MatrixNaiveDense<ad::util::colmat_type<double>, int>;
using matrix_naive_interaction_dense_64F_t = ad::matrix::MatrixNaiveInteractionDense<ad::util::colmat_type<double>, int>;
using matrix_naive_kronecker_eye_64_t = ad::matrix::MatrixNaiveKroneckerEye<double, int>;
using matrix_naive_kronecker_eye_dense_64F_t = ad::matrix::MatrixNaiveKroneckerEyeDense<ad::util::colmat_type<double>, int>;
using matrix_naive_one_hot_dense_64F_t = ad::matrix::MatrixNaiveOneHotDense<ad::util::colmat_type<double>, int>;
using matrix_naive_snp_phased_ancestry_64_t = ad::matrix::MatrixNaiveSNPPhasedAncestry<double, std::shared_ptr<char>, int>;
using matrix_naive_snp_unphased_64_t = ad::matrix::MatrixNaiveSNPUnphased<double, std::shared_ptr<char>, int>;
using matrix_naive_sparse_64F_t = ad::matrix::MatrixNaiveSparse<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, int>;
using matrix_naive_standardize_64_t = ad::matrix::MatrixNaiveStandardize<double, int>;
using matrix_naive_csubset_64_t = ad::matrix::MatrixNaiveCSubset<double, int>;
using matrix_naive_rsubset_64_t = ad::matrix::MatrixNaiveRSubset<double, int>;
using matrix_naive_s4_64_t = ad::matrix::MatrixNaiveS4<double, int>;

class RMatrixCovBase64: public pimpl<matrix_cov_base_64_t>
{
    using base_t = pimpl<matrix_cov_base_64_t>;
public:
    using value_t = double;
    using index_t = int;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using vec_index_t = ad::util::colvec_type<index_t>;

    using base_t::base_t;

    int cols() const { return ptr->cols(); }

    vec_value_t mul(
        const Eigen::Map<vec_index_t>& indices,
        const Eigen::Map<vec_value_t>& values
    ) 
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(mul, indices, values, out); }();
        return out;
    }
};

class RMatrixNaiveBase64: public pimpl<matrix_naive_base_64_t>
{
    using base_t = pimpl<matrix_naive_base_64_t>;
public:
    using value_t = double;
    using index_t = int;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using dense_64F_t = ad::util::colmat_type<value_t>;
    using sp_mat_value_t = Eigen::SparseMatrix<value_t, Eigen::RowMajor>;

    using base_t::base_t;

    int rows() const { return ptr->rows(); }
    int cols() const { return ptr->cols(); }

    vec_value_t mul(
        const Eigen::Map<vec_value_t>& v, 
        const Eigen::Map<vec_value_t>& weights
    )
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(mul, v, weights, out); }();
        return out;
    }

    dense_64F_t cov(
        int j, int q,
        const Eigen::Map<vec_value_t>& sqrt_weights
    )
    {
        dense_64F_t out(q, q);
        dense_64F_t buffer(rows(), q);
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cov, j, q, sqrt_weights, out, buffer); }();
        return out;
    }

    dense_64F_t sp_btmul(
        const sp_mat_value_t& v
    )
    {
        dense_64F_t outT(rows(), v.rows());
        using rowmat_value_t = ad::util::rowmat_type<value_t>;
        Eigen::Map<rowmat_value_t> out(outT.data(), outT.cols(), outT.rows());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(sp_btmul, v, out); }();
        return outT;
    }
};

ADELIE_CORE_PIMPL_DERIVED(RMatrixCovBlockDiag64, RMatrixCovBase64, matrix_cov_block_diag_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovDense64F, RMatrixCovBase64, matrix_cov_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovLazyCov64F, RMatrixCovBase64, matrix_cov_lazy_cov_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovSparse64F, RMatrixCovBase64, matrix_cov_sparse_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovS464, RMatrixCovBase64, matrix_cov_s4_64_t,)

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
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveS464, RMatrixNaiveBase64, matrix_naive_s4_64_t,)

RCPP_EXPOSED_CLASS(RMatrixCovBase64)
RCPP_EXPOSED_CLASS(RMatrixCovBlockDiag64)
RCPP_EXPOSED_CLASS(RMatrixCovDense64F)
RCPP_EXPOSED_CLASS(RMatrixCovLazyCov64F)
RCPP_EXPOSED_CLASS(RMatrixCovSparse64F)
RCPP_EXPOSED_CLASS(RMatrixCovS464)

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
RCPP_EXPOSED_CLASS(RMatrixNaiveS464)

using r_matrix_cov_base_64_t = RMatrixCovBase64;
using r_matrix_cov_block_diag_64_t = RMatrixCovBlockDiag64;
using r_matrix_cov_dense_64F_t = RMatrixCovDense64F;
using r_matrix_cov_lazy_cov_64F_t = RMatrixCovLazyCov64F;
using r_matrix_cov_sparse_64F_t = RMatrixCovSparse64F;
using r_matrix_cov_s4_64_t = RMatrixCovS464;

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
using r_matrix_naive_s4_64_t = RMatrixNaiveS464;