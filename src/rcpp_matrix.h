#pragma once
#include "decl.h"
#include "utils.h"
#include "rcpp_io.h"
#include <adelie_core/matrix/matrix_constraint_base.ipp>
#include <adelie_core/matrix/matrix_constraint_dense.ipp>
#include <adelie_core/matrix/matrix_constraint_sparse.ipp>
#include <adelie_core/matrix/matrix_cov_base.ipp>
#include <adelie_core/matrix/matrix_cov_block_diag.ipp>
#include <adelie_core/matrix/matrix_cov_dense.ipp>
#include <adelie_core/matrix/matrix_cov_lazy_cov.ipp>
#include <adelie_core/matrix/matrix_cov_sparse.ipp>
#include <adelie_core/matrix/matrix_naive_base.ipp>
#include <adelie_core/matrix/matrix_naive_block_diag.ipp>
#include <adelie_core/matrix/matrix_naive_concatenate.ipp>
#include <adelie_core/matrix/matrix_naive_convex_gated_relu.ipp>
#include <adelie_core/matrix/matrix_naive_convex_relu.ipp>
#include <adelie_core/matrix/matrix_naive_dense.ipp>
#include <adelie_core/matrix/matrix_naive_interaction.ipp>
#include <adelie_core/matrix/matrix_naive_kronecker_eye.ipp>
#include <adelie_core/matrix/matrix_naive_one_hot.ipp>
#include <adelie_core/matrix/matrix_naive_snp_phased_ancestry.ipp>
#include <adelie_core/matrix/matrix_naive_snp_unphased.ipp>
#include <adelie_core/matrix/matrix_naive_sparse.ipp>
#include <adelie_core/matrix/matrix_naive_standardize.ipp>
#include <adelie_core/matrix/matrix_naive_subset.ipp>

namespace adelie_core {
namespace matrix {

template <class ValueType, class IndexType=Eigen::Index>
class MatrixConstraintS4: public MatrixConstraintBase<ValueType, IndexType>
{
    Rcpp::S4 _mat;

public:
    using base_t = MatrixConstraintBase<ValueType, IndexType>;
    using typename base_t::value_t;
    using typename base_t::index_t;
    using typename base_t::vec_value_t;
    using typename base_t::vec_index_t;
    using typename base_t::colmat_value_t;
    using colvec_value_t = util::colvec_type<value_t>;
    using colvec_index_t = util::colvec_type<index_t>;

    explicit MatrixConstraintS4(Rcpp::S4 mat): _mat(mat) {}

    void rmmul(
        int j,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(rmmul, _mat, j, Q)
        );
    }

    void rmmul_safe(
        int j,
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(rmmul_safe, _mat, j, Q)
        );
    }

    value_t rvmul(
        int j,
        const Eigen::Ref<const vec_value_t>& v
    ) override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        Rcpp::NumericVector out_r = ADELIE_CORE_S4_PURE_OVERRIDE(rvmul, _mat, j, v_r);
        return out_r[0];
    }

    value_t rvmul_safe(
        int j,
        const Eigen::Ref<const vec_value_t>& v
    ) const override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        Rcpp::NumericVector out_r = ADELIE_CORE_S4_PURE_OVERRIDE(rvmul_safe, _mat, j, v_r);
        return out_r[0];
    }

    void rvtmul(
        int j,
        value_t v,
        Eigen::Ref<vec_value_t> out
    ) override
    {
        out += Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(rvtmul, _mat, j, v)
        );
    }

    void mul(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(mul, _mat, v)
        );
    }

    void tmul(
        const Eigen::Ref<const vec_value_t>& v,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        out += Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(tmul, _mat, v)
        );
    }

    void cov(
        const Eigen::Ref<const colmat_value_t>& Q,
        Eigen::Ref<colmat_value_t> out
    ) const override
    {
        out = Rcpp::as<Eigen::Map<colmat_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(cov, _mat, Q)
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

    void sp_mul(
        const Eigen::Ref<const vec_index_t>& indices,
        const Eigen::Ref<const vec_value_t>& values,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const Eigen::Map<colvec_index_t> indices_r(const_cast<index_t*>(indices.data()), indices.size());
        const Eigen::Map<colvec_value_t> values_r(const_cast<value_t*>(values.data()), values.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(sp_mul, _mat, indices_r, values_r)
        );
    }
};

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
    ) const override
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
    ) const override
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

    value_t cmul_safe(
        int j, 
        const Eigen::Ref<const vec_value_t>& v,
        const Eigen::Ref<const vec_value_t>& weights
    ) const override
    {
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        Rcpp::NumericVector out_r = ADELIE_CORE_S4_PURE_OVERRIDE(cmul_safe, _mat, j, v_r, weights_r);
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

    void bmul_safe(
        int j, int q, 
        const Eigen::Ref<const vec_value_t>& v, 
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    { 
        const Eigen::Map<colvec_value_t> v_r(const_cast<value_t*>(v.data()), v.size());
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(bmul_safe, _mat, j, q, v_r, weights_r)
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
    ) const override
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
        Eigen::Ref<colmat_value_t> out
    ) const override
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

    void sq_mul(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(sq_mul, _mat, weights_r)
        );
    }

    void sp_tmul(
        const sp_mat_value_t& v,
        Eigen::Ref<rowmat_value_t> out
    ) const override
    {
        out = Rcpp::as<colmat_value_t>(
            ADELIE_CORE_S4_PURE_OVERRIDE(sp_tmul, _mat, v)
        ).transpose();
    }

    void mean(
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(mean, _mat, weights_r)
        );
    }

    void var(
        const Eigen::Ref<const vec_value_t>& centers,
        const Eigen::Ref<const vec_value_t>& weights,
        Eigen::Ref<vec_value_t> out
    ) const override
    {
        const Eigen::Map<colvec_value_t> centers_r(const_cast<value_t*>(centers.data()), centers.size());
        const Eigen::Map<colvec_value_t> weights_r(const_cast<value_t*>(weights.data()), weights.size());
        out = Rcpp::as<Eigen::Map<colvec_value_t>>(
            ADELIE_CORE_S4_PURE_OVERRIDE(var, _mat, centers_r, weights_r)
        );
    }
};

} // namespace matrix
} // namespace adelie_core

using matrix_constraint_base_64_t = ad::matrix::MatrixConstraintBase<double, int>;
using matrix_constraint_dense_64F_t = ad::matrix::MatrixConstraintDense<ad::util::rowmat_type<double>, int>;
using matrix_constraint_sparse_64F_t = ad::matrix::MatrixConstraintSparse<Eigen::SparseMatrix<double, Eigen::RowMajor>, int>;
using matrix_constraint_s4_64_t = ad::matrix::MatrixConstraintS4<double, int>;

using matrix_cov_base_64_t = ad::matrix::MatrixCovBase<double, int>;
using matrix_cov_block_diag_64_t = ad::matrix::MatrixCovBlockDiag<double, int>;
using matrix_cov_dense_64F_t = ad::matrix::MatrixCovDense<ad::util::colmat_type<double>, int>;
using matrix_cov_lazy_cov_64F_t = ad::matrix::MatrixCovLazyCov<ad::util::colmat_type<double>, int>;
using matrix_cov_sparse_64F_t = ad::matrix::MatrixCovSparse<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, int>;
using matrix_cov_s4_64_t = ad::matrix::MatrixCovS4<double, int>;

using matrix_naive_base_64_t = ad::matrix::MatrixNaiveBase<double, int>;
using matrix_naive_block_diag_64_t = ad::matrix::MatrixNaiveBlockDiag<double, int>;
using matrix_naive_cconcatenate_64_t = ad::matrix::MatrixNaiveCConcatenate<double, int>;
using matrix_naive_rconcatenate_64_t = ad::matrix::MatrixNaiveRConcatenate<double, int>;
using matrix_naive_convex_gated_relu_dense_64F_t = ad::matrix::MatrixNaiveConvexGatedReluDense<ad::util::colmat_type<double>, ad::util::colmat_type<int>, int>;
using matrix_naive_convex_gated_relu_sparse_64F_t = ad::matrix::MatrixNaiveConvexGatedReluSparse<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, ad::util::colmat_type<int>, int>;
using matrix_naive_convex_relu_dense_64F_t = ad::matrix::MatrixNaiveConvexReluDense<ad::util::colmat_type<double>, ad::util::colmat_type<int>, int>;
using matrix_naive_convex_relu_sparse_64F_t = ad::matrix::MatrixNaiveConvexReluSparse<Eigen::SparseMatrix<double, Eigen::ColMajor, int>, ad::util::colmat_type<int>, int>;
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

class RMatrixConstraintBase64: public pimpl<matrix_constraint_base_64_t>
{
    using base_t = pimpl<matrix_constraint_base_64_t>;
public:
    using value_t = double;
    using index_t = int;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using vec_index_t = ad::util::colvec_type<index_t>;
    using colmat_value_t = ad::util::colmat_type<value_t>;

    using base_t::base_t;

    vec_value_t rmmul(
        int j,
        const Eigen::Map<colmat_value_t>& Q
    ) 
    {
        vec_value_t out(Q.cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rmmul, j, Q, out); }();
        return out;
    }

    vec_value_t rmmul_safe(
        int j,
        const Eigen::Map<colmat_value_t>& Q
    ) 
    {
        vec_value_t out(Q.cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rmmul_safe, j, Q, out); }();
        return out;
    }

    value_t rvmul(
        int j,
        const Eigen::Map<vec_value_t>& v
    )
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rvmul, j, v); }();
    }

    value_t rvmul_safe(
        int j,
        const Eigen::Map<vec_value_t>& v
    )
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rvmul_safe, j, v); }();
    }

    vec_value_t rvtmul(
        int j,
        value_t v,
        const Eigen::Map<vec_value_t>& out_in
    )
    {
        vec_value_t out = out_in;
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rvtmul, j, v, out); }();
        return out;
    }

    vec_value_t mul(
        const Eigen::Map<vec_value_t>& v
    )
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(mul, v, out); }();
        return out;
    }

    vec_value_t tmul(
        const Eigen::Map<vec_value_t>& v,
        const Eigen::Map<vec_value_t>& out_in
    )
    {
        vec_value_t out = out_in;
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(tmul, v, out); }();
        return out;
    }

    colmat_value_t cov(
        const Eigen::Map<colmat_value_t>& Q
    )
    {
        colmat_value_t out(rows(), rows());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cov, Q, out); }();
        return out;
    }

    int rows() const 
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rows,); }();
    }
    int cols() const 
    { 
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cols,); }();
    }

    vec_value_t sp_mul(
        const Eigen::Map<vec_index_t>& indices,
        const Eigen::Map<vec_value_t>& values
    )
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(sp_mul, indices, values, out); }();
        return out;
    }
};

class RMatrixCovBase64: public pimpl<matrix_cov_base_64_t>
{
    using base_t = pimpl<matrix_cov_base_64_t>;
public:
    using value_t = double;
    using index_t = int;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using vec_index_t = ad::util::colvec_type<index_t>;
    using colmat_value_t = ad::util::colmat_type<value_t>;

    using base_t::base_t;

    vec_value_t bmul(
        const Eigen::Map<vec_index_t>& subset,
        const Eigen::Map<vec_index_t>& indices,
        const Eigen::Map<vec_value_t>& values
    )
    {
        vec_value_t out(subset.size());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(bmul, subset, indices, values, out); }();
        return out;
    }

    vec_value_t mul(
        const Eigen::Map<vec_index_t>& indices,
        const Eigen::Map<vec_value_t>& values
    ) 
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(mul, indices, values, out); }();
        return out;
    }

    colmat_value_t to_dense(
        int i, int p
    )
    {
        colmat_value_t out(p, p);
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(to_dense, i, p, out); }();
        return out;
    }

    int rows() const 
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rows,); }();
    }
    int cols() const 
    { 
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cols,); }();
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

    value_t cmul(
        int j,
        const Eigen::Map<vec_value_t>& v, 
        const Eigen::Map<vec_value_t>& weights
    )
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cmul, j, v, weights); }();
    }

    value_t cmul_safe(
        int j,
        const Eigen::Map<vec_value_t>& v, 
        const Eigen::Map<vec_value_t>& weights
    )
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cmul_safe, j, v, weights); }();
    }

    vec_value_t ctmul(
        int j,
        value_t v,
        Eigen::Map<vec_value_t> out_in
    )
    {
        vec_value_t out = out_in;
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(ctmul, j, v, out); }();
        return out;
    }

    vec_value_t bmul(
        int j, int q,
        const Eigen::Map<vec_value_t>& v, 
        const Eigen::Map<vec_value_t>& weights
    )
    {
        vec_value_t out(q);
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(bmul, j, q, v, weights, out); }();
        return out;
    }

    vec_value_t bmul_safe(
        int j, int q,
        const Eigen::Map<vec_value_t>& v, 
        const Eigen::Map<vec_value_t>& weights
    )
    {
        vec_value_t out(q);
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(bmul_safe, j, q, v, weights, out); }();
        return out;
    }

    vec_value_t btmul(
        int j, int q,
        const Eigen::Map<vec_value_t>& v,
        Eigen::Map<vec_value_t> out_in
    )
    {
        vec_value_t out = out_in;
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(btmul, j, q, v, out); }();
        return out;
    }

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
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cov, j, q, sqrt_weights, out); }();
        return out;
    }

    int rows() const 
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(rows,); }();
    }
    int cols() const 
    { 
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(cols,); }();
    }

    vec_value_t sq_mul(
        const Eigen::Map<vec_value_t>& weights
    )
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(sq_mul, weights, out); }();
        return out;
    }

    dense_64F_t sp_tmul(
        const sp_mat_value_t& v
    )
    {
        dense_64F_t outT(rows(), v.rows());
        using rowmat_value_t = ad::util::rowmat_type<value_t>;
        Eigen::Map<rowmat_value_t> out(outT.data(), outT.cols(), outT.rows());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(sp_tmul, v, out); }();
        return outT;
    }

    vec_value_t mean(
        const Eigen::Map<vec_value_t>& weights
    ) 
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(mean, weights, out); }();
        return out;
    }

    vec_value_t var(
        const Eigen::Map<vec_value_t>& centers,
        const Eigen::Map<vec_value_t>& weights
    ) 
    {
        vec_value_t out(cols());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(var, centers, weights, out); }();
        return out;
    }
};

ADELIE_CORE_PIMPL_DERIVED(RMatrixConstraintDense64F, RMatrixConstraintBase64, matrix_constraint_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixConstraintSparse64F, RMatrixConstraintBase64, matrix_constraint_sparse_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixConstraintS464, RMatrixConstraintBase64, matrix_constraint_s4_64_t,)

ADELIE_CORE_PIMPL_DERIVED(RMatrixCovBlockDiag64, RMatrixCovBase64, matrix_cov_block_diag_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovDense64F, RMatrixCovBase64, matrix_cov_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovLazyCov64F, RMatrixCovBase64, matrix_cov_lazy_cov_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovSparse64F, RMatrixCovBase64, matrix_cov_sparse_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixCovS464, RMatrixCovBase64, matrix_cov_s4_64_t,)

ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveBlockDiag64, RMatrixNaiveBase64, matrix_naive_block_diag_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveCConcatenate64, RMatrixNaiveBase64, matrix_naive_cconcatenate_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveRConcatenate64, RMatrixNaiveBase64, matrix_naive_rconcatenate_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveConvexGatedReluDense64F, RMatrixNaiveBase64, matrix_naive_convex_gated_relu_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveConvexGatedReluSparse64F, RMatrixNaiveBase64, matrix_naive_convex_gated_relu_sparse_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveConvexReluDense64F, RMatrixNaiveBase64, matrix_naive_convex_relu_dense_64F_t,)
ADELIE_CORE_PIMPL_DERIVED(RMatrixNaiveConvexReluSparse64F, RMatrixNaiveBase64, matrix_naive_convex_relu_sparse_64F_t,)
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

RCPP_EXPOSED_CLASS(RMatrixConstraintBase64)
RCPP_EXPOSED_CLASS(RMatrixConstraintDense64F)
RCPP_EXPOSED_CLASS(RMatrixConstraintSparse64F)
RCPP_EXPOSED_CLASS(RMatrixConstraintS464)

RCPP_EXPOSED_CLASS(RMatrixCovBase64)
RCPP_EXPOSED_CLASS(RMatrixCovBlockDiag64)
RCPP_EXPOSED_CLASS(RMatrixCovDense64F)
RCPP_EXPOSED_CLASS(RMatrixCovLazyCov64F)
RCPP_EXPOSED_CLASS(RMatrixCovSparse64F)
RCPP_EXPOSED_CLASS(RMatrixCovS464)

RCPP_EXPOSED_CLASS(RMatrixNaiveBase64)
RCPP_EXPOSED_CLASS(RMatrixNaiveBlockDiag64)
RCPP_EXPOSED_CLASS(RMatrixNaiveCConcatenate64)
RCPP_EXPOSED_CLASS(RMatrixNaiveRConcatenate64)
RCPP_EXPOSED_CLASS(RMatrixNaiveConvexGatedReluDense64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveConvexGatedReluSparse64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveConvexReluDense64F)
RCPP_EXPOSED_CLASS(RMatrixNaiveConvexReluSparse64F)
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

using r_matrix_constraint_base_64_t = RMatrixConstraintBase64;
using r_matrix_constraint_dense_64F_t = RMatrixConstraintDense64F;
using r_matrix_constraint_sparse_64F_t = RMatrixConstraintSparse64F;
using r_matrix_constraint_s4_64_t = RMatrixConstraintS464;

using r_matrix_cov_base_64_t = RMatrixCovBase64;
using r_matrix_cov_block_diag_64_t = RMatrixCovBlockDiag64;
using r_matrix_cov_dense_64F_t = RMatrixCovDense64F;
using r_matrix_cov_lazy_cov_64F_t = RMatrixCovLazyCov64F;
using r_matrix_cov_sparse_64F_t = RMatrixCovSparse64F;
using r_matrix_cov_s4_64_t = RMatrixCovS464;

using r_matrix_naive_base_64_t = RMatrixNaiveBase64;
using r_matrix_naive_block_diag_64_t = RMatrixNaiveBlockDiag64;
using r_matrix_naive_cconcatenate_64_t = RMatrixNaiveCConcatenate64;
using r_matrix_naive_rconcatenate_64_t = RMatrixNaiveRConcatenate64;
using r_matrix_naive_convex_gated_relu_dense_64F_t = RMatrixNaiveConvexGatedReluDense64F;
using r_matrix_naive_convex_gated_relu_sparse_64F_t = RMatrixNaiveConvexGatedReluSparse64F;
using r_matrix_naive_convex_relu_dense_64F_t = RMatrixNaiveConvexReluDense64F;
using r_matrix_naive_convex_relu_sparse_64F_t = RMatrixNaiveConvexReluSparse64F;
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