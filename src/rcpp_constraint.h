#pragma once
#include "decl.h"
#include "utils.h"
#include "rcpp_matrix.h"
#include <adelie_core/constraint/constraint_base.ipp>
#include <adelie_core/constraint/constraint_box.ipp>
#include <adelie_core/constraint/constraint_linear.ipp>
#include <adelie_core/constraint/constraint_one_sided.ipp>

using constraint_base_64_t = ad::constraint::ConstraintBase<double, int>;
using constraint_box_64_t = ad::constraint::ConstraintBox<double, int>;
using constraint_linear_64_t = ad::constraint::ConstraintLinear<matrix_constraint_base_64_t, int>;
using constraint_one_sided_64_t = ad::constraint::ConstraintOneSided<double, int>;

class RConstraintBase64: public pimpl<constraint_base_64_t>
{
    using base_t = pimpl<constraint_base_64_t>;
public:
    using value_t = double;
    using index_t = int;
    using vec_index_t = ad::util::colvec_type<index_t>;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using vec_uint64_t = ad::util::colvec_type<uint64_t>;
    using colmat_value_t = ad::util::colmat_type<value_t>;

    using base_t::base_t;

    vec_value_t solve(
        const Eigen::Map<vec_value_t>& x_in,
        const Eigen::Map<vec_value_t>& quad,
        const Eigen::Map<vec_value_t>& linear,
        value_t l1,
        value_t l2,
        const Eigen::Map<colmat_value_t>& Q
    )
    {
        vec_value_t x = x_in;
        vec_uint64_t buffer(ptr->buffer_size());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(solve, x, quad, linear, l1, l2, Q, buffer); }();
        return x;
    }

    vec_value_t gradient(
        const Eigen::Map<vec_value_t>& x,
        const Eigen::Map<vec_value_t>& mu
    )
    {
        vec_value_t out(x.size());
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(gradient, x, mu, out); }();
        return out;
    }

    vec_value_t project(
        const Eigen::Map<vec_value_t>& x_in
    )
    {
        vec_value_t x = x_in;
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(project, x); }();
        return x;
    }

    value_t solve_zero(
        const Eigen::Map<vec_value_t>& v
    )
    {
        vec_uint64_t buffer(ptr->buffer_size());
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(solve_zero, v, buffer); }();
    }

    void clear()
    {
        [&]() { ADELIE_CORE_PIMPL_OVERRIDE(clear,); }();
    }

    // TODO: dual?

    int duals_nnz() const 
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(duals_nnz,); }();
    }
    int duals() const 
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(duals,); }();
    }
    int primals() const 
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(primals,); }();
    }
    size_t buffer_size() const 
    {
        return [&]() { ADELIE_CORE_PIMPL_OVERRIDE(buffer_size,); }();
    }
};

ADELIE_CORE_PIMPL_DERIVED(RConstraintBox64, RConstraintBase64, constraint_box_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RConstraintLinear64, RConstraintBase64, constraint_linear_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RConstraintOneSided64, RConstraintBase64, constraint_one_sided_64_t,)

RCPP_EXPOSED_CLASS(RConstraintBase64)
RCPP_EXPOSED_CLASS(RConstraintBox64)
RCPP_EXPOSED_CLASS(RConstraintLinear64)
RCPP_EXPOSED_CLASS(RConstraintOnesided64)

using r_constraint_base_64_t = RConstraintBase64;
using r_constraint_box_64_t = RConstraintBox64;
using r_constraint_linear_64_t = RConstraintLinear64;
using r_constraint_one_sided_64_t = RConstraintOneSided64;