#include "rcpp_constraint.h"

using value_t = double;
using vec_value_t = ad::util::colvec_type<value_t>;
using dense_64F_t = ad::util::colmat_type<value_t>;

/* Factory functions */

auto make_r_constraint_box_64(Rcpp::List args)
{
    const Eigen::Map<vec_value_t> l = args["l"];
    const Eigen::Map<vec_value_t> u = args["u"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    size_t pinball_max_iters = args["pinball_max_iters"];
    value_t pinball_tol = args["pinball_tol"];
    value_t slack = args["slack"];
    return new r_constraint_box_64_t(
        l, u, max_iters, tol, pinball_max_iters, pinball_tol, slack
    );
}

auto make_r_constraint_linear_64(Rcpp::List args)
{
    r_matrix_constraint_base_64_t* A = args["A"];
    const Eigen::Map<vec_value_t> l = args["l"];
    const Eigen::Map<vec_value_t> u = args["u"];
    const Eigen::Map<vec_value_t> A_vars = args["A_vars"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    size_t nnls_max_iters = args["nnls_max_iters"];
    value_t nnls_tol = args["nnls_tol"];
    size_t pinball_max_iters = args["pinball_max_iters"];
    value_t pinball_tol = args["pinball_tol"];
    value_t slack = args["slack"];
    size_t n_threads = args["n_threads"];
    return new r_constraint_linear_64_t(
        *A->ptr, l, u, A_vars, max_iters, tol, nnls_max_iters, nnls_tol, pinball_max_iters, pinball_tol, slack, n_threads
    );
}

auto make_r_constraint_one_sided_64(Rcpp::List args)
{
    const Eigen::Map<vec_value_t> sgn = args["sgn"];
    const Eigen::Map<vec_value_t> b = args["b"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    size_t pinball_max_iters = args["pinball_max_iters"];
    value_t pinball_tol = args["pinball_tol"];
    value_t slack = args["slack"];
    return new r_constraint_one_sided_64_t(
        sgn, b, max_iters, tol, pinball_max_iters, pinball_tol, slack
    );
}

RCPP_MODULE(adelie_core_constraint)
{
    Rcpp::class_<r_constraint_base_64_t>("RConstraintBase64")
        .method("solve", &r_constraint_base_64_t::solve)
        .method("gradient", &r_constraint_base_64_t::gradient)
        .method("project", &r_constraint_base_64_t::project)
        .method("solve_zero", &r_constraint_base_64_t::solve_zero)
        .method("clear", &r_constraint_base_64_t::clear)
        .property("duals_nnz", &r_constraint_base_64_t::duals_nnz)
        .property("dual_sizes", &r_constraint_base_64_t::duals)
        .property("duals", &r_constraint_base_64_t::duals)
        .property("primals", &r_constraint_base_64_t::primals)
        .property("buffer_size", &r_constraint_base_64_t::buffer_size)
        ;

    Rcpp::class_<r_constraint_box_64_t>("RConstraintBox64")
        .derives<r_constraint_base_64_t>("RConstraintBase64")
        .factory<Rcpp::List>(make_r_constraint_box_64)
        ;
    Rcpp::class_<r_constraint_linear_64_t>("RConstraintLinear64")
        .derives<r_constraint_base_64_t>("RConstraintBase64")
        .factory<Rcpp::List>(make_r_constraint_linear_64)
        ;
    Rcpp::class_<r_constraint_one_sided_64_t>("RConstraintOneSided64")
        .derives<r_constraint_base_64_t>("RConstraintBase64")
        .factory<Rcpp::List>(make_r_constraint_one_sided_64)
        ;
}