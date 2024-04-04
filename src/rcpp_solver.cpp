#include <Rcpp.h>
#include <RcppEigen.h>
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/state/state_multigaussian_naive.hpp>
#include <adelie_core/state/state_multiglm_naive.hpp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/solver/solver_glm_naive.hpp>
#include <adelie_core/solver/solver_multigaussian_naive.hpp>
#include <adelie_core/solver/solver_multiglm_naive.hpp>

namespace ad = adelie_core;

using value_t = double;
using index_t = int;
using bool_t = int;
using safe_bool_t = int;
using glm_base_64_t = ad::glm::GlmBase<value_t>;
using glm_multibase_64_t = ad::glm::GlmMultiBase<value_t>;
using matrix_naive_base_64_t = ad::matrix::MatrixNaiveBase<value_t>;
using state_gaussian_naive_64_t = ad::state::StateGaussianNaive<
    matrix_naive_base_64_t, value_t, index_t, bool_t, safe_bool_t
>;
using state_multigaussian_naive_64_t = ad::state::StateMultiGaussianNaive<
    matrix_naive_base_64_t, value_t, index_t, bool_t, safe_bool_t
>;
using state_glm_naive_64_t = ad::state::StateGlmNaive<
    matrix_naive_base_64_t, value_t, index_t, bool_t, safe_bool_t
>;
using state_multiglm_naive_64_t = ad::state::StateMultiGlmNaive<
    matrix_naive_base_64_t, value_t, index_t, bool_t, safe_bool_t
>;

Rcpp::List solve_gaussian_naive_64(
    state_gaussian_naive_64_t state,
    bool display_progress_bar
)
{
    using sw_t = ad::util::Stopwatch;

    const auto update_coefficients_f = [](
        const auto& L,
        const auto& v,
        auto l1,
        auto l2,
        auto tol,
        size_t max_iters,
        auto& x,
        auto& iters,
        auto& buffer1,
        auto& buffer2
    ){
        ad::solver::gaussian::pin::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&]() {
        Rcpp::checkUserInterrupt();
    };

    std::string error;

    sw_t sw;
    sw.start();
    try {
        ad::solver::gaussian::naive::solve(
            state, display_progress_bar, [](){ return false; },
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return Rcpp::List::create(
        Rcpp::Named("state")=state,
        Rcpp::Named("error")=error, 
        Rcpp::Named("total_time")=total_time
    );
} 

Rcpp::List solve_glm_naive_64(
    state_glm_naive_64_t state,
    glm_base_64_t& glm,
    bool display_progress_bar
)
{
    using sw_t = ad::util::Stopwatch;

    const auto update_coefficients_f = [](
        const auto& L,
        const auto& v,
        auto l1,
        auto l2,
        auto tol,
        size_t max_iters,
        auto& x,
        auto& iters,
        auto& buffer1,
        auto& buffer2
    ){
        ad::solver::gaussian::pin::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&]() {
        Rcpp::checkUserInterrupt();
    };

    std::string error;

    sw_t sw;
    sw.start();
    try {
        ad::solver::glm::naive::solve(
            state, glm, display_progress_bar, [](){ return false; },
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return Rcpp::List::create(
        Rcpp::Named("state")=state,
        Rcpp::Named("error")=error, 
        Rcpp::Named("total_time")=total_time
    );
} 

Rcpp::List solve_multigaussian_naive_64(
    state_multigaussian_naive_64_t state,
    bool display_progress_bar
)
{
    using sw_t = ad::util::Stopwatch;

    const auto update_coefficients_f = [](
        const auto& L,
        const auto& v,
        auto l1,
        auto l2,
        auto tol,
        size_t max_iters,
        auto& x,
        auto& iters,
        auto& buffer1,
        auto& buffer2
    ){
        ad::solver::gaussian::pin::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&]() {
        Rcpp::checkUserInterrupt();
    };

    std::string error;

    sw_t sw;
    sw.start();
    try {
        ad::solver::multigaussian::naive::solve(
            state, display_progress_bar, [](){ return false; },
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return Rcpp::List::create(
        Rcpp::Named("state")=state,
        Rcpp::Named("error")=error, 
        Rcpp::Named("total_time")=total_time
    );
} 

Rcpp::List solve_multiglm_naive_64(
    state_multiglm_naive_64_t state,
    glm_multibase_64_t& glm,
    bool display_progress_bar
)
{
    using sw_t = ad::util::Stopwatch;

    const auto update_coefficients_f = [](
        const auto& L,
        const auto& v,
        auto l1,
        auto l2,
        auto tol,
        size_t max_iters,
        auto& x,
        auto& iters,
        auto& buffer1,
        auto& buffer2
    ){
        ad::solver::gaussian::pin::update_coefficients(
            L, v, l1, l2, tol, max_iters, x, iters, buffer1, buffer2
        );
    };

    const auto check_user_interrupt = [&]() {
        Rcpp::checkUserInterrupt();
    };

    std::string error;

    sw_t sw;
    sw.start();
    try {
        ad::solver::multiglm::naive::solve(
            state, glm, display_progress_bar, [](){ return false; },
            update_coefficients_f, check_user_interrupt
        );
    } catch(const std::exception& e) {
        error = e.what(); 
    }
    double total_time = sw.elapsed();

    return Rcpp::List::create(
        Rcpp::Named("state")=state,
        Rcpp::Named("error")=error, 
        Rcpp::Named("total_time")=total_time
    );
} 

RCPP_EXPOSED_AS(glm_base_64_t)
RCPP_EXPOSED_AS(glm_multibase_64_t)
RCPP_EXPOSED_AS(state_gaussian_naive_64_t)
RCPP_EXPOSED_WRAP(state_gaussian_naive_64_t)
RCPP_EXPOSED_AS(state_glm_naive_64_t)
RCPP_EXPOSED_WRAP(state_glm_naive_64_t)
RCPP_EXPOSED_AS(state_multigaussian_naive_64_t)
RCPP_EXPOSED_WRAP(state_multigaussian_naive_64_t)
RCPP_EXPOSED_AS(state_multiglm_naive_64_t)
RCPP_EXPOSED_WRAP(state_multiglm_naive_64_t)

RCPP_MODULE(adelie_core_solver)
{
    Rcpp::function(
        "solve_gaussian_naive_64",
        &solve_gaussian_naive_64
    );
    Rcpp::function(
        "solve_glm_naive_64",
        &solve_glm_naive_64
    );
    Rcpp::function(
        "solve_multigaussian_naive_64",
        &solve_multigaussian_naive_64
    );
    Rcpp::function(
        "solve_multiglm_naive_64",
        &solve_multiglm_naive_64
    );
}