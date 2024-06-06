#include "decl.hpp"
#include "rcpp_glm.hpp"
#include "rcpp_matrix.hpp"
#include "rcpp_state.hpp"
#include <adelie_core/solver/solver_gaussian_naive.hpp>
#include <adelie_core/solver/solver_glm_naive.hpp>
#include <adelie_core/solver/solver_multigaussian_naive.hpp>
#include <adelie_core/solver/solver_multiglm_naive.hpp>

Rcpp::List r_solve_gaussian_naive_64(
    r_state_gaussian_naive_64_t state,
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
            static_cast<state_gaussian_naive_64_t&>(state), 
            display_progress_bar, [](){ return false; },
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

Rcpp::List r_solve_glm_naive_64(
    r_state_glm_naive_64_t state,
    r_glm_base_64_t& glm,
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
            static_cast<state_glm_naive_64_t&>(state), 
            *glm.ptr, display_progress_bar, [](){ return false; },
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

Rcpp::List r_solve_multigaussian_naive_64(
    r_state_multigaussian_naive_64_t state,
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
            static_cast<state_multigaussian_naive_64_t&>(state), 
            display_progress_bar, [](){ return false; },
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

Rcpp::List r_solve_multiglm_naive_64(
    r_state_multiglm_naive_64_t state,
    r_glm_multibase_64_t& glm,
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
            static_cast<state_multiglm_naive_64_t&>(state), 
            *glm.ptr, display_progress_bar, [](){ return false; },
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

RCPP_MODULE(adelie_core_solver)
{
    Rcpp::function(
        "r_solve_gaussian_naive_64",
        &r_solve_gaussian_naive_64
    );
    Rcpp::function(
        "r_solve_glm_naive_64",
        &r_solve_glm_naive_64
    );
    Rcpp::function(
        "r_solve_multigaussian_naive_64",
        &r_solve_multigaussian_naive_64
    );
    Rcpp::function(
        "r_solve_multiglm_naive_64",
        &r_solve_multiglm_naive_64
    );
}