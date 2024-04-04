#include <Rcpp.h>
#include <RcppEigen.h>
#include <adelie_core/matrix/matrix_naive_base.hpp>
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/state/state_multigaussian_naive.hpp>
#include <adelie_core/state/state_multiglm_naive.hpp>

namespace ad = adelie_core;

template <class BetasType>
static auto convert_betas(
    size_t p,
    const BetasType& betas
)
{
    using value_t = typename std::decay_t<BetasType>::value_type::Scalar;
    using index_t = int;
    using vec_value_t = ad::util::rowvec_type<value_t>;
    using vec_index_t = ad::util::rowvec_type<index_t>;
    using sp_mat_t = Eigen::SparseMatrix<value_t, Eigen::RowMajor, index_t>;

    const size_t l = betas.size();
    size_t nnz = 0;
    for (const auto& beta : betas) {
        nnz += beta.nonZeros();
    }
    vec_value_t values(nnz);
    vec_index_t inners(nnz); 
    vec_index_t outers(l+1);
    outers[0] = 0;
    int inner_idx = 0;
    for (size_t i = 0; i < l; ++i) {
        const auto& curr = betas[i];
        const auto nnz_curr = curr.nonZeros();
        Eigen::Map<vec_value_t>(
            values.data() + inner_idx,
            nnz_curr
        ) = Eigen::Map<const vec_value_t>(
            curr.valuePtr(),
            nnz_curr
        );
        Eigen::Map<vec_index_t>(
            inners.data() + inner_idx,
            nnz_curr
        ) = Eigen::Map<const vec_index_t>(
            curr.innerIndexPtr(),
            nnz_curr
        );
        outers[i+1] = outers[i] + nnz_curr;
        inner_idx += nnz_curr;
    }
    sp_mat_t out;
    out = Eigen::Map<const sp_mat_t>(
        l, 
        p,
        nnz,
        outers.data(),
        inners.data(),
        values.data()
    );
    return out;
}

using value_t = double;
using index_t = int;
using bool_t = int;
using safe_bool_t = int;
using vec_value_t = ad::util::colvec_type<value_t>;
using vec_index_t = ad::util::colvec_type<index_t>;
using vec_bool_t = ad::util::colvec_type<bool_t>;
using matrix_naive_base_64_t = ad::matrix::MatrixNaiveBase<value_t>;

using state_base_64_t = ad::state::StateBase<value_t, index_t, bool_t, safe_bool_t>;
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

// TODO: port over the following
// - gaussian_cov

auto make_state_gaussian_naive_64(
    matrix_naive_base_64_t& X,
    const Eigen::Map<vec_value_t>& X_means,
    value_t y_mean,
    value_t y_var,
    const Eigen::Map<vec_value_t>& resid,
    value_t resid_sum,
    const Eigen::Map<vec_index_t>& groups,
    const Eigen::Map<vec_index_t>& group_sizes,
    value_t alpha, 
    const Eigen::Map<vec_value_t>& penalty,
    const Eigen::Map<vec_value_t>& weights,
    const Eigen::Map<vec_value_t>& lmda_path,
    value_t lmda_max,
    value_t min_ratio,
    size_t lmda_path_size,
    size_t max_screen_size,
    size_t max_active_size,
    value_t pivot_subset_ratio,
    size_t pivot_subset_min,
    value_t pivot_slack_ratio,
    const std::string& screen_rule,
    size_t max_iters,
    value_t tol,
    value_t adev_tol,
    value_t ddev_tol,
    value_t newton_tol,
    size_t newton_max_iters,
    bool early_exit,
    bool setup_lmda_max,
    bool setup_lmda_path,
    bool intercept,
    size_t n_threads,
    const Eigen::Map<vec_index_t>& screen_set,
    const Eigen::Map<vec_value_t>& screen_beta, 
    const Eigen::Map<vec_bool_t>& screen_is_active,
    value_t rsq,
    value_t lmda,
    const Eigen::Map<vec_value_t>& grad
)
{
    return state_gaussian_naive_64_t(
        X, X_means, y_mean, y_var, resid, resid_sum, groups, group_sizes, alpha, penalty, weights, lmda_path,
        lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, max_iters, tol, adev_tol, ddev_tol, 
        newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, rsq, lmda, grad
    );
}

auto make_state_glm_naive_64(
    matrix_naive_base_64_t& X,
    const Eigen::Map<vec_value_t>& eta,
    const Eigen::Map<vec_value_t>& resid,
    const Eigen::Map<vec_index_t>& groups, 
    const Eigen::Map<vec_index_t>& group_sizes,
    value_t alpha, 
    const Eigen::Map<vec_value_t>& penalty,
    const Eigen::Map<vec_value_t>& offsets,
    const Eigen::Map<vec_value_t>& lmda_path,
    value_t loss_null,
    value_t loss_full,
    value_t lmda_max,
    value_t min_ratio,
    size_t lmda_path_size,
    size_t max_screen_size,
    size_t max_active_size,
    value_t pivot_subset_ratio,
    size_t pivot_subset_min,
    value_t pivot_slack_ratio,
    const std::string& screen_rule,
    size_t irls_max_iters,
    value_t irls_tol,
    size_t max_iters,
    value_t tol,
    value_t adev_tol,
    value_t ddev_tol,
    value_t newton_tol,
    size_t newton_max_iters,
    bool early_exit,
    bool setup_loss_null, 
    bool setup_lmda_max,
    bool setup_lmda_path,
    bool intercept,
    size_t n_threads,
    const Eigen::Map<vec_index_t>& screen_set,
    const Eigen::Map<vec_value_t>& screen_beta,
    const Eigen::Map<vec_bool_t>& screen_is_active,
    value_t beta0,
    value_t lmda,
    const Eigen::Map<vec_value_t>& grad
)
{
    return state_glm_naive_64_t(
        X, eta, resid, groups, group_sizes, alpha, penalty, offsets, lmda_path,
        loss_null, loss_full, lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
        irls_max_iters, irls_tol, max_iters, tol, adev_tol, ddev_tol, newton_tol, newton_max_iters,
        early_exit, setup_loss_null, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, beta0, lmda, grad
    );
}

auto make_state_multigaussian_naive_64(
    const std::string& group_type,
    size_t n_classes,
    bool multi_intercept,
    matrix_naive_base_64_t& X,
    const Eigen::Map<vec_value_t>& X_means,
    value_t y_mean,
    value_t y_var,
    const Eigen::Map<vec_value_t>& resid,
    value_t resid_sum,
    const Eigen::Map<vec_index_t>& groups,
    const Eigen::Map<vec_index_t>& group_sizes,
    value_t alpha, 
    const Eigen::Map<vec_value_t>& penalty,
    const Eigen::Map<vec_value_t>& weights,
    const Eigen::Map<vec_value_t>& lmda_path,
    value_t lmda_max,
    value_t min_ratio,
    size_t lmda_path_size,
    size_t max_screen_size,
    size_t max_active_size,
    value_t pivot_subset_ratio,
    size_t pivot_subset_min,
    value_t pivot_slack_ratio,
    const std::string& screen_rule,
    size_t max_iters,
    value_t tol,
    value_t adev_tol,
    value_t ddev_tol,
    value_t newton_tol,
    size_t newton_max_iters,
    bool early_exit,
    bool setup_lmda_max,
    bool setup_lmda_path,
    bool intercept,
    size_t n_threads,
    const Eigen::Map<vec_index_t>& screen_set,
    const Eigen::Map<vec_value_t>& screen_beta, 
    const Eigen::Map<vec_bool_t>& screen_is_active,
    value_t rsq,
    value_t lmda,
    const Eigen::Map<vec_value_t>& grad 
)
{
    return state_multigaussian_naive_64_t(
        group_type, n_classes, multi_intercept,
        X, X_means, y_mean, y_var, resid, resid_sum, groups, group_sizes, alpha, penalty, weights, lmda_path,
        lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, max_iters, tol, adev_tol, ddev_tol, 
        newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, rsq, lmda, grad
    );
}

auto make_state_multiglm_naive_64(
    const std::string& group_type,
    size_t n_classes,
    bool multi_intercept,
    matrix_naive_base_64_t& X,
    const Eigen::Map<vec_value_t>& eta,
    const Eigen::Map<vec_value_t>& resid,
    const Eigen::Map<vec_index_t>& groups, 
    const Eigen::Map<vec_index_t>& group_sizes,
    value_t alpha, 
    const Eigen::Map<vec_value_t>& penalty,
    const Eigen::Map<vec_value_t>& offsets,
    const Eigen::Map<vec_value_t>& lmda_path,
    value_t loss_null,
    value_t loss_full,
    value_t lmda_max,
    value_t min_ratio,
    size_t lmda_path_size,
    size_t max_screen_size,
    size_t max_active_size,
    value_t pivot_subset_ratio,
    size_t pivot_subset_min,
    value_t pivot_slack_ratio,
    const std::string& screen_rule,
    size_t irls_max_iters,
    value_t irls_tol,
    size_t max_iters,
    value_t tol,
    value_t adev_tol,
    value_t ddev_tol,
    value_t newton_tol,
    size_t newton_max_iters,
    bool early_exit,
    bool setup_loss_null, 
    bool setup_lmda_max,
    bool setup_lmda_path,
    bool intercept,
    size_t n_threads,
    const Eigen::Map<vec_index_t>& screen_set,
    const Eigen::Map<vec_value_t>& screen_beta,
    const Eigen::Map<vec_bool_t>& screen_is_active,
    value_t beta0,
    value_t lmda,
    const Eigen::Map<vec_value_t>& grad
)
{
    return state_multiglm_naive_64_t(
        group_type, n_classes, multi_intercept,
        X, eta, resid, groups, group_sizes, alpha, penalty, offsets, lmda_path,
        loss_null, loss_full, lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
        irls_max_iters, irls_tol, max_iters, tol, adev_tol, ddev_tol, newton_tol, newton_max_iters,
        early_exit, setup_loss_null, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, beta0, lmda, grad
    );
}

template <class StateType>
auto betas(StateType* state)
{
    using state_t = std::decay_t<StateType>;
    if constexpr (
        std::is_same_v<state_t, state_gaussian_naive_64_t> ||
        std::is_same_v<state_t, state_glm_naive_64_t>
    ) {
        return convert_betas(state->X->cols(), state->betas);
    } else if constexpr (
        std::is_same_v<state_t, state_multigaussian_naive_64_t> ||
        std::is_same_v<state_t, state_multiglm_naive_64_t>
    ) {
        return convert_betas(state->X->cols() - state->multi_intercept * state->n_classes, state->betas);
    } else {
        static_assert("Unexpected state type.");
    }
}

RCPP_EXPOSED_AS(matrix_naive_base_64_t)
RCPP_EXPOSED_WRAP(state_gaussian_naive_64_t)
RCPP_EXPOSED_WRAP(state_glm_naive_64_t)
RCPP_EXPOSED_WRAP(state_multigaussian_naive_64_t)
RCPP_EXPOSED_WRAP(state_multiglm_naive_64_t)

RCPP_MODULE(adelie_core_state) 
{
    Rcpp::class_<state_base_64_t>("StateBase64")
        .field_readonly("lmda_max", &state_base_64_t::lmda_max)
        .field_readonly("lmda", &state_base_64_t::lmda)
        .field_readonly("screen_set", &state_base_64_t::screen_set)
        .field_readonly("screen_beta", &state_base_64_t::screen_beta)
        .field_readonly("screen_is_active", &state_base_64_t::screen_is_active)
        .field_readonly("intercepts", &state_base_64_t::intercepts)
        .field_readonly("grad", &state_base_64_t::grad)
        .field_readonly("devs", &state_base_64_t::devs)
        .field_readonly("lmdas", &state_base_64_t::lmdas)
        .field_readonly("benchmark_fit_active", &state_base_64_t::benchmark_fit_active)
        ;
    Rcpp::class_<state_gaussian_naive_64_t>("StateGaussianNaive64")
        .derives<state_base_64_t>("StateBase64")
        .field_readonly("X_means", &state_gaussian_naive_64_t::X_means)
        .field_readonly("y_mean", &state_gaussian_naive_64_t::y_mean)
        .field_readonly("y_var", &state_gaussian_naive_64_t::y_var)
        .field_readonly("rsq", &state_gaussian_naive_64_t::rsq)
        .field_readonly("resid", &state_gaussian_naive_64_t::resid)
        .field_readonly("resid_sum", &state_gaussian_naive_64_t::resid_sum)
        .property("betas", &betas<state_gaussian_naive_64_t>, "")
        ;
    Rcpp::class_<state_glm_naive_64_t>("StateGlmNaive64")
        .derives<state_base_64_t>("StateBase64")
        .field_readonly("beta0", &state_glm_naive_64_t::beta0)
        .field_readonly("eta", &state_glm_naive_64_t::eta)
        .field_readonly("resid", &state_glm_naive_64_t::resid)
        .field_readonly("loss_null", &state_glm_naive_64_t::loss_null)
        .field_readonly("loss_full", &state_glm_naive_64_t::loss_full)
        .property("betas", &betas<state_glm_naive_64_t>, "")
        ;
    // TODO: Rcpp cannot handle inheritance properly. 
    // If field/property is already defined in the base class,
    // then even if they are overriden in the derived class,
    // the most base version will get called.
    // To get around this issue, we suffix the name with "multi" to make a unique name.
    Rcpp::class_<state_multigaussian_naive_64_t>("StateMultiGaussianNaive64")
        .derives<state_gaussian_naive_64_t>("StateGaussianNaive64")
        .field_readonly("intercepts_multi", &state_multigaussian_naive_64_t::intercepts)
        .property("betas_multi", &betas<state_multigaussian_naive_64_t>, "")
        ;
    Rcpp::class_<state_multiglm_naive_64_t>("StateMultiGlmNaive64")
        .derives<state_glm_naive_64_t>("StateGlmNaive64")
        .field_readonly("intercepts_multi", &state_multiglm_naive_64_t::intercepts)
        .property("betas_multi", &betas<state_multiglm_naive_64_t>, "")
        ;

    Rcpp::function(
        "make_state_gaussian_naive_64", 
        &make_state_gaussian_naive_64
    );
    Rcpp::function(
        "make_state_glm_naive_64", 
        &make_state_glm_naive_64
    );
    Rcpp::function(
        "make_state_multigaussian_naive_64", 
        &make_state_multigaussian_naive_64
    );
    Rcpp::function(
        "make_state_multiglm_naive_64", 
        &make_state_multiglm_naive_64
    );
}