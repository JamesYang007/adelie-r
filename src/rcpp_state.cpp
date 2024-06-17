#include "rcpp_constraint.h"
#include "rcpp_state.h"
#include "rcpp_matrix.h"
#include <adelie_core/matrix/matrix_naive_base.hpp>

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
using dyn_vec_constraint_t = std::vector<constraint_base_64_t*>;

auto make_r_state_gaussian_cov_64(Rcpp::List args)
{
    r_matrix_cov_base_64_t* A = args["A"];
    const Eigen::Map<vec_value_t> v = args["v"];
    const Rcpp::List constraints_r = args["constraints"];
    dyn_vec_constraint_t constraints;
    constraints.reserve(constraints_r.size());
    for (auto c : constraints_r) {
        if (c == R_NilValue) {
            constraints.push_back(nullptr);
        } else {
            constraints.push_back(Rcpp::as<r_constraint_base_64_t*>(c)->ptr.get());
        }
    }
    const Eigen::Map<vec_index_t> groups = args["groups"];
    const Eigen::Map<vec_index_t> group_sizes = args["group_sizes"];
    value_t alpha = args["alpha"];
    const Eigen::Map<vec_value_t> penalty = args["penalty"];
    const Eigen::Map<vec_value_t> lmda_path = args["lmda_path"];
    value_t lmda_max = args["lmda_max"];
    value_t min_ratio = args["min_ratio"];
    size_t lmda_path_size = args["lmda_path_size"];
    size_t max_screen_size = args["max_screen_size"];
    size_t max_active_size = args["max_active_size"];
    value_t pivot_subset_ratio = args["pivot_subset_ratio"];
    size_t pivot_subset_min = args["pivot_subset_min"];
    value_t pivot_slack_ratio = args["pivot_slack_ratio"];
    const std::string screen_rule = args["screen_rule"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    value_t rdev_tol = args["rdev_tol"];
    value_t newton_tol = args["newton_tol"];
    size_t newton_max_iters = args["newton_max_iters"];
    bool early_exit = args["early_exit"];
    bool setup_lmda_max = args["setup_lmda_max"];
    bool setup_lmda_path = args["setup_lmda_path"];
    size_t n_threads = args["n_threads"];
    const Eigen::Map<vec_index_t> screen_set = args["screen_set"];
    const Eigen::Map<vec_value_t> screen_beta = args["screen_beta"]; 
    const Eigen::Map<vec_bool_t> screen_is_active = args["screen_is_active"];
    const Eigen::Map<vec_value_t> screen_dual = args["screen_dual"];
    size_t active_set_size = args["active_set_size"];
    const Eigen::Map<vec_index_t> active_set = args["active_set"];
    value_t rsq = args["rsq"];
    value_t lmda = args["lmda"];
    const Eigen::Map<vec_value_t> grad = args["grad"];
    return new r_state_gaussian_cov_64_t(
        *A->ptr, v, constraints, groups, group_sizes, alpha, penalty, lmda_path,
        lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, max_iters, tol, rdev_tol,
        newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, n_threads,
        screen_set, screen_beta, screen_is_active, screen_dual, active_set_size, active_set, rsq, lmda, grad
    );
}

auto make_r_state_gaussian_naive_64(Rcpp::List args)
{
    r_matrix_naive_base_64_t* X = args["X"];
    const Eigen::Map<vec_value_t> X_means = args["X_means"];
    value_t y_mean = args["y_mean"];
    value_t y_var = args["y_var"];
    const Eigen::Map<vec_value_t> resid = args["resid"];
    value_t resid_sum = args["resid_sum"];
    const Rcpp::List constraints_r = args["constraints"];
    dyn_vec_constraint_t constraints;
    constraints.reserve(constraints_r.size());
    for (auto c : constraints_r) {
        if (c == R_NilValue) {
            constraints.push_back(nullptr);
        } else {
            constraints.push_back(Rcpp::as<r_constraint_base_64_t*>(c)->ptr.get());
        }
    }
    const Eigen::Map<vec_index_t> groups = args["groups"];
    const Eigen::Map<vec_index_t> group_sizes = args["group_sizes"];
    value_t alpha = args["alpha"];
    const Eigen::Map<vec_value_t> penalty = args["penalty"];
    const Eigen::Map<vec_value_t> weights = args["weights"];
    const Eigen::Map<vec_value_t> lmda_path = args["lmda_path"];
    value_t lmda_max = args["lmda_max"];
    value_t min_ratio = args["min_ratio"];
    size_t lmda_path_size = args["lmda_path_size"];
    size_t max_screen_size = args["max_screen_size"];
    size_t max_active_size = args["max_active_size"];
    value_t pivot_subset_ratio = args["pivot_subset_ratio"];
    size_t pivot_subset_min = args["pivot_subset_min"];
    value_t pivot_slack_ratio = args["pivot_slack_ratio"];
    const std::string screen_rule = args["screen_rule"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    value_t adev_tol = args["adev_tol"];
    value_t ddev_tol = args["ddev_tol"];
    value_t newton_tol = args["newton_tol"];
    size_t newton_max_iters = args["newton_max_iters"];
    bool early_exit = args["early_exit"];
    bool setup_lmda_max = args["setup_lmda_max"];
    bool setup_lmda_path = args["setup_lmda_path"];
    bool intercept = args["intercept"];
    size_t n_threads = args["n_threads"];
    const Eigen::Map<vec_index_t> screen_set = args["screen_set"];
    const Eigen::Map<vec_value_t> screen_beta = args["screen_beta"]; 
    const Eigen::Map<vec_bool_t> screen_is_active = args["screen_is_active"];
    const Eigen::Map<vec_value_t> screen_dual = args["screen_dual"];
    size_t active_set_size = args["active_set_size"];
    const Eigen::Map<vec_index_t> active_set = args["active_set"];
    value_t rsq = args["rsq"];
    value_t lmda = args["lmda"];
    const Eigen::Map<vec_value_t> grad = args["grad"];
    return new r_state_gaussian_naive_64_t(
        *X->ptr, X_means, y_mean, y_var, resid, resid_sum, constraints, groups, group_sizes, alpha, penalty, weights, lmda_path,
        lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, max_iters, tol, adev_tol, ddev_tol, 
        newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, screen_dual, active_set_size, active_set, rsq, lmda, grad
    );
}

auto make_r_state_glm_naive_64(Rcpp::List args)
{
    r_matrix_naive_base_64_t* X = args["X"];
    const Eigen::Map<vec_value_t> eta = args["eta"];
    const Eigen::Map<vec_value_t> resid = args["resid"];
    const Rcpp::List constraints_r = args["constraints"];
    dyn_vec_constraint_t constraints;
    constraints.reserve(constraints_r.size());
    for (auto c : constraints_r) {
        if (c == R_NilValue) {
            constraints.push_back(nullptr);
        } else {
            constraints.push_back(Rcpp::as<r_constraint_base_64_t*>(c)->ptr.get());
        }
    }
    const Eigen::Map<vec_index_t> groups = args["groups"]; 
    const Eigen::Map<vec_index_t> group_sizes = args["group_sizes"];
    value_t alpha = args["alpha"]; 
    const Eigen::Map<vec_value_t> penalty = args["penalty"];
    const Eigen::Map<vec_value_t> offsets = args["offsets"];
    const Eigen::Map<vec_value_t> lmda_path = args["lmda_path"];
    value_t loss_null = args["loss_null"];
    value_t loss_full = args["loss_full"];
    value_t lmda_max = args["lmda_max"];
    value_t min_ratio = args["min_ratio"];
    size_t lmda_path_size = args["lmda_path_size"];
    size_t max_screen_size = args["max_screen_size"];
    size_t max_active_size = args["max_active_size"];
    value_t pivot_subset_ratio = args["pivot_subset_ratio"];
    size_t pivot_subset_min = args["pivot_subset_min"];
    value_t pivot_slack_ratio = args["pivot_slack_ratio"];
    const std::string screen_rule = args["screen_rule"];
    size_t irls_max_iters = args["irls_max_iters"];
    value_t irls_tol = args["irls_tol"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    value_t adev_tol = args["adev_tol"];
    value_t ddev_tol = args["ddev_tol"];
    value_t newton_tol = args["newton_tol"];
    size_t newton_max_iters = args["newton_max_iters"];
    bool early_exit = args["early_exit"];
    bool setup_loss_null = args["setup_loss_null"]; 
    bool setup_lmda_max = args["setup_lmda_max"];
    bool setup_lmda_path = args["setup_lmda_path"];
    bool intercept = args["intercept"];
    size_t n_threads = args["n_threads"];
    const Eigen::Map<vec_index_t> screen_set = args["screen_set"];
    const Eigen::Map<vec_value_t> screen_beta = args["screen_beta"];
    const Eigen::Map<vec_bool_t> screen_is_active = args["screen_is_active"];
    const Eigen::Map<vec_value_t> screen_dual = args["screen_dual"];
    size_t active_set_size = args["active_set_size"];
    const Eigen::Map<vec_index_t> active_set = args["active_set"];
    value_t beta0 = args["beta0"];
    value_t lmda = args["lmda"];
    const Eigen::Map<vec_value_t> grad = args["grad"];
    return new r_state_glm_naive_64_t(
        *X->ptr, eta, resid, constraints, groups, group_sizes, alpha, penalty, offsets, lmda_path,
        loss_null, loss_full, lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
        irls_max_iters, irls_tol, max_iters, tol, adev_tol, ddev_tol, newton_tol, newton_max_iters,
        early_exit, setup_loss_null, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, screen_dual, active_set_size, active_set, beta0, lmda, grad
    );
}

auto make_r_state_multigaussian_naive_64(Rcpp::List args)
{
    const std::string group_type = args["group_type"];
    size_t n_classes = args["n_classes"];
    bool multi_intercept = args["multi_intercept"];
    r_matrix_naive_base_64_t* X = args["X"];
    const Eigen::Map<vec_value_t> X_means = args["X_means"];
    value_t y_mean = args["y_mean"];
    value_t y_var = args["y_var"];
    const Eigen::Map<vec_value_t> resid = args["resid"];
    value_t resid_sum = args["resid_sum"];
    const Rcpp::List constraints_r = args["constraints"];
    dyn_vec_constraint_t constraints;
    constraints.reserve(constraints_r.size());
    for (auto c : constraints_r) {
        if (c == R_NilValue) {
            constraints.push_back(nullptr);
        } else {
            constraints.push_back(Rcpp::as<r_constraint_base_64_t*>(c)->ptr.get());
        }
    }
    const Eigen::Map<vec_index_t> groups = args["groups"];
    const Eigen::Map<vec_index_t> group_sizes = args["group_sizes"];
    value_t alpha = args["alpha"]; 
    const Eigen::Map<vec_value_t> penalty = args["penalty"];
    const Eigen::Map<vec_value_t> weights = args["weights"];
    const Eigen::Map<vec_value_t> lmda_path = args["lmda_path"];
    value_t lmda_max = args["lmda_max"];
    value_t min_ratio = args["min_ratio"];
    size_t lmda_path_size = args["lmda_path_size"];
    size_t max_screen_size = args["max_screen_size"];
    size_t max_active_size = args["max_active_size"];
    value_t pivot_subset_ratio = args["pivot_subset_ratio"];
    size_t pivot_subset_min = args["pivot_subset_min"];
    value_t pivot_slack_ratio = args["pivot_slack_ratio"];
    const std::string screen_rule = args["screen_rule"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    value_t adev_tol = args["adev_tol"];
    value_t ddev_tol = args["ddev_tol"];
    value_t newton_tol = args["newton_tol"];
    size_t newton_max_iters = args["newton_max_iters"];
    bool early_exit = args["early_exit"];
    bool setup_lmda_max = args["setup_lmda_max"];
    bool setup_lmda_path = args["setup_lmda_path"];
    bool intercept = args["intercept"];
    size_t n_threads = args["n_threads"];
    const Eigen::Map<vec_index_t> screen_set = args["screen_set"];
    const Eigen::Map<vec_value_t> screen_beta = args["screen_beta"]; 
    const Eigen::Map<vec_bool_t> screen_is_active = args["screen_is_active"];
    const Eigen::Map<vec_value_t> screen_dual = args["screen_dual"];
    size_t active_set_size = args["active_set_size"];
    const Eigen::Map<vec_index_t> active_set = args["active_set"];
    value_t rsq = args["rsq"];
    value_t lmda = args["lmda"];
    const Eigen::Map<vec_value_t> grad = args["grad"];
    return new r_state_multigaussian_naive_64_t(
        group_type, n_classes, multi_intercept,
        *X->ptr, X_means, y_mean, y_var, resid, resid_sum, constraints, groups, group_sizes, alpha, penalty, weights, lmda_path,
        lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, max_iters, tol, adev_tol, ddev_tol, 
        newton_tol, newton_max_iters, early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, screen_dual, active_set_size, active_set, rsq, lmda, grad
    );
}

auto make_r_state_multiglm_naive_64(Rcpp::List args)
{
    const std::string group_type = args["group_type"];
    size_t n_classes = args["n_classes"];
    bool multi_intercept = args["multi_intercept"];
    r_matrix_naive_base_64_t* X = args["X"];
    const Eigen::Map<vec_value_t> eta = args["eta"];
    const Eigen::Map<vec_value_t> resid = args["resid"];
    const Rcpp::List constraints_r = args["constraints"];
    dyn_vec_constraint_t constraints;
    constraints.reserve(constraints_r.size());
    for (auto c : constraints_r) {
        if (c == R_NilValue) {
            constraints.push_back(nullptr);
        } else {
            constraints.push_back(Rcpp::as<r_constraint_base_64_t*>(c)->ptr.get());
        }
    }
    const Eigen::Map<vec_index_t> groups = args["groups"]; 
    const Eigen::Map<vec_index_t> group_sizes = args["group_sizes"];
    value_t alpha = args["alpha"]; 
    const Eigen::Map<vec_value_t> penalty = args["penalty"];
    const Eigen::Map<vec_value_t> offsets = args["offsets"];
    const Eigen::Map<vec_value_t> lmda_path = args["lmda_path"];
    value_t loss_null = args["loss_null"];
    value_t loss_full = args["loss_full"];
    value_t lmda_max = args["lmda_max"];
    value_t min_ratio = args["min_ratio"];
    size_t lmda_path_size = args["lmda_path_size"];
    size_t max_screen_size = args["max_screen_size"];
    size_t max_active_size = args["max_active_size"];
    value_t pivot_subset_ratio = args["pivot_subset_ratio"];
    size_t pivot_subset_min = args["pivot_subset_min"];
    value_t pivot_slack_ratio = args["pivot_slack_ratio"];
    const std::string screen_rule = args["screen_rule"];
    size_t irls_max_iters = args["irls_max_iters"];
    value_t irls_tol = args["irls_tol"];
    size_t max_iters = args["max_iters"];
    value_t tol = args["tol"];
    value_t adev_tol = args["adev_tol"];
    value_t ddev_tol = args["ddev_tol"];
    value_t newton_tol = args["newton_tol"];
    size_t newton_max_iters = args["newton_max_iters"];
    bool early_exit = args["early_exit"];
    bool setup_loss_null = args["setup_loss_null"]; 
    bool setup_lmda_max = args["setup_lmda_max"];
    bool setup_lmda_path = args["setup_lmda_path"];
    bool intercept = args["intercept"];
    size_t n_threads = args["n_threads"];
    const Eigen::Map<vec_index_t> screen_set = args["screen_set"];
    const Eigen::Map<vec_value_t> screen_beta = args["screen_beta"];
    const Eigen::Map<vec_bool_t> screen_is_active = args["screen_is_active"];
    const Eigen::Map<vec_value_t> screen_dual = args["screen_dual"];
    size_t active_set_size = args["active_set_size"];
    const Eigen::Map<vec_index_t> active_set = args["active_set"];
    value_t beta0 = args["beta0"];
    value_t lmda = args["lmda"];
    const Eigen::Map<vec_value_t> grad = args["grad"];
    return new r_state_multiglm_naive_64_t(
        group_type, n_classes, multi_intercept,
        *X->ptr, eta, resid, constraints, groups, group_sizes, alpha, penalty, offsets, lmda_path,
        loss_null, loss_full, lmda_max, min_ratio, lmda_path_size, max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio, screen_rule, 
        irls_max_iters, irls_tol, max_iters, tol, adev_tol, ddev_tol, newton_tol, newton_max_iters,
        early_exit, setup_loss_null, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        screen_set, screen_beta, screen_is_active, screen_dual, active_set_size, active_set, beta0, lmda, grad
    );
}

template <class StateType>
auto betas(StateType* state)
{
    using state_t = std::decay_t<StateType>;
    if constexpr (
        std::is_same_v<state_t, r_state_gaussian_naive_64_t> ||
        std::is_same_v<state_t, r_state_glm_naive_64_t>
    ) {
        return convert_betas(state->X->cols(), state->betas);
    } else if constexpr (
        std::is_same_v<state_t, r_state_gaussian_cov_64_t>
    ) {
        return convert_betas(state->A->cols(), state->betas);
    } else if constexpr (
        std::is_same_v<state_t, r_state_multigaussian_naive_64_t> ||
        std::is_same_v<state_t, r_state_multiglm_naive_64_t>
    ) {
        return convert_betas(state->X->cols() - state->multi_intercept * state->n_classes, state->betas);
    } else {
        static_assert("Unexpected state type.");
    }
}

RCPP_MODULE(adelie_core_state) 
{
    Rcpp::class_<state_base_64_t>("StateBase64")
        .field_readonly("lmda_max", &state_base_64_t::lmda_max)
        .field_readonly("lmda", &state_base_64_t::lmda)
        .field_readonly("screen_set", &state_base_64_t::screen_set)
        .field_readonly("screen_beta", &state_base_64_t::screen_beta)
        .field_readonly("screen_is_active", &state_base_64_t::screen_is_active)
        .field_readonly("screen_dual", &state_base_64_t::screen_dual)
        .field_readonly("active_set_size", &state_base_64_t::active_set_size)
        .field_readonly("active_set", &state_base_64_t::active_set)
        .field_readonly("intercepts", &state_base_64_t::intercepts)
        .field_readonly("grad", &state_base_64_t::grad)
        .field_readonly("devs", &state_base_64_t::devs)
        .field_readonly("lmdas", &state_base_64_t::lmdas)
        .field_readonly("benchmark_fit_active", &state_base_64_t::benchmark_fit_active)
        ;
    Rcpp::class_<r_state_base_64_t>("RStateBase64")
        .derives<state_base_64_t>("StateBase64")
        ;

    Rcpp::class_<state_gaussian_cov_64_t>("StateGaussianCov64")
        .derives<state_base_64_t>("StateBase64")
        .field_readonly("rsq", &state_gaussian_cov_64_t::rsq)
        ;
    Rcpp::class_<r_state_gaussian_cov_64_t>("RStateGaussianCov64")
        .derives<state_gaussian_cov_64_t>("StateGaussianCov64")
        .factory<Rcpp::List>(make_r_state_gaussian_cov_64)
        .property("betas", &betas<r_state_gaussian_cov_64_t>, "")
        ;

    Rcpp::class_<state_gaussian_naive_64_t>("StateGaussianNaive64")
        .derives<state_base_64_t>("StateBase64")
        .field_readonly("X_means", &state_gaussian_naive_64_t::X_means)
        .field_readonly("y_mean", &state_gaussian_naive_64_t::y_mean)
        .field_readonly("y_var", &state_gaussian_naive_64_t::y_var)
        .field_readonly("rsq", &state_gaussian_naive_64_t::rsq)
        .field_readonly("resid", &state_gaussian_naive_64_t::resid)
        .field_readonly("resid_sum", &state_gaussian_naive_64_t::resid_sum)
        ;
    Rcpp::class_<r_state_gaussian_naive_64_t>("RStateGaussianNaive64")
        .derives<state_gaussian_naive_64_t>("StateGaussianNaive64")
        .factory<Rcpp::List>(make_r_state_gaussian_naive_64)
        .property("betas", &betas<r_state_gaussian_naive_64_t>, "")
        ;

    Rcpp::class_<state_glm_naive_64_t>("StateGlmNaive64")
        .derives<state_base_64_t>("StateBase64")
        .field_readonly("beta0", &state_glm_naive_64_t::beta0)
        .field_readonly("eta", &state_glm_naive_64_t::eta)
        .field_readonly("resid", &state_glm_naive_64_t::resid)
        .field_readonly("loss_null", &state_glm_naive_64_t::loss_null)
        .field_readonly("loss_full", &state_glm_naive_64_t::loss_full)
        ;
    Rcpp::class_<r_state_glm_naive_64_t>("RStateGlmNaive64")
        .derives<state_glm_naive_64_t>("StateGlmNaive64")
        .factory<Rcpp::List>(make_r_state_glm_naive_64)
        .property("betas", &betas<r_state_glm_naive_64_t>, "")
        ;

    // TODO: Rcpp cannot handle inheritance properly. 
    // If field/property is already defined in the base class,
    // then even if they are overriden in the derived class,
    // the most base version will get called.
    // To get around this issue, we suffix the name with "multi" to make a unique name.
    Rcpp::class_<state_multigaussian_naive_64_t>("StateMultiGaussianNaive64")
        .derives<state_gaussian_naive_64_t>("StateGaussianNaive64")
        .field_readonly("intercepts_multi", &state_multigaussian_naive_64_t::intercepts)
        ;
    Rcpp::class_<r_state_multigaussian_naive_64_t>("RStateMultiGaussianNaive64")
        .derives<state_multigaussian_naive_64_t>("StateMultiGaussianNaive64")
        .factory<Rcpp::List>(make_r_state_multigaussian_naive_64)
        .property("betas_multi", &betas<r_state_multigaussian_naive_64_t>, "")
        ;

    Rcpp::class_<state_multiglm_naive_64_t>("StateMultiGlmNaive64")
        .derives<state_glm_naive_64_t>("RStateGlmNaive64")
        .field_readonly("intercepts_multi", &state_multiglm_naive_64_t::intercepts)
        ;
    Rcpp::class_<r_state_multiglm_naive_64_t>("RStateMultiGlmNaive64")
        .derives<state_multiglm_naive_64_t>("StateMultiGlmNaive64")
        .factory<Rcpp::List>(make_r_state_multiglm_naive_64)
        .property("betas_multi", &betas<r_state_multiglm_naive_64_t>, "")
        ;
}