// Pure-C++ reprex of the kkt/screen 1-ULP infinite loop in adelie_core.
//
// Reads bit-exact state inputs from cpp_data/*.bin (written by dump_state.R)
// and calls ad::solver::gaussian::naive::solve directly via the adelie_core
// templates, with no R, Python, or Rcpp linkage.
//
// Build:  ./build_cpp_reprex.sh
// Run:    timeout 30 ./cpp_reprex
//
// Expected on unfixed solver_base.hpp: 100% CPU until killed by `timeout`.
// Expected on fixed solver_base.hpp:   prints "completed".

#include <Eigen/Core>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <adelie_core/configs.hpp>
#include <adelie_core/util/types.hpp>
#include <adelie_core/util/tqdm.hpp>
#include <adelie_core/constraint/constraint_base.hpp>
#include <adelie_core/state/state_base.ipp>
#include <adelie_core/state/state_gaussian_pin_base.ipp>
#include <adelie_core/state/state_gaussian_pin_naive.ipp>
#include <adelie_core/state/state_gaussian_naive.ipp>
#include <adelie_core/matrix/matrix_naive_base.ipp>
#include <adelie_core/matrix/matrix_naive_dense.ipp>
#include <adelie_core/matrix/matrix_naive_standardize.ipp>
#include <adelie_core/solver/solver_gaussian_naive.hpp>

namespace ad = adelie_core;

template <typename T>
static std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    f.seekg(0, std::ios::end);
    std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<T> v(static_cast<std::size_t>(bytes) / sizeof(T));
    if (bytes > 0) f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}
template <typename T>
static T read_scalar(const std::string& path) {
    auto v = read_bin<T>(path);
    if (v.empty()) { std::fprintf(stderr, "empty scalar file %s\n", path.c_str()); std::exit(1); }
    return v[0];
}

int main() {
    using value_t = double;
    using index_t = int;
    using bool_t  = bool;
    using sbool_t = int8_t;

    using vec_value_t = ad::util::rowvec_type<value_t>;
    using vec_index_t = ad::util::rowvec_type<index_t>;
    using vec_bool_t  = ad::util::rowvec_type<bool_t>;

    using constraint_t = ad::constraint::ConstraintBase<value_t, index_t>;
    using mat_naive_t  = ad::matrix::MatrixNaiveBase<value_t, index_t>;
    using mat_dense_t  = ad::matrix::MatrixNaiveDense<Eigen::Matrix<value_t, -1, -1>, index_t>;
    using mat_std_t    = ad::matrix::MatrixNaiveStandardize<value_t, index_t>;
    using state_t      = ad::state::StateGaussianNaive<constraint_t, mat_naive_t, value_t, index_t, bool_t, sbool_t>;

    std::printf("[1] entered main\n"); std::fflush(stdout);
    // ------ shape ------
    auto shape = read_bin<int32_t>("cpp_data/shape.bin");
    const int n = shape[0], p = shape[1], G = shape[2];
    std::printf("n=%d  p=%d  G=%d\n", n, p, G);

    // ------ inputs (bit-exact via fread of little-endian doubles/int32) ------
    auto X_data       = read_bin<value_t>("cpp_data/X_train.bin");
    auto weights_data = read_bin<value_t>("cpp_data/weights.bin");
    auto centers_data = read_bin<value_t>("cpp_data/centers.bin");
    auto scales_data  = read_bin<value_t>("cpp_data/scales.bin");
    auto X_means_data = read_bin<value_t>("cpp_data/X_means.bin");
    auto resid_data   = read_bin<value_t>("cpp_data/resid.bin");
    auto grad_data    = read_bin<value_t>("cpp_data/grad.bin");
    auto penalty_data = read_bin<value_t>("cpp_data/penalty.bin");
    auto groups_data  = read_bin<index_t>("cpp_data/groups.bin");
    auto gsize_data   = read_bin<index_t>("cpp_data/group_sizes.bin");
    auto dual_data    = read_bin<index_t>("cpp_data/dual_groups.bin");
    auto sset_data    = read_bin<index_t>("cpp_data/screen_set.bin");
    auto sbeta_data   = read_bin<value_t>("cpp_data/screen_beta.bin");
    auto siact_data   = read_bin<int32_t>("cpp_data/screen_is_active.bin");
    auto aset_data    = read_bin<index_t>("cpp_data/active_set.bin");
    const value_t y_mean    = read_scalar<value_t>("cpp_data/y_mean.bin");
    const value_t y_var     = read_scalar<value_t>("cpp_data/y_var.bin");
    const value_t resid_sum = read_scalar<value_t>("cpp_data/resid_sum.bin");

    // R is column-major; X_data is laid out that way.
    Eigen::Map<const Eigen::Matrix<value_t, -1, -1>> X_view(X_data.data(), n, p);
    Eigen::Matrix<value_t, -1, -1> X_owned = X_view;            // own a copy

    std::printf("[2] read all bin files\n"); std::fflush(stdout);
    // ------ matrix tree: Standardize<Dense> (the buggy code path) ------
    mat_dense_t X_dense(X_owned, /*n_threads*/ 1);
    Eigen::Map<const vec_value_t> centers(centers_data.data(), p);
    Eigen::Map<const vec_value_t> scales(scales_data.data(), p);
    mat_std_t   X_std(X_dense, centers, scales, /*n_threads*/ 1);
    std::printf("[3] built matrix tree Standardize<Dense>\n"); std::fflush(stdout);

    // ------ state inputs as Eigen::Map ------
    Eigen::Map<const vec_value_t> X_means(X_means_data.data(), p);
    Eigen::Map<const vec_value_t> resid(resid_data.data(), n);
    Eigen::Map<const vec_value_t> weights_v(weights_data.data(), n);
    Eigen::Map<const vec_value_t> penalty_v(penalty_data.data(), G);
    Eigen::Map<const vec_index_t> groups_v(groups_data.data(), G);
    Eigen::Map<const vec_index_t> gsize_v(gsize_data.data(), G);
    // Eigen::Map<...,0> with nullptr storage segfaults on some libc++; back
    // each empty vector with a one-element dummy and Map size 0.
    static const index_t  one_idx[1] = {0};
    static const value_t  one_val[1] = {0.0};
    Eigen::Map<const vec_index_t> dual_v (dual_data.data(), G);
    Eigen::Map<const vec_index_t> sset_v (sset_data.empty()  ? one_idx : sset_data.data(),  0);
    Eigen::Map<const vec_value_t> sbeta_v(sbeta_data.empty() ? one_val : sbeta_data.data(), 0);
    vec_bool_t siact_v(0);
    Eigen::Map<const vec_index_t> aset_v(aset_data.data(), G);
    Eigen::Map<const vec_value_t> grad_v(grad_data.data(), p);

    // No supplied lambda path; let the solver compute lmda_max and the path itself.
    vec_value_t lmda_path(0);

    // No actual constraints, but vector must have length G with nullptrs
    std::vector<constraint_t*> constraints(G, nullptr);

    // Defaults matching grpnet() in R/solver.R
    const value_t alpha               = 0.7;
    const value_t lmda_max            = 0.0;
    const value_t min_ratio           = 1e-2;
    const std::size_t lmda_path_size  = 1;        // matches the original reprex
    const std::size_t max_screen_size = static_cast<std::size_t>(G);
    const std::size_t max_active_size = static_cast<std::size_t>(G);
    const value_t pivot_subset_ratio  = 0.1;
    const std::size_t pivot_subset_min= 1;
    const value_t pivot_slack_ratio   = 1.25;
    const std::string screen_rule     = "pivot";
    const std::size_t max_iters       = 100000;
    const value_t tol                 = 1e-7;
    const value_t adev_tol            = 0.9;
    const value_t ddev_tol            = 1e-4;
    const value_t newton_tol          = 1e-12;
    const std::size_t newton_max_iters= 1000;
    const bool early_exit             = true;
    const bool setup_lmda_max         = true;
    const bool setup_lmda_path        = true;
    const bool intercept              = true;
    const std::size_t n_threads       = 1;

    std::printf("[4] about to construct StateGaussianNaive\n"); std::fflush(stdout);
    state_t state(
        X_std,
        X_means, y_mean, y_var,
        resid, resid_sum,
        constraints,
        groups_v, gsize_v, dual_v,
        alpha, penalty_v, weights_v, lmda_path,
        lmda_max, min_ratio, lmda_path_size,
        max_screen_size, max_active_size,
        pivot_subset_ratio, pivot_subset_min, pivot_slack_ratio,
        screen_rule,
        max_iters, tol, adev_tol, ddev_tol, newton_tol, newton_max_iters,
        early_exit, setup_lmda_max, setup_lmda_path, intercept, n_threads,
        sset_v, sbeta_v, siact_v, /*active_set_size*/ 0, aset_v,
        /*rsq*/ 0.0,
        /*lmda*/ std::numeric_limits<value_t>::infinity(),
        grad_v
    );

    std::printf("[5] state constructed\n"); std::fflush(stdout);
    auto pb = ad::util::tq::trange(0);
    pb.set_display(false);

    std::printf("calling adelie solver ...\n"); std::fflush(stdout);
    ad::solver::gaussian::naive::solve(
        state, pb,
        []() { return false; },          // exit_cond
        []() {}                           // check_user_interrupt
    );
    std::printf("returned (this line should not be reached on unfixed adelie)\n");
    return 0;
}
