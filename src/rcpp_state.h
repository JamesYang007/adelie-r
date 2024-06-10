#pragma once
#include "decl.h"
#include "rcpp_matrix.h"
#include <adelie_core/state/state_gaussian_naive.hpp>
#include <adelie_core/state/state_glm_naive.hpp>
#include <adelie_core/state/state_multigaussian_naive.hpp>
#include <adelie_core/state/state_multiglm_naive.hpp>

// TODO: port over the following
// - gaussian_cov

using state_base_64_t = ad::state::StateBase<double, int, int, int>;
using state_gaussian_naive_64_t = ad::state::StateGaussianNaive<
    matrix_naive_base_64_t, double, int, int, int
>;
using state_multigaussian_naive_64_t = ad::state::StateMultiGaussianNaive<
    matrix_naive_base_64_t, double, int, int, int
>;
using state_glm_naive_64_t = ad::state::StateGlmNaive<
    matrix_naive_base_64_t, double, int, int, int
>;
using state_multiglm_naive_64_t = ad::state::StateMultiGlmNaive<
    matrix_naive_base_64_t, double, int, int, int
>;

class RStateBase64: public state_base_64_t 
{
    using base_t = state_base_64_t; 
public: 
    using base_t::base_t; 
};
class RStateGaussianNaive64: public state_gaussian_naive_64_t 
{ 
    using base_t = state_gaussian_naive_64_t; 
public:
    using base_t::base_t;
};
class RStateMultiGaussianNaive64: public state_multigaussian_naive_64_t 
{
    using base_t = state_multigaussian_naive_64_t; 
public:
    using base_t::base_t;
};
class RStateGlmNaive64: public state_glm_naive_64_t 
{
    using base_t = state_glm_naive_64_t; 
public:
    using base_t::base_t;
};
class RStateMultiGlmNaive64: public state_multiglm_naive_64_t 
{
    using base_t = state_multiglm_naive_64_t; 
public:
    using base_t::base_t;
};

RCPP_EXPOSED_CLASS(RStateBase64)
RCPP_EXPOSED_CLASS(RStateGaussianNaive64)
RCPP_EXPOSED_CLASS(RStateMultiGaussianNaive64)
RCPP_EXPOSED_CLASS(RStateGlmNaive64)
RCPP_EXPOSED_CLASS(RStateMultiGlmNaive64)

using r_state_base_64_t = RStateBase64;
using r_state_gaussian_naive_64_t = RStateGaussianNaive64;
using r_state_multigaussian_naive_64_t = RStateMultiGaussianNaive64;
using r_state_glm_naive_64_t = RStateGlmNaive64;
using r_state_multiglm_naive_64_t = RStateMultiGlmNaive64;