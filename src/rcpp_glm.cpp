#include <Rcpp.h>
#include <RcppEigen.h>
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_binomial.hpp>
#include <adelie_core/glm/glm_cox.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/glm/glm_multigaussian.hpp>
#include <adelie_core/glm/glm_multinomial.hpp>
#include <adelie_core/glm/glm_poisson.hpp>

namespace ad = adelie_core;

using value_t = double;
using string_t = std::string;
using vec_value_t = ad::util::colvec_type<value_t>;
using mat_value_t = ad::util::colmat_type<value_t>;
using glm_base_64_t = ad::glm::GlmBase<value_t>;
using glm_multibase_64_t = ad::glm::GlmMultiBase<value_t>;
using glm_binomial_64_t = ad::glm::GlmBinomialLogit<value_t>;
using glm_cox_64_t = ad::glm::GlmCox<value_t>;
using glm_gaussian_64_t = ad::glm::GlmGaussian<value_t>;
using glm_poisson_64_t = ad::glm::GlmPoisson<value_t>;
using glm_multigaussian_64_t = ad::glm::GlmMultiGaussian<value_t>;
using glm_multinomial_64_t = ad::glm::GlmMultinomial<value_t>;

auto make_glm_binomial_64(
    const Eigen::Map<vec_value_t>& y,
    const Eigen::Map<vec_value_t>& weights
)
{
    return glm_binomial_64_t(y, weights);
}

auto make_glm_cox_64(
    const Eigen::Map<vec_value_t>& start,
    const Eigen::Map<vec_value_t>& stop,
    const Eigen::Map<vec_value_t>& status,
    const Eigen::Map<vec_value_t>& weights,
    const std::string& tie_method
)
{
    return glm_cox_64_t(start, stop, status, weights, tie_method);
}

auto make_glm_gaussian_64(
    const Eigen::Map<vec_value_t>& y,
    const Eigen::Map<vec_value_t>& weights
)
{
    return glm_gaussian_64_t(y, weights);
}

auto make_glm_poisson_64(
    const Eigen::Map<vec_value_t>& y,
    const Eigen::Map<vec_value_t>& weights
)
{
    return glm_poisson_64_t(y, weights);
}

auto make_glm_multigaussian_64(
    const Eigen::Map<mat_value_t>& yT,
    const Eigen::Map<vec_value_t>& weights
)
{
    return glm_multigaussian_64_t(yT.transpose(), weights);
}

auto make_glm_multinomial_64(
    const Eigen::Map<mat_value_t>& yT,
    const Eigen::Map<vec_value_t>& weights
)
{
    return glm_multinomial_64_t(yT.transpose(), weights);
}

void gradient(
    glm_base_64_t* glm,
    const Eigen::Map<vec_value_t>& eta,
    Eigen::Map<vec_value_t> grad
)
{
    glm->gradient(eta, grad);
}

void gradient(
    glm_multibase_64_t* glm,
    const Eigen::Map<mat_value_t>& etaT,
    Eigen::Map<mat_value_t> gradT
)
{
    auto eta = etaT.transpose();
    auto grad = gradT.transpose();
    glm->gradient(eta, grad);
}

RCPP_EXPOSED_WRAP(glm_binomial_64_t)
RCPP_EXPOSED_WRAP(glm_cox_64_t)
RCPP_EXPOSED_WRAP(glm_gaussian_64_t)
RCPP_EXPOSED_WRAP(glm_poisson_64_t)
RCPP_EXPOSED_WRAP(glm_multigaussian_64_t)
RCPP_EXPOSED_WRAP(glm_multinomial_64_t)

RCPP_MODULE(adelie_core_glm)
{
    /* base classes */
    Rcpp::class_<glm_base_64_t>("GlmBase64")
        .field_readonly("is_multi", &glm_base_64_t::is_multi)
        .field_readonly("name", &glm_base_64_t::name)
        .field_readonly("y", &glm_base_64_t::y)
        .field_readonly("weights", &glm_base_64_t::weights)
        .method("gradient", &gradient)
        .method("loss_full", &glm_base_64_t::loss_full)
        ;
    Rcpp::class_<glm_multibase_64_t>("GlmMultiBase64")
        .field_readonly("is_multi", &glm_multibase_64_t::is_multi)
        .field_readonly("name", &glm_multibase_64_t::name)
        .field_readonly("y", &glm_multibase_64_t::y)
        .field_readonly("weights", &glm_multibase_64_t::weights)
        .method("gradient", &gradient)
        .method("loss_full", &glm_multibase_64_t::loss_full)
        ;

    /* GLM classes */
    Rcpp::class_<glm_binomial_64_t>("GlmBinomial64")
        .derives<glm_base_64_t>("GlmBase64")
        ;
    Rcpp::class_<glm_cox_64_t>("GlmCox64")
        .derives<glm_base_64_t>("GlmBase64")
        ;
    Rcpp::class_<glm_gaussian_64_t>("GlmGaussian64")
        .derives<glm_base_64_t>("GlmBase64")
        ;
    Rcpp::class_<glm_poisson_64_t>("GlmPoisson64")
        .derives<glm_base_64_t>("GlmBase64")
        ;
    Rcpp::class_<glm_multigaussian_64_t>("GlmMultiGaussian64")
        .derives<glm_multibase_64_t>("GlmMultiBase64")
        ;
    Rcpp::class_<glm_multinomial_64_t>("GlmMultinomial64")
        .derives<glm_multibase_64_t>("GlmMultiBase64")
        ;

    /* factory functions */
    Rcpp::function(
        "make_glm_binomial_64", 
        &make_glm_binomial_64
    );
    Rcpp::function(
        "make_glm_cox_64", 
        &make_glm_cox_64
    );
    Rcpp::function(
        "make_glm_gaussian_64", 
        &make_glm_gaussian_64
    );
    Rcpp::function(
        "make_glm_poisson_64", 
        &make_glm_poisson_64
    );
    Rcpp::function(
        "make_glm_multigaussian_64", 
        &make_glm_multigaussian_64
    );
    Rcpp::function(
        "make_glm_multinomial_64", 
        &make_glm_multinomial_64
    );
}