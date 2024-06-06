#include "rcpp_glm.hpp"

using value_t = double;
using vec_value_t = ad::util::colvec_type<value_t>;
using rowarr_value_t = ad::util::rowarr_type<value_t>;
using colmat_value_t = ad::util::colmat_type<value_t>;

/* Factory functions */

auto make_r_glm_binomial_logit_64(
    const Eigen::Map<vec_value_t>& y,
    const Eigen::Map<vec_value_t>& weights
)
{
    return r_glm_binomial_logit_64_t(y, weights);
}

auto make_r_glm_binomial_probit_64(
    const Eigen::Map<vec_value_t>& y,
    const Eigen::Map<vec_value_t>& weights
)
{
    return r_glm_binomial_probit_64_t(y, weights);
}

auto make_r_glm_cox_64(
    const Eigen::Map<vec_value_t>& start,
    const Eigen::Map<vec_value_t>& stop,
    const Eigen::Map<vec_value_t>& status,
    const Eigen::Map<vec_value_t>& weights,
    const std::string& tie_method
)
{
    return r_glm_cox_64_t(start, stop, status, weights, tie_method);
}

auto make_r_glm_gaussian_64(
    const Eigen::Map<vec_value_t>& y,
    const Eigen::Map<vec_value_t>& weights
)
{
    return r_glm_gaussian_64_t(y, weights);
}

auto make_r_glm_poisson_64(
    const Eigen::Map<vec_value_t>& y,
    const Eigen::Map<vec_value_t>& weights
)
{
    return r_glm_poisson_64_t(y, weights);
}

auto make_r_glm_multigaussian_64(
    const Eigen::Map<colmat_value_t>& yT,
    const Eigen::Map<vec_value_t>& weights
)
{
    Eigen::Map<const rowarr_value_t> y(yT.data(), yT.cols(), yT.rows());
    return r_glm_multigaussian_64_t(y, weights);
}

auto make_r_glm_multinomial_64(
    const Eigen::Map<colmat_value_t>& yT,
    const Eigen::Map<vec_value_t>& weights
)
{
    Eigen::Map<const rowarr_value_t> y(yT.data(), yT.cols(), yT.rows());
    return r_glm_multinomial_64_t(y, weights);
}

RCPP_MODULE(adelie_core_glm)
{
    /* base classes */
    Rcpp::class_<r_glm_base_64_t>("RGlmBase64")
        .constructor()
        .property("is_multi", &r_glm_base_64_t::is_multi, "")
        .property("name", &r_glm_base_64_t::name, "")
        .property("y", &r_glm_base_64_t::y, "")
        .property("weights", &r_glm_base_64_t::weights, "")
        .method("gradient", &r_glm_base_64_t::gradient)
        .method("loss_full", &r_glm_base_64_t::loss_full)
        ;

    Rcpp::class_<r_glm_multibase_64_t>("RGlmMultiBase64")
        .constructor()
        .property("is_multi", &r_glm_multibase_64_t::is_multi, "")
        .property("name", &r_glm_multibase_64_t::name, "")
        .property("y", &r_glm_multibase_64_t::y, "")
        .property("weights", &r_glm_multibase_64_t::weights, "")
        .method("gradient", &r_glm_multibase_64_t::gradient)
        .method("loss_full", &r_glm_multibase_64_t::loss_full)
        ;

    /* GLM classes */
    Rcpp::class_<r_glm_binomial_logit_64_t>("RGlmBinomialLogit64")
        .derives<r_glm_base_64_t>("RGlmBase64")
        ;
    Rcpp::class_<r_glm_binomial_probit_64_t>("RGlmBinomialProbit64")
        .derives<r_glm_base_64_t>("RGlmBase64")
        ;
    Rcpp::class_<r_glm_cox_64_t>("RGlmCox64")
        .derives<r_glm_base_64_t>("RGlmBase64")
        ;
    Rcpp::class_<r_glm_gaussian_64_t>("RGlmGaussian64")
        .derives<r_glm_base_64_t>("RGlmBase64")
        ;
    Rcpp::class_<r_glm_poisson_64_t>("RGlmPoisson64")
        .derives<r_glm_base_64_t>("RGlmBase64")
        ;
    Rcpp::class_<r_glm_multigaussian_64_t>("RGlmMultiGaussian64")
        .derives<r_glm_multibase_64_t>("RGlmMultiBase64")
        ;
    Rcpp::class_<r_glm_multinomial_64_t>("RGlmMultinomial64")
        .derives<r_glm_multibase_64_t>("RGlmMultiBase64")
        ;

    /* factory functions */
    Rcpp::function(
        "make_r_glm_binomial_logit_64", 
        &make_r_glm_binomial_logit_64
    );
    Rcpp::function(
        "make_r_glm_binomial_probit_64", 
        &make_r_glm_binomial_probit_64
    );
    Rcpp::function(
        "make_r_glm_cox_64", 
        &make_r_glm_cox_64
    );
    Rcpp::function(
        "make_r_glm_gaussian_64", 
        &make_r_glm_gaussian_64
    );
    Rcpp::function(
        "make_r_glm_poisson_64", 
        &make_r_glm_poisson_64
    );
    Rcpp::function(
        "make_r_glm_multigaussian_64", 
        &make_r_glm_multigaussian_64
    );
    Rcpp::function(
        "make_r_glm_multinomial_64", 
        &make_r_glm_multinomial_64
    );
}