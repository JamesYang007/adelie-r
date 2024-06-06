#pragma once
#include "decl.hpp"
#include "utils.hpp"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_binomial.hpp>
#include <adelie_core/glm/glm_cox.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/glm/glm_multigaussian.hpp>
#include <adelie_core/glm/glm_multinomial.hpp>
#include <adelie_core/glm/glm_poisson.hpp>

using glm_base_64_t = ad::glm::GlmBase<double>;
using glm_multibase_64_t = ad::glm::GlmMultiBase<double>;
using glm_binomial_logit_64_t = ad::glm::GlmBinomialLogit<double>;
using glm_binomial_probit_64_t = ad::glm::GlmBinomialProbit<double>;
using glm_cox_64_t = ad::glm::GlmCox<double>;
using glm_gaussian_64_t = ad::glm::GlmGaussian<double>;
using glm_poisson_64_t = ad::glm::GlmPoisson<double>;
using glm_multigaussian_64_t = ad::glm::GlmMultiGaussian<double>;
using glm_multinomial_64_t = ad::glm::GlmMultinomial<double>;

class RGlmBase64: public pimpl<glm_base_64_t>
{
    using base_t = pimpl<glm_base_64_t>;
public:
    using value_t = double;
    using string_t = std::string;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using rowarr_value_t = ad::util::rowarr_type<value_t>;
    using colmat_value_t = ad::util::colmat_type<value_t>;

    using base_t::base_t;

    virtual ~RGlmBase64() {}

    bool is_multi() const { return ptr->is_multi; }
    string_t name() const { return ptr->name; }
    vec_value_t y() const { return ptr->y; }
    vec_value_t weights() const { return ptr->weights; }

    virtual void gradient(
        const Eigen::Map<vec_value_t>& eta,
        Eigen::Map<vec_value_t> grad
    ) 
    {
        ADELIE_CORE_OVERRIDE(gradient, eta, grad);
    }

    virtual void hessian(
        const Eigen::Map<vec_value_t>& eta,
        const Eigen::Map<vec_value_t>& grad,
        Eigen::Map<vec_value_t> hess
    )
    {
        ADELIE_CORE_OVERRIDE(hessian, eta, grad, hess);
    }

    virtual void inv_hessian_gradient(
        const Eigen::Map<vec_value_t>& eta,
        const Eigen::Map<vec_value_t>& grad,
        const Eigen::Map<vec_value_t>& hess,
        Eigen::Map<vec_value_t> inv_hess_grad
    )
    {
        ADELIE_CORE_OVERRIDE(inv_hessian_gradient, eta, grad, hess, inv_hess_grad);
    }

    virtual value_t loss(
        const Eigen::Map<vec_value_t>& eta
    )
    {
        ADELIE_CORE_OVERRIDE(loss, eta);
    }

    virtual value_t loss_full()
    {
        ADELIE_CORE_OVERRIDE(loss_full,);
    }
};

class RGlmMultiBase64: public pimpl<glm_multibase_64_t>
{
    using base_t = pimpl<glm_multibase_64_t>;
public:
    using value_t = double;
    using string_t = std::string;
    using vec_value_t = ad::util::colvec_type<value_t>;
    using rowarr_value_t = ad::util::rowarr_type<value_t>;
    using colmat_value_t = ad::util::colmat_type<value_t>;

    using base_t::base_t;

    virtual ~RGlmMultiBase64() {}

    bool is_multi() const { return ptr->is_multi; }
    string_t name() const { return ptr->name; }
    colmat_value_t y() const { return ptr->y; }
    vec_value_t weights() const { return ptr->weights; }

    virtual void gradient(
        const Eigen::Map<colmat_value_t>& etaT,
        Eigen::Map<colmat_value_t> gradT
    ) 
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        Eigen::Map<rowarr_value_t> grad(gradT.data(), gradT.cols(), gradT.rows());
        ADELIE_CORE_OVERRIDE(gradient, eta, grad);
    }

    virtual void hessian(
        const Eigen::Map<colmat_value_t>& etaT,
        const Eigen::Map<colmat_value_t>& gradT,
        Eigen::Map<colmat_value_t> hessT
    )
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        Eigen::Map<const rowarr_value_t> grad(gradT.data(), gradT.cols(), gradT.rows());
        Eigen::Map<rowarr_value_t> hess(hessT.data(), hessT.cols(), hessT.rows());
        ADELIE_CORE_OVERRIDE(hessian, eta, grad, hess);
    }

    virtual void inv_hessian_gradient(
        const Eigen::Map<colmat_value_t>& etaT,
        const Eigen::Map<colmat_value_t>& gradT,
        const Eigen::Map<colmat_value_t>& hessT,
        Eigen::Map<colmat_value_t> inv_hess_gradT
    )
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        Eigen::Map<const rowarr_value_t> grad(gradT.data(), gradT.cols(), gradT.rows());
        Eigen::Map<const rowarr_value_t> hess(hessT.data(), hessT.cols(), hessT.rows());
        Eigen::Map<rowarr_value_t> inv_hess_grad(inv_hess_gradT.data(), inv_hess_gradT.cols(), inv_hess_gradT.rows());
        ADELIE_CORE_OVERRIDE(inv_hessian_gradient, eta, grad, hess, inv_hess_grad);
    }

    virtual value_t loss(
        const Eigen::Map<colmat_value_t>& etaT
    )
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        ADELIE_CORE_OVERRIDE(loss, eta);
    }

    virtual value_t loss_full()
    {
        ADELIE_CORE_OVERRIDE(loss_full,);
    }
};

ADELIE_CORE_PIMPL_DERIVED(RGlmBinomialLogit64, RGlmBase64, glm_binomial_logit_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmBinomialProbit64, RGlmBase64, glm_binomial_probit_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmCox64, RGlmBase64, glm_cox_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmGaussian64, RGlmBase64, glm_gaussian_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmPoisson64, RGlmBase64, glm_poisson_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmMultiGaussian64, RGlmMultiBase64, glm_multigaussian_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmMultinomial64, RGlmMultiBase64, glm_multinomial_64_t,)

RCPP_EXPOSED_CLASS(RGlmBase64)
RCPP_EXPOSED_CLASS(RGlmMultiBase64)
RCPP_EXPOSED_CLASS(RGlmBinomialLogit64)
RCPP_EXPOSED_CLASS(RGlmBinomialProbit64)
RCPP_EXPOSED_CLASS(RGlmCox64)
RCPP_EXPOSED_CLASS(RGlmGaussian64)
RCPP_EXPOSED_CLASS(RGlmPoisson64)
RCPP_EXPOSED_CLASS(RGlmMultiGaussian64)
RCPP_EXPOSED_CLASS(RGlmMultinomial64)

using r_glm_base_64_t = RGlmBase64;
using r_glm_multibase_64_t = RGlmMultiBase64;
using r_glm_binomial_logit_64_t = RGlmBinomialLogit64;
using r_glm_binomial_probit_64_t = RGlmBinomialProbit64;
using r_glm_cox_64_t = RGlmCox64;
using r_glm_gaussian_64_t = RGlmGaussian64;
using r_glm_poisson_64_t = RGlmPoisson64;
using r_glm_multigaussian_64_t = RGlmMultiGaussian64;
using r_glm_multinomial_64_t = RGlmMultinomial64;
