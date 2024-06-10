#pragma once
#include "decl.h"
#include "utils.h"
#include <adelie_core/glm/glm_base.hpp>
#include <adelie_core/glm/glm_binomial.hpp>
#include <adelie_core/glm/glm_cox.hpp>
#include <adelie_core/glm/glm_gaussian.hpp>
#include <adelie_core/glm/glm_multibase.hpp>
#include <adelie_core/glm/glm_multigaussian.hpp>
#include <adelie_core/glm/glm_multinomial.hpp>
#include <adelie_core/glm/glm_poisson.hpp>

namespace adelie_core {
namespace glm {

template <class ValueType>
class GlmS4: public GlmBase<ValueType>
{
    Rcpp::S4 _glm;

public:
    using base_t = GlmBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using colvec_value_t = util::colvec_type<value_t>;

    explicit GlmS4(
        Rcpp::S4 glm,
        const Eigen::Ref<const vec_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("s4", y, weights),
        _glm(glm)
    {}

    void gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        Eigen::Ref<vec_value_t> grad
    ) override
    {
        const Eigen::Map<colvec_value_t> eta_r(const_cast<value_t*>(eta.data()), eta.size());
        Eigen::Map<colvec_value_t> grad_r(grad.data(), grad.size());
        ADELIE_CORE_S4_PURE_OVERRIDE(void, gradient, _glm, eta_r, grad_r);
    }

    void hessian(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        Eigen::Ref<vec_value_t> hess
    ) override
    {
        const Eigen::Map<colvec_value_t> eta_r(const_cast<value_t*>(eta.data()), eta.size());
        const Eigen::Map<colvec_value_t> grad_r(const_cast<value_t*>(grad.data()), grad.size());
        Eigen::Map<colvec_value_t> hess_r(hess.data(), hess.size());
        ADELIE_CORE_S4_PURE_OVERRIDE(void, hessian, _glm, eta_r, grad_r, hess_r);
    }

    void inv_hessian_gradient(
        const Eigen::Ref<const vec_value_t>& eta,
        const Eigen::Ref<const vec_value_t>& grad,
        const Eigen::Ref<const vec_value_t>& hess,
        Eigen::Ref<vec_value_t> inv_hess_grad
    ) override
    {
        const Eigen::Map<colvec_value_t> eta_r(const_cast<value_t*>(eta.data()), eta.size());
        const Eigen::Map<colvec_value_t> grad_r(const_cast<value_t*>(grad.data()), grad.size());
        const Eigen::Map<colvec_value_t> hess_r(const_cast<value_t*>(hess.data()), hess.size());
        Eigen::Map<colvec_value_t> inv_hess_grad_r(inv_hess_grad.data(), inv_hess_grad.size());
        ADELIE_CORE_S4_OVERRIDE(void, inv_hessian_gradient, _glm, eta_r, grad_r, hess_r, inv_hess_grad_r);
        return base_t::inv_hessian_gradient(eta, grad, hess, inv_hess_grad);
    }

    value_t loss(
        const Eigen::Ref<const vec_value_t>& eta
    ) override
    {
        const Eigen::Map<colvec_value_t> eta_r(const_cast<value_t*>(eta.data()), eta.size());
        Rcpp::NumericVector out = [&]() {ADELIE_CORE_S4_PURE_OVERRIDE(value_t, loss, _glm, eta_r);}();
        return out[0];
    }

    value_t loss_full() override 
    {
        Rcpp::NumericVector out = [&]() {ADELIE_CORE_S4_PURE_OVERRIDE(value_t, loss_full, _glm);}();
        return out[0];
    }
};

template <class ValueType>
class GlmMultiS4: public GlmMultiBase<ValueType>
{
    Rcpp::S4 _glm;

public:
    using base_t = GlmMultiBase<ValueType>;
    using typename base_t::value_t;
    using typename base_t::vec_value_t;
    using typename base_t::rowarr_value_t;
    using colarr_value_t = util::colarr_type<value_t>;

    explicit GlmMultiS4(
        Rcpp::S4 glm,
        const Eigen::Ref<const rowarr_value_t>& y,
        const Eigen::Ref<const vec_value_t>& weights
    ):
        base_t("multis4", y, weights, false /* TODO: remove */),
        _glm(glm)
    {}

    void gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        Eigen::Ref<rowarr_value_t> grad
    ) override
    {
        const Eigen::Map<colarr_value_t> etaT_r(const_cast<value_t*>(eta.data()), eta.cols(), eta.rows());
        Eigen::Map<colarr_value_t> gradT_r(grad.data(), grad.cols(), grad.rows());
        ADELIE_CORE_S4_PURE_OVERRIDE(void, gradient, _glm, etaT_r, gradT_r);
    }

    void hessian(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        Eigen::Ref<rowarr_value_t> hess
    ) override
    {
        const Eigen::Map<colarr_value_t> etaT_r(const_cast<value_t*>(eta.data()), eta.cols(), eta.rows());
        const Eigen::Map<colarr_value_t> gradT_r(const_cast<value_t*>(grad.data()), grad.cols(), grad.rows());
        Eigen::Map<colarr_value_t> hessT_r(hess.data(), hess.cols(), hess.rows());
        ADELIE_CORE_S4_PURE_OVERRIDE(void, hessian, _glm, etaT_r, gradT_r, hessT_r);
    }

    void inv_hessian_gradient(
        const Eigen::Ref<const rowarr_value_t>& eta,
        const Eigen::Ref<const rowarr_value_t>& grad,
        const Eigen::Ref<const rowarr_value_t>& hess,
        Eigen::Ref<rowarr_value_t> inv_hess_grad
    ) override
    {
        const Eigen::Map<colarr_value_t> etaT_r(const_cast<value_t*>(eta.data()), eta.cols(), eta.rows());
        const Eigen::Map<colarr_value_t> gradT_r(const_cast<value_t*>(grad.data()), grad.cols(), grad.rows());
        const Eigen::Map<colarr_value_t> hessT_r(const_cast<value_t*>(hess.data()), hess.cols(), hess.rows());
        Eigen::Map<colarr_value_t> inv_hess_gradT_r(inv_hess_grad.data(), inv_hess_grad.cols(), inv_hess_grad.rows());
        ADELIE_CORE_S4_OVERRIDE(void, inv_hessian_gradient, _glm, etaT_r, gradT_r, hessT_r, inv_hess_gradT_r);
        return base_t::inv_hessian_gradient(eta, grad, hess, inv_hess_grad);
    }

    value_t loss(
        const Eigen::Ref<const rowarr_value_t>& eta
    ) override
    {
        const Eigen::Map<colarr_value_t> etaT_r(const_cast<value_t*>(eta.data()), eta.cols(), eta.rows());
        Rcpp::NumericVector out = [&]() {ADELIE_CORE_S4_PURE_OVERRIDE(value_t, loss, _glm, etaT_r);}();
        return out[0];
    }

    value_t loss_full() override 
    {
        Rcpp::NumericVector out = [&]() {ADELIE_CORE_S4_PURE_OVERRIDE(value_t, loss_full, _glm);}();
        return out[0];
    }
};

} // namespace glm
} // namespace adelie_core

using glm_base_64_t = ad::glm::GlmBase<double>;
using glm_multibase_64_t = ad::glm::GlmMultiBase<double>;
using glm_binomial_logit_64_t = ad::glm::GlmBinomialLogit<double>;
using glm_binomial_probit_64_t = ad::glm::GlmBinomialProbit<double>;
using glm_cox_64_t = ad::glm::GlmCox<double>;
using glm_gaussian_64_t = ad::glm::GlmGaussian<double>;
using glm_poisson_64_t = ad::glm::GlmPoisson<double>;
using glm_s4_64_t = ad::glm::GlmS4<double>;
using glm_multigaussian_64_t = ad::glm::GlmMultiGaussian<double>;
using glm_multinomial_64_t = ad::glm::GlmMultinomial<double>;
using glm_multis4_64_t = ad::glm::GlmMultiS4<double>;

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

    bool is_multi() const { return ptr->is_multi; }
    string_t name() const { return ptr->name; }
    vec_value_t y() const { return ptr->y; }
    vec_value_t weights() const { return ptr->weights; }

    void gradient(
        const Eigen::Map<vec_value_t>& eta,
        Eigen::Map<vec_value_t> grad
    ) 
    {
        ADELIE_CORE_PIMPL_OVERRIDE(gradient, eta, grad);
    }

    void hessian(
        const Eigen::Map<vec_value_t>& eta,
        const Eigen::Map<vec_value_t>& grad,
        Eigen::Map<vec_value_t> hess
    )
    {
        ADELIE_CORE_PIMPL_OVERRIDE(hessian, eta, grad, hess);
    }

    void inv_hessian_gradient(
        const Eigen::Map<vec_value_t>& eta,
        const Eigen::Map<vec_value_t>& grad,
        const Eigen::Map<vec_value_t>& hess,
        Eigen::Map<vec_value_t> inv_hess_grad
    )
    {
        ADELIE_CORE_PIMPL_OVERRIDE(inv_hessian_gradient, eta, grad, hess, inv_hess_grad);
    }

    value_t loss(
        const Eigen::Map<vec_value_t>& eta
    )
    {
        ADELIE_CORE_PIMPL_OVERRIDE(loss, eta);
    }

    value_t loss_full()
    {
        ADELIE_CORE_PIMPL_OVERRIDE(loss_full,);
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
    using colarr_value_t = ad::util::colarr_type<value_t>;

    using base_t::base_t;

    bool is_multi() const { return ptr->is_multi; }
    string_t name() const { return ptr->name; }
    colarr_value_t y() const { return ptr->y; }
    vec_value_t weights() const { return ptr->weights; }

    void gradient(
        const Eigen::Map<colarr_value_t>& etaT,
        Eigen::Map<colarr_value_t> gradT
    ) 
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        Eigen::Map<rowarr_value_t> grad(gradT.data(), gradT.cols(), gradT.rows());
        ADELIE_CORE_PIMPL_OVERRIDE(gradient, eta, grad);
    }

    void hessian(
        const Eigen::Map<colarr_value_t>& etaT,
        const Eigen::Map<colarr_value_t>& gradT,
        Eigen::Map<colarr_value_t> hessT
    )
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        Eigen::Map<const rowarr_value_t> grad(gradT.data(), gradT.cols(), gradT.rows());
        Eigen::Map<rowarr_value_t> hess(hessT.data(), hessT.cols(), hessT.rows());
        ADELIE_CORE_PIMPL_OVERRIDE(hessian, eta, grad, hess);
    }

    void inv_hessian_gradient(
        const Eigen::Map<colarr_value_t>& etaT,
        const Eigen::Map<colarr_value_t>& gradT,
        const Eigen::Map<colarr_value_t>& hessT,
        Eigen::Map<colarr_value_t> inv_hess_gradT
    )
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        Eigen::Map<const rowarr_value_t> grad(gradT.data(), gradT.cols(), gradT.rows());
        Eigen::Map<const rowarr_value_t> hess(hessT.data(), hessT.cols(), hessT.rows());
        Eigen::Map<rowarr_value_t> inv_hess_grad(inv_hess_gradT.data(), inv_hess_gradT.cols(), inv_hess_gradT.rows());
        ADELIE_CORE_PIMPL_OVERRIDE(inv_hessian_gradient, eta, grad, hess, inv_hess_grad);
    }

    value_t loss(
        const Eigen::Map<colarr_value_t>& etaT
    )
    {
        Eigen::Map<const rowarr_value_t> eta(etaT.data(), etaT.cols(), etaT.rows());
        ADELIE_CORE_PIMPL_OVERRIDE(loss, eta);
    }

    value_t loss_full()
    {
        ADELIE_CORE_PIMPL_OVERRIDE(loss_full,);
    }
};

ADELIE_CORE_PIMPL_DERIVED(RGlmBinomialLogit64, RGlmBase64, glm_binomial_logit_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmBinomialProbit64, RGlmBase64, glm_binomial_probit_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmCox64, RGlmBase64, glm_cox_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmGaussian64, RGlmBase64, glm_gaussian_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmPoisson64, RGlmBase64, glm_poisson_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmS464, RGlmBase64, glm_s4_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmMultiGaussian64, RGlmMultiBase64, glm_multigaussian_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmMultinomial64, RGlmMultiBase64, glm_multinomial_64_t,)
ADELIE_CORE_PIMPL_DERIVED(RGlmMultiS464, RGlmMultiBase64, glm_multis4_64_t,)

RCPP_EXPOSED_CLASS(RGlmBase64)
RCPP_EXPOSED_CLASS(RGlmMultiBase64)
RCPP_EXPOSED_CLASS(RGlmBinomialLogit64)
RCPP_EXPOSED_CLASS(RGlmBinomialProbit64)
RCPP_EXPOSED_CLASS(RGlmCox64)
RCPP_EXPOSED_CLASS(RGlmGaussian64)
RCPP_EXPOSED_CLASS(RGlmPoisson64)
RCPP_EXPOSED_CLASS(RGlmS464)
RCPP_EXPOSED_CLASS(RGlmMultiGaussian64)
RCPP_EXPOSED_CLASS(RGlmMultinomial64)
RCPP_EXPOSED_CLASS(RGlmMultiS464)

using r_glm_base_64_t = RGlmBase64;
using r_glm_multibase_64_t = RGlmMultiBase64;
using r_glm_binomial_logit_64_t = RGlmBinomialLogit64;
using r_glm_binomial_probit_64_t = RGlmBinomialProbit64;
using r_glm_cox_64_t = RGlmCox64;
using r_glm_gaussian_64_t = RGlmGaussian64;
using r_glm_poisson_64_t = RGlmPoisson64;
using r_glm_s4_64_t = RGlmS464;
using r_glm_multigaussian_64_t = RGlmMultiGaussian64;
using r_glm_multinomial_64_t = RGlmMultinomial64;
using r_glm_multis4_64_t = RGlmMultiS464;
