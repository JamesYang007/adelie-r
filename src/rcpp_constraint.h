#pragma once
#include "decl.h"
#include "utils.h"
#include <adelie_core/constraint/constraint_base.hpp>

// TODO: fill core aliases
using constraint_base_64_t = ad::constraint::ConstraintBase<double>;

class RConstraintBase64: public pimpl<constraint_base_64_t>
{
    using base_t = pimpl<constraint_base_64_t>;
public:
    using base_t::base_t;

    // TODO: fill
    size_t duals() const 
    { 
        ADELIE_CORE_PIMPL_OVERRIDE(duals,);
    }
};

// TODO: fill ADELIE_CORE_PIMPL_DERIVED

// TODO: fill RCPP_EXPOSED_CLASS
RCPP_EXPOSED_CLASS(RConstraintBase64)

// TODO: fill r aliases
using r_constraint_base_64_t = RConstraintBase64;