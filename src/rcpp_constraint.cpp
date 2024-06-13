#include "rcpp_constraint.h"

/* Factory functions */

RCPP_MODULE(adelie_core_constraint)
{
    Rcpp::class_<r_constraint_base_64_t>("RConstraintBase64")
        .constructor()
        .property("dual_size", &r_constraint_base_64_t::duals, "")
        ;
}