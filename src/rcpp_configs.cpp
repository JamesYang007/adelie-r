#include <Rcpp.h>
#include <adelie_core/configs.hpp>

namespace ad = adelie_core;

using configs_t = ad::Configs;

double get_hessian_min_def(configs_t* configs)
{
    return configs->hessian_min_def;
}

std::string get_pb_symbol_def(configs_t* configs)
{
    return configs->pb_symbol_def;
}

auto get_hessian_min(configs_t* configs)
{
    return configs->hessian_min;
}

auto get_pb_symbol(configs_t* configs)
{
    return configs->pb_symbol;
}

void set_hessian_min(configs_t* configs, double hessian_min)
{
    configs->hessian_min = hessian_min;
}

void set_pb_symbol(configs_t* configs, std::string pb_symbol)
{
    configs->pb_symbol = pb_symbol;
}

RCPP_MODULE(adelie_core_configs)
{
    Rcpp::class_<configs_t>("Configs")
        .constructor()
        .property("hessian_min_def", &get_hessian_min_def, "")
        .property("pb_symbol_def", &get_pb_symbol_def, "")
        .property("hessian_min", &get_hessian_min, &set_hessian_min, "")
        .property("pb_symbol", &get_pb_symbol, &set_pb_symbol, "")
        ;
}