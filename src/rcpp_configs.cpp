#include "decl.h"
#include <adelie_core/configs.hpp>

using configs_t = ad::Configs;
using r_configs_t = configs_t;

std::string get_pb_symbol_def(r_configs_t* configs)
{
    return configs->pb_symbol_def;
}

double get_hessian_min_def(r_configs_t* configs)
{
    return configs->hessian_min_def;
}

double get_dbeta_tol_def(r_configs_t* configs)
{
    return configs->dbeta_tol_def;
}

size_t get_min_bytes_def(r_configs_t* configs)
{
    return configs->min_bytes_def;
}

std::string get_pb_symbol(r_configs_t* configs)
{
    return configs->pb_symbol;
}

double get_hessian_min(r_configs_t* configs)
{
    return configs->hessian_min;
}

double get_dbeta_tol(r_configs_t* configs)
{
    return configs->dbeta_tol;
}

size_t get_min_bytes(r_configs_t* configs)
{
    return configs->min_bytes;
}

void set_pb_symbol(r_configs_t* configs, std::string pb_symbol)
{
    configs->pb_symbol = pb_symbol;
}

void set_hessian_min(r_configs_t* configs, double hessian_min)
{
    configs->hessian_min = hessian_min;
}

void set_dbeta_tol(r_configs_t* configs, double dbeta_tol)
{
    configs->dbeta_tol = dbeta_tol;
}

void set_min_bytes(r_configs_t* configs, size_t min_bytes)
{
    configs->min_bytes = min_bytes;
}

RCPP_MODULE(adelie_core_configs)
{
    Rcpp::class_<r_configs_t>("RConfigs")
        .constructor()
        .property("pb_symbol_def", &get_pb_symbol_def, "")
        .property("hessian_min_def", &get_hessian_min_def, "")
        .property("dbeta_tol_def", &get_dbeta_tol_def, "")
        .property("min_bytes_def", &get_min_bytes_def, "")
        .property("pb_symbol", &get_pb_symbol, &set_pb_symbol, "")
        .property("hessian_min", &get_hessian_min, &set_hessian_min, "")
        .property("dbeta_tol", &get_dbeta_tol, &set_dbeta_tol, "")
        .property("min_bytes", &get_min_bytes, &set_min_bytes, "")
        ;
}