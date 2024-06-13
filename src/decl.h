#pragma once
// Ignore all warnings for pybind + Eigen
#if defined(_MSC_VER)
#pragma warning( push, 0 )
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall" 
#endif
#include <Rcpp.h>
#include <RcppEigen.h>
#if defined(_MSC_VER)
#pragma warning( pop )
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace adelie_core {}
namespace ad = adelie_core;