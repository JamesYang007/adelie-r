#pragma once
#include <memory>

#ifndef ADELIE_CORE_S4_OVERRIDE
#define ADELIE_CORE_S4_OVERRIDE(name, object, ...) \
    do { \
        Rcpp::Environment global = Rcpp::Environment::global_env(); \
        if (global.exists(#name)) { \
            Rcpp::Function f = global[#name]; \
            return f(object, __VA_ARGS__); \
        } \
    } while (false)
#endif

#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments" 
#endif
#ifndef ADELIE_CORE_S4_PURE_OVERRIDE
#define ADELIE_CORE_S4_PURE_OVERRIDE(name, object, ...) \
    [&]() { \
        Rcpp::Environment global = Rcpp::Environment::global_env(); \
        Rcpp::Function f = global[#name]; \
        return f(object, ## __VA_ARGS__); \
    }()
#endif
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifndef ADELIE_CORE_PIMPL_OVERRIDE
#define ADELIE_CORE_PIMPL_OVERRIDE(name, ...) \
    do { \
        if (!ptr) { \
            Rcpp::stop("Object uninitialized!"); \
        } \
        return ptr->name(__VA_ARGS__); \
    } while (false)
#endif

#ifndef ADELIE_CORE_PIMPL_DERIVED
#define ADELIE_CORE_PIMPL_DERIVED(name, bname, aname, api) \
    class name: public bname \
    { \
    public: \
        template <class... Args> \
        name(Args&& ...args): \
            bname(std::make_shared<aname>(std::forward<Args>(args)...)) \
        {} \
        api \
    };
#endif

#ifndef ADELIE_CORE_PIMPL_MEMBER
#define ADELIE_CORE_PIMPL_MEMBER(cname, fname, name) \
    auto fname(cname* obj) { return obj->ptr->name; }
#endif

template <class T>
class pimpl
{
public:
    std::shared_ptr<T> ptr;

    pimpl(): ptr(nullptr) {}
    pimpl(const std::shared_ptr<T>& ptr): ptr(ptr) {}
    pimpl(std::shared_ptr<T>&& ptr): ptr(std::move(ptr)) {}
};