
#ifndef _LPBO_LPBO_HPP_
#define _LPBO_LPBO_HPP_

#include "GP.hpp"
#include "SMC.hpp"
#include "acquisition.hpp"

#include <dlib/global_optimization.h>
#include <vector>

namespace lpbo
{
    inline std::tuple<double, double, double, double>
    bayesian_optimization(lpbo::smc_gp const& model,
                          double epsilon,
                          int iter,
                          int max_iter) noexcept
    {
        auto f = [&](double x){
                     auto [mean, var] = model.predict(x);
                     return lpbo::UCB(mean, var, iter);
                 };

        auto result = dlib::find_min_global(
            f, {epsilon}, {1.0},
            dlib::max_function_calls(max_iter),
            dlib::FOREVER, 1e-2);

        auto [m, v] = model.predict(result.x);
        double cb = lpbo::UCB(m, v, iter);
        return {result.x, m, v, cb};
    }

    inline std::vector<double>
    render_acquisition(lpbo::smc_gp const& model,
                       size_t iter,
                       size_t resolution) noexcept
    {
        auto f = [&](double x){
                     auto [mean, var] = model.predict(x);
                     return lpbo::UCB(mean, var, iter);
                 };

        auto y = std::vector<double>(resolution);
        for(size_t i = 0; i < resolution; ++i)
        {
            y[i] = f(static_cast<double>(i) / resolution);
        }
        return y;
    }

    using mean_t = double;
    using var_t = double;

    inline std::pair<std::vector<mean_t>,
                     std::vector<var_t>>
    render_gp(lpbo::smc_gp const& model, size_t resolution) noexcept
    {
        auto means = std::vector<double>(resolution);
        auto vars  = std::vector<double>(resolution);
        for(size_t i = 0; i < resolution; ++i)
        {
            auto [mean, var] = model.predict(
                static_cast<double>(i) / resolution);
            means[i] = mean;
            vars[i] = var;
        }
        return {means, vars};
    }
}

#endif
