
#ifndef _LPBO_LPBO_HPP_
#define _LPBO_LPBO_HPP_

#include "GP.hpp"
#include "SMC.hpp"

#include <vector>

namespace lpbo
{
    inline std::tuple<double, double, double>
    bayesian_optimiztion(lpbo::smc_gp const& model,
                         int iter,
                         int max_iter) noexcept
    {
        return {0, 0, 0};
    }

    inline std::vector<double>
    render_acquisition(lpbo::smc_gp const& model,
                       size_t resolution) noexcept
    {
        return {};
    }

    using mean_t = double;
    using var_t = double;

    inline std::pair<std::vector<mean_t>,
                     std::vector<var_t>>
    render_gp(lpbo::smc_gp const& model,
              size_t resolution) noexcept
    {
        return {};
    }
}

#endif
