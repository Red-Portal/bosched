
#ifndef _LPBO_LPBO_HPP_
#define _LPBO_LPBO_HPP_

#include "GP.hpp"
#include "SMC.hpp"
#include "acquisition.hpp"

#include "cma-es/cmaes.h"
#include <vector>

namespace cmaes
{
    template<typename F>
    inline double
    optimize(F f, double min, double max, size_t max_iter)
    {
        auto is_feasible = [min, max](double x){
                               return min < x && x < max;
                           };

        CMAES<double> evo;
        double* arFunvals;
        double* const* pop;

        double xstart = 0.5;
        double stddev = 1.0;

        Parameters<double> parameters;
        parameters.init(1, &xstart, &stddev);
        parameters.stopMaxIter = max_iter;
        parameters.stopTolFun = 1e-7;
        parameters.lambda = 512;
    
        arFunvals = evo.init(parameters);

        while(!evo.testForTermination())
        {
            pop = evo.samplePopulation();
            for (size_t i = 0; i < evo.get(CMAES<double>::PopSize); ++i)
            {
                while (!is_feasible(*pop[i]))
                    evo.reSampleSingle(i);
            }
            for (size_t i = 0; i < evo.get(CMAES<double>::Lambda); ++i)
                arFunvals[i] = f(*pop[i]);
            evo.updateDistribution(arFunvals);
        }

        double* xfinal = evo.getNew(CMAES<double>::XMean);
        double result = xfinal[0];
        delete[] xfinal;
        return result;
    }
}

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

        auto result = cmaes::optimize(
            f, epsilon, 1.0, max_iter);

        auto [m, v] = model.predict(result);
        double cb = lpbo::UCB(m, v, iter);
        return {result, m, v, cb};
    }

    inline std::tuple<double, double, double>
    find_best_mean(lpbo::smc_gp const& model,
                   double epsilon,
                   int max_iter) noexcept
    {
        auto f = [&](double x){
                     auto [mean, var] = model.predict(x);
                     return mean;
                 };

        auto result = cmaes::optimize(
            f, epsilon, 1.0, max_iter);

        auto [m, v] = model.predict(result);
        return {result, m, v};
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
