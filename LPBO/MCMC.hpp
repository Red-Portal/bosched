
#ifndef _LPBO_MCMC_HPP_
#define _LPBO_MCMC_HPP_

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

namespace lpbo
{
    double normal_pdf(double x, double mean, double var) noexcept
    {
        double s= (x - mean) / sqrt(var);
        return 1 / sqrt(2 * 3.14159265359 * var) * exp(s * s / -2);
    }

    template<typename Type>
    double log_exp_pdf(Type x, double lambda) noexcept
    {
        auto temp = lambda * x;
        return 2 * log(lambda) - temp[0] - temp[1];
    }

    template<typename F, typename Param>
    Param
    metropolis_hastings(F f, Param&& initial, size_t iterations)
    {
        std::random_device seed{}; 
        std::mt19937 rng(seed());
        double randomwalk_variance = 0.1;
        double exp_prior_lambda = 2;

        auto propose =
            [randomwalk_variance](std::mt19937& rand, Param means){
                auto prop = Param(2);
                auto dist1 = std::normal_distribution<double>(means[0], randomwalk_variance);   
                auto dist2 = std::normal_distribution<double>(means[1], randomwalk_variance);   
                prop[0] = dist1(rand);
                prop[1] = dist2(rand);
                return prop;
            };

        auto decide = [](std::mt19937& rand, double prob){
                          auto dist = std::bernoulli_distribution(prob);
                          return dist(rand);
                      };

        Param theta = initial;
        double prev_like = f(theta);
        double prev_prior = log_exp_pdf(theta, exp_prior_lambda);
        Param sum = Param(2);

        for(size_t k = 0; k < iterations; ++k)
        {
            Param proposed = propose(rng, theta);
            if(proposed[0] <= 0 || proposed[1] <= 0)
                continue;

            double like = f(proposed);
            double log_prior = log_exp_pdf(proposed, exp_prior_lambda);

            double rate = exp((like + log_prior) - (prev_like + prev_prior));
            double alpha = std::min(1.0, rate);
            if(decide(rng, alpha))
            {
                theta = proposed;
                prev_like = like;
                prev_prior = log_prior;
                sum += theta;
            }
        }
        return sum / iterations;
    }
}

#endif
