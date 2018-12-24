
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

    template<typename Type>
    inline 
    double proposal_likelihood(Type x, Type prev)
    {
        double u0 = 4 * prev[0] / 3;
        double l0 = 3 * prev[0] / 4;
        double u1 = 4 * prev[1] / 3;
        double l1 = 3 * prev[1] / 4;
        
        double ind = static_cast<double>((x[0] <= u0 && x[0] >= l0) && (x[1] <= u1 && x[1] >= l1));
        double p = 1 / ((u0 - l0) * (u1 - l1));
        return  ind * p;
    }

    template<typename F, typename Param>
    Param
    metropolis_hastings(F f, Param&& initial, size_t iterations)
    {
        std::random_device seed{}; 
        std::mt19937 rng(seed());
        //double randomwalk_variance = 2.8322;
        double exp_prior_lambda = 2;

        auto propose =
            [](std::mt19937& rand, Param means){
                auto prop = Param(2);
                // auto dist = std::normal_distribution<double>(0, randomwalk_variance);   
                // prop[0] = means[0] + dist(rand);
                // prop[1] = means[1] + dist(rand);
                auto dist0 = std::uniform_real_distribution<double>(3 * means[0] / 4, 4 * means[0] / 3);   
                auto dist1 = std::uniform_real_distribution<double>(3 * means[1] / 4, 4 * means[1] / 3);   
                prop[0] = dist0(rand);
                prop[1] = dist1(rand);
                return prop;
            };

        auto decide = [](std::mt19937& rand, double prob){
                          auto dist = std::bernoulli_distribution(prob);
                          return dist(rand);
                      };

        Param theta = initial;
        double prev_like = f(theta);
        double prev_q_like = proposal_likelihood(theta, theta);
        double prev_prior = log_exp_pdf(theta, exp_prior_lambda);
        Param sum = Param(2);

        //size_t avg_rate = 0;

        for(size_t k = 0; k < iterations; ++k)
        {
            Param proposed = propose(rng, theta);
            if(proposed[0] <= 0 || proposed[1] <= 0)
                continue;

            double like = f(proposed);
            double q_like = proposal_likelihood(proposed, theta);
            double log_prior = log_exp_pdf(proposed, exp_prior_lambda);

            double rate = exp((like + log_prior) - (prev_like + prev_prior)) * q_like / prev_q_like;
            //double rate = exp(like - prev_like);// * q_like / prev_q_like;
            double alpha = std::min(1.0, rate);

            // std::cout << "proposed  : \n" << proposed << '\n'
            //           << "prev      : \n" << theta << '\n'
            //           << "alpha     : " << alpha << '\n'
            //           << "like      : " << like << '\n'
            //           << "prior     : " << log_prior << '\n'
            //           << "prev_like : " << prev_like << '\n'
            //     //<< "prev prior: " << prev_prior << '\n'
            //           << std::endl;

            if(decide(rng, alpha))
            {
                theta = proposed;
                prev_like = like;
                prev_prior = log_prior;
                prev_q_like = q_like;

                if(k > 0.25 * iterations)
                {
                    //++avg_rate;
                    sum += theta;
                }
            }
        }
        //std::cout << static_cast<double>(avg_rate) / iterations << std::endl;
        return sum / iterations;
    }
}

#endif
