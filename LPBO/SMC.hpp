
#ifndef _LPBO_SMC_HPP_
#define _LPBO_SMC_HPP_

#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include "GP.hpp"

namespace lpbo
{

    class smc_gp
    {
    private:
        double _survival_rate;
        std::vector<gp_model> _particles;
        std::vector<double> _weights;
        std::mt19937 _rng;

        template<typename Rng>
        std::vector<size_t>
        selection(Rng&& rng, std::vector<double>& weights) const
        {
            size_t num_casualties = static_cast<size_t>(
                floor(weights.size() * (1 - _survival_rate)));
            auto casualties = std::vector<size_t>(num_casualties);
            auto anti_weights = std::vector<double>(weights.size());
            std::transform(weights.begin(), weights.end(), anti_weights.begin(),
                           [](double x){
                               return 1 - x;
                           });
            for(size_t i = 0; i < num_casualties; ++i)
            {
                auto dist = std::discrete_distribution<size_t>(
                    anti_weights.begin(), anti_weights.end());
                size_t casualty = dist(rng);
                casualties[i] = casualty;
                anti_weights[i] = 0;
                weights[i] = 0;
            }
            return casualties;
        }

        template<typename Rng>
        size_t rejuvenate_select(Rng&& rng, std::vector<double> weights) const
        {
            auto dist = std::discrete_distribution<size_t>(
                weights.begin(), weights.end());
            return dist(rng);
        }

        inline void
        update_weights()
        {
            auto weights = std::vector<double>(_particles.size());
            for(size_t i = 0; i < _particles.size(); ++i)
                weights[i] = exp(_particles[i].likelihood());

            auto sum = std::accumulate(weights.begin(), weights.end(), 0.0);
            std::transform(weights.begin(), weights.end(), weights.begin(),
                           [sum](double x){ return x / sum; });
            _weights = std::move(weights);
        }
        
    public:
        inline
        smc_gp(
            lpbo::vec const& x,
            lpbo::vec const& y,
            size_t particle_num,
            double survival_rate)
            : _survival_rate(survival_rate), _particles(),
              _weights(), _rng()
        {
            _particles.reserve(particle_num);
            for(size_t i = 0; i < particle_num; ++i)
            {
                _particles.emplace_back(x, y);
            }
            update_weights();
        }

        inline void
        update(lpbo::vec const& x, lpbo::vec const& y)
        {
            auto weights = std::vector<double>(_particles.size());
            for(size_t i = 0; i < _particles.size(); ++i)
                weights[i] = exp(_particles[i].likelihood());

            auto casualties = selection(_rng, weights);

            for(auto i : casualties)
            {
                size_t sel = rejuvenate_select(_rng, weights);
                _particles[i].rejuvenate(x, y, _particles[sel].parameters(), 10);
            }

            for(auto& particle : _particles)
            {
                particle.update(x, y);
            }
            update_weights();

            double l = 0.0;
            double g = 0.0;
            for(size_t i = 0; i < _weights.size(); ++i)
            {
                auto param = _particles[i].parameters();
                l += param[0] * _weights[i];
                g += param[1] * _weights[i];
            }
            std::cout << "avg l: " << l << " avg g: " << g << std::endl;
        }

        inline void
        update(double x, double y)
        {
            update(lpbo::vec({x}), lpbo::vec({y}));
        }

        inline std::pair<double, double>
        predict(double x) const
        {
            double mean = 0.0;
            double var  = 0.0;
            for(size_t i = 0 ; i < _particles.size(); ++i)
            {
                auto local_prediction = _particles[i].predict(x);
                mean += (local_prediction.first * _weights[i]) ;
                var  += (local_prediction.second * _weights[i]);
            }
            return {mean, var};
        }
    };
}

#endif
