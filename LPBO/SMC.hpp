
#ifndef _LPBO_SMC_HPP_
#define _LPBO_SMC_HPP_

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
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
        lpbo::vec _data_x;
        lpbo::vec _data_y;

        template<typename Rng>
        inline std::vector<size_t>
        selection(Rng&& rng, std::vector<double> weights) const
        {
            size_t num_casualties = static_cast<size_t>(
                floor(weights.size() * (1 - _survival_rate)));
            auto casualties = std::vector<size_t>(num_casualties);
            std::transform(weights.begin(), weights.end(), weights.begin(),
                           [](double x){
                               return 1 - x;
                           });

            for(size_t i = 0; i < num_casualties; ++i)
            {
                auto dist = std::discrete_distribution<size_t>(
                    weights.begin(), weights.end());
                size_t casualty = dist(rng);
                casualties[i] = casualty;
                weights[i] = 0;
            }
            return casualties;
        }

        template<typename Rng>
        inline size_t
        rejuvenate_select(Rng&& rng, std::vector<double> const& weights) const
        {
            auto dist = std::discrete_distribution<size_t>(
                weights.cbegin(), weights.cend());
            return dist(rng);
        }

        inline std::vector<double>
        normalize_weight(std::vector<double>&& weights) const noexcept
        {
            auto max_ll = *std::max_element(weights.begin(), weights.end());
            auto sum = std::accumulate(weights.begin(), weights.end(), 0.0,
                                       [=](double sum, double loglikelihood){
                                           return sum + exp(loglikelihood - max_ll);
                                       });
            std::transform(weights.begin(), weights.end(), weights.begin(),
                           [sum, max_ll](double loglikelihood){
                               return exp(loglikelihood - max_ll) / sum;
                           });
            return std::move(weights);
        }
        
    public:
        inline
        smc_gp(
            lpbo::vec const& x,
            lpbo::vec const& y,
            size_t particle_num,
            double survival_rate)
            : _survival_rate(survival_rate), _particles(),
              _weights(), _rng(), _data_x(x), _data_y(y)
        {
            _particles.reserve(particle_num);
            for(size_t i = 0; i < particle_num; ++i)
                _particles.emplace_back(x, y);

            auto weights = std::vector<double>(particle_num);
            for(size_t i = 0; i < particle_num; ++i)
                weights[i] = exp(_particles[i].likelihood());

            _weights = normalize_weight(std::move(weights));
        }

        inline
        smc_gp(
            std::vector<double> const& x,
            std::vector<double> const& y,
            size_t particle_num,
            double survival_rate)
            : _survival_rate(survival_rate), _particles(),
              _weights(), _rng(),
              _data_x(x.size(), x.data()),
              _data_y(y.size(), y.data())
        {
            _particles.reserve(particle_num);
            for(size_t i = 0; i < particle_num; ++i)
                _particles.emplace_back(_data_x, _data_y);

            auto weights = std::vector<double>(particle_num);
            for(size_t i = 0; i < particle_num; ++i)
                weights[i] = exp(_particles[i].likelihood());

            _weights = normalize_weight(std::move(weights));
        }

        inline
        smc_gp(std::string const& serialized)
            : _survival_rate(),
              _weights(),
              _rng()

        {
            deserialize(serialized);
        }

        inline
        smc_gp( smc_gp const& other) = default;

        inline
        smc_gp( smc_gp&& other) noexcept = default;

        inline smc_gp&
        operator=(smc_gp const& other) = default;

        inline smc_gp&
        operator=( smc_gp&& other) noexcept = default;

        inline void
        update(lpbo::vec const& x, lpbo::vec const& y)
        {
            auto unnormalized_weights = std::vector<double>(_particles.size());
            for(size_t i = 0; i < _particles.size(); ++i)
                unnormalized_weights[i] = exp(_particles[i].likelihood());

            auto weights = normalize_weight(std::move(unnormalized_weights));
            auto casualties = selection(_rng, weights);

            std::for_each(casualties.begin(), casualties.end(),
                          [&](size_t i){
                              weights[i] = 0;
                          });

            for(auto i : casualties)
            {
                size_t sel = rejuvenate_select(_rng, weights);
                _particles[i].rejuvenate(_data_x,
                                         _data_y,
                                         _particles[sel].parameters(),
                                         200);
            }

            size_t old_n = _data_x.size();
            size_t add_n = x.size();
            _data_x.resize(old_n + add_n);
            blaze::subvector(_data_x, old_n, add_n) = x;

            _data_y.resize(old_n + add_n);
            blaze::subvector(_data_y, old_n, add_n) = y;

            for(auto& particle : _particles)
            {
                try
                {
                    particle.update(_data_x, y);
                }
                catch(std::exception const& )
                {
                    size_t sel = rejuvenate_select(_rng, weights);
                    particle.rejuvenate(_data_x,
                                        _data_y,
                                        _particles[sel].parameters(),
                                        200);
                }
            }

            for(size_t i = 0; i < _particles.size(); ++i)
                weights[i] = exp(_particles[i].likelihood());

            _weights = normalize_weight(std::move(weights));
        }

        inline void
        update(double x, double y)
        {
            update(lpbo::vec({x}), lpbo::vec({y}));
        }

        inline void
        update(std::vector<double> const& x,
               std::vector<double> const& y)
        {
            update(lpbo::vec(x.size(), x.data()),
                   lpbo::vec(y.size(), y.data()));
        }

        inline std::string
        serialize() const
        {
            std::stringstream stream;
            stream << std::to_string(_survival_rate) + '\n';
            stream << std::to_string(static_cast<unsigned long long>(_particles.size())) + '\n';

            auto archive = blaze::Archive<std::stringstream>(stream);
            archive << _data_x;
            archive << _data_y;

            for(size_t i = 0; i < _particles.size(); ++i)
            {
                _particles[i] >> stream;
            }
            return stream.str();
        }

        inline void
        deserialize(std::string const& serialized)
        {
            if(_particles.size() != 0)
            {
                _particles.clear();
                _weights.clear();
                _survival_rate = 0;
            }

            std::stringstream stream;
            stream << serialized;

            std::string buf;
            std::getline(stream, buf);
            _survival_rate = std::stod(buf);

            std::getline(stream, buf);
            size_t particle_num = std::stoull(buf);

            auto archive = blaze::Archive<std::stringstream>(stream);
            archive >> _data_x;
            archive >> _data_y;

            _particles.reserve(particle_num);
            for(size_t i = 0; i < particle_num; ++i)
            {
                _particles.emplace_back();
                _particles[i] << stream;
            }

            auto weights = std::vector<double>(particle_num);
            for(size_t i = 0; i < particle_num; ++i)
                weights[i] = exp(_particles[i].likelihood());

            _weights = normalize_weight(std::move(weights));
        }

        inline lpbo::vec& 
        data_x()
        {
            return _data_x;
        }

        inline lpbo::vec&
        data_y()
        {
            return _data_y;
        }

        inline std::pair<double, double>
        predict(double x) const
        {
            double mean = 0.0;
            double var  = 0.0;
            for(size_t i = 0 ; i < _particles.size(); ++i)
            {
                auto local_prediction = _particles[i].predict(_data_x, x);
                mean += local_prediction.first * _weights[i] ;
                var  += local_prediction.second * _weights[i];
            }
            return {mean, var};
        }
    };
}

#endif
