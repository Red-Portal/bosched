
#ifndef _LPBO_SMC_HPP_
#define _LPBO_SMC_HPP_

#include <vector>
#include "GP.hpp"

namespace lpbo
{

    class sequential_monte_carlo
    {
        size_t _n;
        std::vector<gp_model> _particles;

        inline
        sequential_monte_carlo(size_t particle_num,
                               lpbo::vec const& x,
                               lpbo::vec const& y)
            : _n(particle_num), _particles()
        {
            _particles.reserve(_n);
            for(size_t i = 0; i < _n; ++i)
            {
                _particles.emplace_back(x, y);
            }
        }
        
    };
}

#endif
