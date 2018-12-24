
#ifndef _LPBO_SMC_HPP_
#define _LPBO_SMC_HPP_

#include <vector>
#include "GP.hpp"

namespace lpbo
{

    class smc_gp
    {
        size_t _n;
        std::vector<gp_model> _particles;

    private:
        inline void propagate()
        {
            
        }

    public:

        inline
        smc_gp(size_t particle_num,
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

        inline void
        update(lpbo::vec const& x, lpbo::vec const& y)
        {
            
        }

        inline void
        update(double x, double y)
        {
            
        }

        inline std::pair<double, double>
        predict(double x)
        {
            
        }
    };
}

#endif
