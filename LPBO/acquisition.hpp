
#ifndef _LPBO_ACQUISITION_HPP_
#define _LPBO_ACQUISITION_HPP_

#include <cstdlib>
#include <cmath>

namespace lpbo
{
    inline double UCB(double mean,
                      double var,
                      double beta,
                      double annealing,
                      size_t iteration)
    {
        return mean - (beta / (annealing * iteration)) * var;
    }
}

#endif
