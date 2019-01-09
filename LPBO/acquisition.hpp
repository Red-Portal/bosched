
#ifndef _LPBO_ACQUISITION_HPP_
#define _LPBO_ACQUISITION_HPP_

#include <cstdlib>
#include <cmath>

namespace lpbo
{
    inline double UCB(double mean,
                      double var,
                      size_t iteration)
    {
        double beta = log(iteration * iteration * 3.14 * 3.14 / 0.6);
        return mean - sqrt(beta) * var;
    }
}

#endif
