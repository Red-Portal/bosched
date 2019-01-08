
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
        (void)(beta);
        (void)(annealing);
        //return mean - (beta / (annealing * iteration)) * var;
        return mean - sqrt(log(iteration * iteration * 3.14 * 3.14 / 0.6)) * var;
    }
}

#endif
