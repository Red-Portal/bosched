

#ifndef _PERFORMANCE_HPP_
#define _PERFORMANCE_HPP_

#include <algorithm> 
#include <cmath> 
#include <numeric> 
#include <vector> 

namespace bosched
{
    inline double
    coeff_of_variation(std::vector<double> const& durations)
    {
        double mean = std::accumulate(
            durations.begin(), durations.end(), 0.0,
            [](double sum, double duration){
                return sum + duration;
            }) / durations.size();

        double second_moment = std::accumulate(
            durations.begin(), durations.end(), 0.0,
            [mean](double sum, double duration){
                auto diff = duration - mean;
                return sum + diff * diff;
            }) / (durations.size() + 1);
        double sigma = sqrt(second_moment);
        return sigma / mean;
    }

    inline double
    slowdown(std::vector<double> const& durations)
    {
        double slow = *std::max_element(durations.begin(), durations.end());
        double fast = *std::min_element(durations.begin(), durations.end());
        return slow / fast;
    }
}

#endif
