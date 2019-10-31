
#ifndef _BOSCHED_LOOP_STATE_HPP_
#define _BOSCHED_LOOP_STATE_HPP_

#include "metrics.hpp"

#include <atomic>
#include <vector>
#include <optional>

namespace bosched
{
    struct loop_state_t
    {
        bosched::time_point_t start;
        double param;
        double eval_param;
        size_t num_tasks;
        std::vector<double> obs_y;
        std::vector<double> obs_x;
        bool warming_up;
        //std::atomic<double> mean_us;
        std::vector<double> gmm_weight;
        std::vector<double> gmm_mean;
        std::vector<double> gmm_stddev;

        inline bosched::time_point_t
        loop_start() noexcept
        {
            start = bosched::clock::now();
            return start;
        }

        template<typename Duration>
        inline Duration
        loop_stop() const noexcept
        {
            return std::chrono::duration_cast<Duration>
                (bosched::clock::now() - start);
        }
    };
}

#endif
