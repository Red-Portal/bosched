
#ifndef _BOSCHED_LOOP_STATE_HPP_
#define _BOSCHED_LOOP_STATE_HPP_

#include "LPBO/GP.hpp"
#include "LPBO/LPBO.hpp"
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
        size_t num_tasks;
        std::vector<double> obs_y;
        std::vector<double> obs_x;
        bool warming_up;
        size_t iteration;
        //std::atomic<double> mean_us;
        std::vector<double> pred_mean;
        std::vector<double> pred_var;
        std::vector<double> pred_acq;
        std::optional<lpbo::smc_gp> gp;

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
