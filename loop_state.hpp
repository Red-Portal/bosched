
#ifndef _BOSCHED_LOOP_STATE_HPP_
#define _BOSCHED_LOOP_STATE_HPP_

#include "LPBO/GP.hpp"
#include "LPBO/LPBO.hpp"

#include <vector>
#include <optional>

namespace bosched
{
    using clock = std::chrono::steady_clock;
    using time_point_t = clock::time_point;
    using duration_t = clock::duration;

    using microsecond = std::chrono::duration<double, std::micro>;
    using millisecond = std::chrono::duration<double, std::milli>;

    struct loop_state_t
    {
        size_t id;
        size_t num_tasks;
        size_t iteration;
        double param;
        bosched::time_point_t start;
        bool warming_up;
        bool is_bo_schedule;
        std::vector<double> mean;
        std::vector<double> var;
        std::vector<double> acq;
        std::optional<lpbo::smc_gp> gp;
        std::vector<double> obs_x;
        std::vector<double> obs_y;

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
