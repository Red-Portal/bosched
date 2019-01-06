
#ifndef _BOSCHED_LOOP_STATE_HPP_
#define _BOSCHED_LOOP_STATE_HPP_

#include "LPBO/GP.hpp"
#include "LPBO/LPBO.hpp"

#include <atomic>
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
        double param;
        bosched::time_point_t start;
        size_t num_tasks;
        std::vector<double> obs_y;
        std::vector<double> obs_x;
        bool warming_up;
        size_t iteration;
        std::atomic<double> mean_us;
        std::vector<double> pred_mean;
        std::vector<double> pred_var;
        std::vector<double> pred_acq;
        std::optional<lpbo::smc_gp> gp;

        inline loop_state_t() = default;

        inline loop_state_t(loop_state_t&& other)
            : param(other.param),
              start(std::move(other.start)),
              obs_y(std::move(other.obs_y)),
              obs_x(std::move(other.obs_x)),
              warming_up(other.warming_up),
              iteration(other.iteration),
              mean_us(other.mean_us.load()),
              pred_mean(std::move(other.pred_mean)),
              pred_var(std::move(other.pred_var)),
              pred_acq(std::move(other.pred_acq)),
              gp(std::move(other.gp))
        {}

        inline loop_state_t&
        operator=(loop_state_t&& other)
        {
            param      = other.param;
            start      = std::move(other.start);
            obs_y      = std::move(other.obs_y);
            obs_x      = std::move(other.obs_x);
            warming_up = other.warming_up;
            iteration  = other.iteration;
            mean_us.store(other.mean_us.load());
            pred_mean  = std::move(other.pred_mean);
            pred_var   = std::move(other.pred_var);
            pred_acq   = std::move(other.pred_acq);
            gp         = std::move(other.gp);
        }

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
