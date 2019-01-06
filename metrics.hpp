
#ifndef _BOSCHED_METRICS_HPP_
#define _BOSCHED_METRICS_HPP_

#include <chrono>

namespace bosched
{
    using clock = std::chrono::steady_clock;
    using time_point_t = clock::time_point;
    using duration_t = clock::duration;

    using microsecond = std::chrono::duration<double, std::micro>;
    using millisecond = std::chrono::duration<double, std::milli>;
}

#endif
