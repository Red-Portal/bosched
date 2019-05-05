
#include "metrics.hpp"
#include <chrono>
#include <omp.h>
#include "profile.hpp"

namespace prof
{
    using iter_id = size_t;
    std::vector<std::pair<iter_id, bosched::clock::time_point>> _timestamp;
    std::vector<float>  _loads;

    void profiling_init(size_t num_tasks)
    {
        _timestamp = std::vector<std::pair<iter_id, bosched::clock::time_point>>(
            omp_get_max_threads());
        _loads = std::vector<float>(num_tasks);
    }

    void iteration_profile_start(size_t iter)
    {
        size_t tid = omp_get_thread_num();
        _timestamp[tid] = std::make_pair(static_cast<iter_id>(iter),
                                         bosched::clock::now());
    }

    void iteration_profile_stop()
    {
        auto stop_stamp = bosched::clock::now();
        size_t tid      = omp_get_thread_num();
        auto [iter, start_stamp] = _timestamp[tid];
        _loads[iter] = bosched::microsecond(stop_stamp - start_stamp).count();
    }

    std::vector<float>
    load_profile()
    {
        return _loads;
    }
}
