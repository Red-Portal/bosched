
#include "tls.hpp"
#include "metrics.hpp"

#include <algorithm>
#include <atomic>
#include <boost/thread/tss.hpp>
#include <cmath>
#include <iostream>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace statistic
{
    std::atomic<size_t> _total_runtime;
    std::vector<bosched::clock::time_point> _timestamp;

    bosched::clock::time_point _startstamp;
    std::vector<bosched::clock::time_point> _tailstamp;

    void init_tls()
    {
        _total_runtime.store(size_t(0));
        _startstamp = bosched::clock::now();
        _timestamp  = std::vector<bosched::clock::time_point>(
            omp_get_max_threads(), _startstamp);
        _tailstamp  = std::vector<bosched::clock::time_point>(
            omp_get_max_threads(), _startstamp);
    }

    void iteration_start_record()
    {
        _timestamp[omp_get_thread_num()] = bosched::clock::now();
    }

    void iteration_stop_record()
    {
        auto start_point = _timestamp[omp_get_thread_num()];
        auto curr_point  = bosched::clock::now();
        auto duration    = curr_point - start_point;
        auto discrete    = std::chrono::duration_cast<
            std::chrono::nanoseconds>(duration);
        _total_runtime  += discrete.count();

	std::cout << discrete.count() << '\n';

        _tailstamp[omp_get_thread_num()] = curr_point;
    }

    std::vector<double>
    work_per_processor()
    {
        auto durations = std::vector<double>(_tailstamp.size());
        std::transform(_tailstamp.begin(), _tailstamp.end(), durations.begin(),
                       [](auto timepoint){
                           auto duration = timepoint - _startstamp;
                           return bosched::millisecond(duration).count();
                       });
        return durations;
    }

    bosched::millisecond
    total_work()
    {
        return std::chrono::nanoseconds(_total_runtime.load());
    }
}
