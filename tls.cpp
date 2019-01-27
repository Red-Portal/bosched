
#include "tls.hpp"
#include "metrics.hpp"

#include <iostream>
#include <atomic>
#include <boost/thread/tss.hpp>
#include <cmath>

#include <optional>
#include <unordered_map>
#include <thread>
#include <mutex>

std::atomic<size_t> _total_runtime;
std::mutex _mtx;
std::unordered_map<std::thread::id, bosched::clock::time_point> _map;

namespace bosched
{
    void init_tls()
    {
        _total_runtime.store(size_t(0));
    }

    void iteration_start_record()
    {
        _mtx.lock();
        _map[std::this_thread::get_id()] = bosched::clock::now();
        _mtx.unlock();
    }

    void iteration_stop_record()
    {
        _mtx.lock();
        auto stop_point = _map[std::this_thread::get_id()];
        _mtx.unlock();
        
        auto duration = bosched::clock::now() - stop_point;
        auto discrete = std::chrono::duration_cast<
            std::chrono::nanoseconds>(duration);
        _total_runtime += discrete.count();
    }

        bosched::millisecond
        fetch_total_runtime()
        {
            return std::chrono::nanoseconds(_total_runtime.load());
        }
    }
