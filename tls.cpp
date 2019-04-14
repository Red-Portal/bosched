
#include "tls.hpp"
#include "metrics.hpp"

#include <iostream>
#include <numeric>
#include <atomic>
#include <boost/thread/tss.hpp>
#include <cmath>

#include <optional>
#include <unordered_map>
#include <thread>
#include <mutex>

std::atomic<size_t> _total_runtime;
std::mutex _mtx1;
std::unordered_map<std::thread::id, bosched::clock::time_point> _map;

std::mutex _mtx2;
std::unordered_map<std::thread::id, std::chrono::nanoseconds> _duration;

namespace bosched
{
    void init_tls()
    {
        _total_runtime.store(size_t(0));
        _duration.clear();
    }

    void iteration_start_record()
    {
        _mtx1.lock();
        _map[std::this_thread::get_id()] = bosched::clock::now();
        _mtx1.unlock();
    }

    void iteration_stop_record()
    {
        _mtx1.lock();
        auto start_point = _map[std::this_thread::get_id()];
        _mtx1.unlock();
        
        auto duration = bosched::clock::now() - start_point;
        auto discrete = std::chrono::duration_cast<
            std::chrono::nanoseconds>(duration);
        _total_runtime += discrete.count();

        _mtx2.lock();
        _duration[std::this_thread::get_id()] = discrete;
        _mtx2.unlock();
    }

    double
    coeff_of_variation()
    {
        double mean = std::accumulate(
            _duration.begin(),
            _duration.end(),
            0.0,
            [](double sum, auto pair){
                return sum + bosched::millisecond(pair.second).count();
            }) / _duration.size();

        double second_moment = std::accumulate(
            _duration.begin(),
            _duration.end(),
            0.0,
            [mean](double sum, auto pair){
                auto diff = bosched::millisecond(pair.second).count() - mean;
                return sum + diff * diff;
            }) / _duration.size();

        double sigma = sqrt(second_moment);
        return sigma / mean;
    }

    bosched::millisecond
    fetch_total_runtime()
    {
        return std::chrono::nanoseconds(_total_runtime.load());
    }
}
