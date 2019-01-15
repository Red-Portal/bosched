
#include "tls.hpp"
#include "metrics.hpp"

#include <atomic>
#include <boost/thread/tss.hpp>
#include <cmath>


#include <optional>

std::atomic<size_t> _total_runtime;
std::optional<boost::thread_specific_ptr<bosched::clock::time_point>> _start;

namespace bosched
{
    void init_tls()
    {
        _total_runtime.store(size_t(0));
    }

    void iteration_start_record()
    {
        if(!_start)
        {
            _start.emplace();
            _start->reset(new bosched::clock::time_point(bosched::clock::now()));
        }
        // if(_start.get() == nullptr)
        // {
        //     _start.reset(new bosched::clock::time_point(bosched::clock::now()));
        // }
        **_start = bosched::clock::now();
    }

    void iteration_stop_record()
    {
        auto duration = bosched::clock::now() - **_start;
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
