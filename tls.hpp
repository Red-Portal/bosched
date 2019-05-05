
#ifndef _BOSCHED_TLS_HPP_
#define _BOSCHED_TLS_HPP_

#include <boost/thread/tss.hpp>
#include <vector>

#include "metrics.hpp"

namespace stat
{
    void init_tls();

    void iteration_start_record();

    void iteration_stop_record();

    std::vector<double> work_per_processor();

    bosched::millisecond total_work();
}

#endif
