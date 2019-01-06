
#ifndef _BOSCHED_TLS_HPP_
#define _BOSCHED_TLS_HPP_

#include <boost/thread/tss.hpp>

#include "metrics.hpp"

namespace bosched
{
    void init_tls();

    void iteration_start_record();

    void iteration_stop_record();

    bosched::millisecond fetch_total_runtime();
}

#endif
