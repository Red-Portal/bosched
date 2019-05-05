
#ifndef _PROFILE_HPP_
#define _PROFILE_HPP_

#include <cstdlib>
#include <vector>

namespace prof
{
    void profiling_init(size_t num_tasks);

    void iteration_profile_start(size_t iter);

    void iteration_profile_stop();

    std::vector<float> load_profile();
}

#endif
