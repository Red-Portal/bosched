
#include "utils.hpp"
#include "loop_state"

#include <blaze/Blaze.h>
#include <chrono>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_map>

std::unordered_map<size_t, loop_state_t> _loop_states;
bool _is_new_file;

namespace bosched
{
    inline double
    warmup_next_param()
    {
        // assert generated parameter is never zero
    }

    inline void
    update_loop_parameters(size_t iteration,
                           std::unordered_map<size_t, loop_state_t>&& loop_states)
    {
        for(auto& l : loop_states)
        {
            size_t loop_id = l.first;
            auto& loop_state = l.second;

            if(loop_state.warming_up)
            {
                if(loop_state.obs_x.size() > 10)
                {
                    size_t particle_num = 10;
                    double survival_rate = 0.8;
                    loop_state.warming_up = false;
                    loop_state.emplace(loop_state.obs_x,
                                       loop_state.obs_y,
                                       particle_num,
                                       survival_rate);
                    loop_state.iteration = 1;

                    auto [next, mean, var] = bayesian_optimization(*loop_state.gp,
                                                                   iteration,
                                                                   200);
                    /* */
                }
                
                if(getenv("DEBUG"))
                {
                    std::cout << " loop id: " << loop_id << " is warming up."
                              << ' ' << loop_state.obs_x.size() << " observations"
                              << std::endl;
                }
            }
            else
            {
                if(loop_state.obs_x.size() > 1)
                    update(loop_state.obs_x, loop_state.obs_y);

                auto [next, mean, var] = bayesian_optimization(*loop_state.gp,
                                                               loop_state.iteration,
                                                               200);
                ++loop_state.iteration;

                /* */
                
                if(getenv("DEBUG"))
                {
                    std::cout << " loop id: " << loop_id
                              << " next point: " << next_param
                              << std::endl;
                }
            }
        }
        return std::move(loop_states);
    }
}

extern "C"
{
    void
    bo_load_data(char const* progname, size_t sched_id)
    {
        using namespace std::literals::string_literals;
        
        auto file_name = ".bostate."s
            + std::to_string(sched_id)
            + std::string_view(progname);

        std::ifstream stream(file_name + ".json"s);

        if(stream.fail())
            is_new_file = true;
        else
        {
            is_new_file = false;
            auto data = nlohmann::json();
            stream >> data;
            auto [iteration, loop_states] = read_loops(std::move(data));
            _iteration = iteration;
            _loop_states = std::move(loop_states);
        }
    }

    void
    bo_save_data(char const* progname, size_t sched_id)
    {
        auto update_states = update_loop_parameters(
            _iteration, std::move(_loop_states));

        ++_iteration;
        auto next = write_loops(_iteration, std::move(updated_states));

        using namespace std::literals::string_literals;
        auto file_name = ".bostate."s
            + std::to_string(sched_id)
            + std::string_view(progname);
        std::ifstream stream(file_name + ".json"s);
        stream << next; 
    }

    double
    bo_schedule_parameter(unsigned long long region_id)
    {
        double param = 0.0;

        auto& loop_state = _loop_states[region_id];

        if(is_new_file)
        {
            loop_state.warming_up = true;
            loop_state.iteration = 0;
        }

        if(loop_state.warming_up)
        {
            param = generate_warmup();
            loop_state.param = param;
        }
        else
        {
            param = loop_state.param;
        }

        if(getenv("DEBUG"))
        {
            std::cout << "-- loop " << region_id
                      << " requested schedule parameter " << param
                      << std::endl;
        }
        return param;
    }
    
    void bo_schedule_begin(unsigned long long region_id)
    {
        if(getenv("DEBUG"))
        {
            std::cout << "-- loop " << region_id << " starting execution."
                      << std::endl;
        }
        _loop_states[region_id].loop_start();
    }

    void bo_schedule_end(unsigned long long region_id)
    {
        auto& loop_state = _loop_states[region_id];
        auto duration = loop_state.loop_stop<bosched::microsecond>();

        loop_state.obs_x.push_back(region_id);
        loop_state.obs_y.push_back(duration.count());

        if(getenv("DEBUG"))
        {
            std::cout << "-- loop " << region_id << " ending execution with runtime "
                      << loop_state.obs_y.back() << "us" << std::endl;
        }
    }
}

