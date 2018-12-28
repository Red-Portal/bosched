
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

inline double
generate_warmup()
{
    
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
        {
            is_new_file = true;
            nlohmann::json new_file;
        }
        else
        {
            is_new_file = false;
            auto data = nlohmann::json();
            stream >> data;
            read_loops(std::move(data), file_name);
        }
    }

    double
    bo_schedule_parameter(unsigned long long region_id)
    {
        double param = 0.5;
        if(is_new_file)
            _loop_states[region_id].param = param;
        else
            param = _loop_states[region_id].param;

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

    void
    bo_save_data(char const* progname, size_t sched_id)
    {
        for(auto& l : _global_state.loop_states)
        {
            if(l.obs_x.size() == 0)
                continue;

            size_t loop_id = l.first;
            auto& json_state = _global_state.json_data["loops"]["loop_id"];

            /* finish rest  */
            
            double next_param = 0.5;
            auto num_obs = l.obs_x.size() + ;
            if(num_obs < warm_up_num)
            {
                next_param = generate_warmup();
                l.param = next_param;
            }
            else
            {
                double pred_mean = 0;
                double pred_var  = 0;
                next_param =
                    next_point(l.obs_x.data(), l.obs_y.data(), l.obs_x.size(),
                               , 500, &pred_mean, &pred_var);
                l.pred_mean = pred_mean;
                l.pred_var = pred_var;

                if(getenv("DEBUG"))
                {
                    std::cout << " loop id: " << loop_id
                              << " next point: " << next_param
                              << std::endl;
                }
            }
            l.param = next_param;
        }

        auto prev = std::move(_global_state.json_data);
        auto next = write_loops(prev);

        using namespace std::literals::string_literals;
        auto file_name = ".bostate."s
            + std::to_string(sched_id)
            + std::string_view(progname);
        std::ifstream stream(file_name + ".json"s);
        stream << next; 
    }

}

