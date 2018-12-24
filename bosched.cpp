
#include "LPBO/GP.hpp"
#include "LPBO/LPBO.hpp"
#include <blaze/Blaze.h>
#include <chrono>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <unordered_map>

size_t const warm_up_num = 10;

bool is_new_file = false;

struct loop_state_t
{
    //lpbo::gp_model gp_model;
    std::vector<double> obs_x;
    std::vector<double> obs_y;
    double param;
    chrono::steady_clock::time_point start;
};

struct global_state_t
{
    nlohmann::json json_data;
    size_t iters;
    size_t num_loops;
    std::unordered_map<size_t, loop_state_t> loop_states;
};

global_state_t _global_state;

inline std::string
format_current_time()
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    return std::string(buffer);
}

inline void read_loops(nlohmann::json&& data, std::string const& file_name)
{
    using namespace std::literals::string_literals;

    _global_state.iters = data["iterations"];
    _global_state.num_loops = data["num_loop"];

    std::cout << "--------- bayesian optimization ---------"
              << " modified date   | " << data["date"]
              << " iterations      | " << _global_state.iters
              << " number of loops | " << _global_state.num_loops
              << '\n'
              << " set environment variable DEBUG for detailed info"
              << std::endl;

    //std::ifstream matrix_archive(file_name + ".mat.blaze");
    //std::ifstream vector_archive(file_name + ".mat.blaze");

    for(auto& l : data["loops"])
    {
        loop_state_t state;
        state.id = l.key();
        state.param = l.value()["param"];
        if(getenv("DEBUG"))
        {
            std::cout << " loop id: " << loop_id
                      << " number of observations: " << l.value()["obs_x"].size()
                      << std::endl;
        }
        _global_state.loop_states[loop_id] = std::move(state);
    }
    _global_state.json_data = std::move(data);
}

inline nlohmann::json
write_loops(nlohmann::json&& prev)
{
    prev["date"] = format_current_time();
    ++prev["iterations"];

    for(auto& l : prev["loops"])
    {
        size_t loop_id = l.key();
        auto& loop_state = _global_state.loop_states[loop_id];
        l.value()["param"] = loop_state.param;
        l.value()["obs_x"] = std::move(loop_state.obs_x);
        l.value()["obs_y"] = std::move(loop_state.obs_y);

        l.value()["pred_mean"].push_back(loop_state.pred_mean);
        l.value()["pred_var"].push_back(loop_state.red_var);
    }
    return prev;
}

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
            new_file["iters"] = 1;
            _global_state.json_data = std::move(new_file);
        }
        else
        {
            is_new_file = false;
            auto data = nlohmann::json();
            stream >> data;
            read_loops(std::move(data), file_name);
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

    double
    bo_schedule_parameter(unsigned long long region_id)
    {
        double param = 0.5;
        if(is_new_file)
        {
            _global_state.loop_states[region_id].param = param;
            ++_global_state.num_loops;
        }
        else
        {
            param = _global_state.loop_states[region_id].param;
        }

        if(getenv("DEBUG"))
        {
            std::cout << "-- loop " << region_id << " requested schedule parameter " << param
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
        _global_state.loop_states[region_id].start =
            std::chrono::steady_clock::now();
    }

    void bo_schedule_end(unsigned long long region_id)
    {
        auto& loop_state = _global_state.loop_states[region_id];
        auto start_point = loop_state.start;
        auto duration = std::chrono::steady_clock::now() - start_point;

        loop_state.obs_x.push_back(region_id);
        loop_state.obs_y.push_back(
            std::chrono::duration_cast<double, std::micro>(
                duration).count());

        if(getenv("DEBUG"))
        {
            std::cout << "-- loop " << region_id << " ending execution with runtime "
                      << loop_state.obs_y.back() << "us" << std::endl;
        }
    }
}

