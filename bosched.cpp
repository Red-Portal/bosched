
#include <blaze/Blaze.h>
#include "LPBO/GP.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <string_view>
#include <unordered_map>
#include <ctime>
#include <cstdlib>
#include <nlohmann/json.hpp>

size_t const warm_up_num = 10;

bool is_new_file = false;

struct loop_state_t
{
    size_t observations;
    nlohmann::json json_data;
    lpbo::gp_model gp_model;
    std::vector<double> obs_x;
    std::vector<double> obs_y;
};

struct global_state_t
{
    std::string  date;
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

    _global_state.date = data["date"];
    _global_state.iters = data["iterations"];
    _global_state.num_loops = data["num_loop"];

    std::cout << "--------- bayesian optimization ---------"
              << " modified date   | " << _global_state.date
              << " iterations      | " << _global_state.iters
              << " number of loops | " << _global_state.num_loops
              << '\n'
              << " set environment variable DEBUG for detailed info"
              << std::endl;

    std::ifstream matrix_archive(file_name + ".mat.blaze");
    std::ifstream vector_archive(file_name + ".mat.blaze");

    for(auto l : data["loops"])
    {
        loop_state_t state;
        size_t loop_id = l["id"];
        size_t num_obs = l["num_obs"];
        state.obs_x = std::vector<double>(l["obs_x"].begin(), l["obs_x"].end());
        state.obs_x = std::vector<double>(l["obs_y"].begin(), l["obs_y"].end());

        if(num_obs)
            state.gp_model.deserialize(archive_stream);

        _global_state.loop_states[loop_id] = state;

        if(getenv("DEBUG"))
        {
            std::cout << " loop id: " << loop_id
                      << " number of observations: " << num_obs
                      << " observations: " << std::endl;
        }
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
        {
            is_new_file = true;
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
        
    }

    double
    bo_schedule_parameter(unsigned long long region_id)
    {
        (void)region_id;
        return 0.0;
    }
    
    void bo_schedule_begin(unsigned long long region_id)
    {

        std::cout << "-- parallel loop " <<  region_id << " begin"
                  << std::endl;;
    }

    void bo_schedule_end(unsigned long long region_id)
    {
        std::cout << "-- parallel loop " <<  region_id << " end"
                  << std::endl;;
    }
}

