
#include "utils.hpp"
#include "loop_state.hpp"
#include "state_io.hpp"
#include "LPBO/LPBO.hpp"

#include <blaze/Blaze.h>
#include <chrono>
#include <random>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <unordered_map>

extern char const* __progname;

bool _is_bo_schedule;
bool _is_new_file;
std::mt19937 _rng __attribute__((init_priority(101)));
std::string _progname __attribute__((init_priority(101)));
std::unordered_map<size_t, bosched::loop_state_t> _loop_states __attribute__((init_priority(101)));

namespace bosched
{
    inline double
    warmup_next_param()
    {
        auto dist = std::uniform_real_distribution<double>(0.0, 1.0);
        double next = 0;
        do 
        {
            next = dist(_rng);
        } while(next == 0);
        return next;
    }

    inline void
    update_param_warmup(loop_state_t& loop_state)
    {
        if(loop_state.obs_x.size() > 10)
        {
            size_t particle_num = 10;
            double survival_rate = 0.8;
            loop_state.warming_up = false;

            loop_state.gp.emplace(loop_state.obs_x,
                                  loop_state.obs_y,
                                  particle_num,
                                  survival_rate);
            loop_state.iteration = 1;

            auto [next, mean, var, acq] =
                lpbo::bayesian_optimization(*loop_state.gp,
                                            1e-7,
                                            loop_state.iteration,
                                            200);
            loop_state.mean.push_back(mean);
            loop_state.var.push_back(var);
            loop_state.acq.push_back(acq);
            loop_state.param = next;
        }
    }

    inline void
    update_param_non_warmup(loop_state_t& loop_state)
    {
        if(loop_state.obs_x.size() > 1)
        {
            loop_state.gp->update(loop_state.obs_x,
                                  loop_state.obs_y);
        }

        auto [next, mean, var, acq] =
            lpbo::bayesian_optimization(*loop_state.gp,
                                        1e-7,
                                        loop_state.iteration,
                                        200);
        ++loop_state.iteration;
        loop_state.mean.push_back(mean);
        loop_state.var.push_back(var);
        loop_state.acq.push_back(acq);
        loop_state.param = next;
    }

    inline std::unordered_map<size_t, loop_state_t>
    update_loop_parameters(std::unordered_map<size_t, loop_state_t>&& loop_states)
    {
        for(auto& l : loop_states)
        {
            auto& loop_state = l.second;

            if(loop_state.warming_up)
            {
                update_param_warmup(loop_state);

                if(getenv("DEBUG"))
                {
                    std::cout << "-- warming up loop " << loop_state.id
                              << " current observations: " << loop_state.obs_x.size() 
                              << std::endl;
                }
            }
            else
            {
                update_param_non_warmup(loop_state);

                if(getenv("DEBUG"))
                {
                    std::cout << "-- updating GP of loop " << loop_state.id
                              << " next point: " << loop_state.param
                              << std::endl;
                }
            }
        }
        return std::move(loop_states);
    }
}


extern "C"
{
    void __attribute__ ((constructor(102)))
    bo_load_data()
    {
        using namespace std::literals::string_literals;
        _progname = std::string(__progname);

        auto seed = std::random_device();
        _rng = std::mt19937(seed());

        auto file_name = ".bostate."s + _progname;
        std::ifstream stream(file_name + ".json"s);

        if(stream)
        {
            _is_new_file = false;
            auto data = nlohmann::json();
            stream >> data;
            _loop_states = bosched::read_loops(data);
        }
        else
        {
            _is_new_file = true;
        }
        stream.close();
    }

    void __attribute__ ((destructor))
    bo_save_data()
    {
        auto updated_states = _is_bo_schedule ?
            update_loop_parameters(std::move(_loop_states)) 
            : std::move(_loop_states);

        auto next = bosched::write_loops(updated_states);

        using namespace std::literals::string_literals;
        auto file_name = ".bostate."s + _progname;
        auto stream = std::ofstream(file_name + ".json"s);
        stream << next.dump(2); 
        stream.close();
    }

    double
    bo_schedule_parameter(unsigned long long region_id,
                          int is_bo_schedule)
    {
        double param = 0.5;
        auto& loop_state = _loop_states[region_id];
        _is_bo_schedule = static_cast<bool>(is_bo_schedule);

        if(_is_new_file)
        {
            loop_state.id = region_id;
            loop_state.warming_up = true;
            loop_state.iteration = 0;
        }
        
        if(loop_state.warming_up && is_bo_schedule)
        {
            param = bosched::warmup_next_param();
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
        auto duration = loop_state.loop_stop<bosched::millisecond>();

        loop_state.obs_x.push_back(loop_state.param);
        loop_state.obs_y.push_back(duration.count());

        if(getenv("DEBUG"))
        {
            std::cout << "-- loop " << region_id << " ending execution with runtime "
                      << loop_state.obs_y.back() << "us" << std::endl;
        }
    }
}


