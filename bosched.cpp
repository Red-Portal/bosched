
#include "utils.hpp"
#include "loop_state.hpp"
#include "state_io.hpp"
#include "LPBO/LPBO.hpp"

#include <atomic>
#include <blaze/Blaze.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <thread>
#include <unordered_map>

extern char const* __progname;

double const _epsilon = 1e-7;

std::atomic<bool> _show_loop_stat;
std::atomic<bool> _is_debug;
std::atomic<bool> _is_bo_schedule;
std::atomic<bool> _is_new_file;
std::mt19937 _rng __attribute__((init_priority(101)));
std::string _progname __attribute__((init_priority(101)));
std::unordered_map<size_t, bosched::loop_state_t> _loop_states __attribute__((init_priority(101)));

namespace bosched
{
    inline double
    warmup_next_param()
    {
        auto dist = std::uniform_real_distribution<double>(_epsilon, 1.0);
        double next = dist(_rng);
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
                                            _epsilon,
                                            loop_state.iteration,
                                            200);
            loop_state.param = next;
            loop_state.pred_mean.push_back(mean);
            loop_state.pred_var.push_back(var);
            loop_state.pred_acq.push_back(acq);
        }
    }

    inline void
    update_param_non_warmup(loop_state_t& loop_state)
    {
        if(loop_state.obs_y.size() > 0)
            return;

        auto sum = std::accumulate(loop_state.obs_y.begin(),
                                   loop_state.obs_y.end(), 0.0);
        auto y_avg = sum / loop_state.obs_y.size();
        loop_state.gp->update(loop_state.param, y_avg);

        auto [next, mean, var, acq] =
            lpbo::bayesian_optimization(*loop_state.gp,
                                        1e-7,
                                        loop_state.iteration,
                                        200);
        ++loop_state.iteration;
        loop_state.pred_mean.push_back(mean);
        loop_state.pred_var.push_back(var);
        loop_state.pred_acq.push_back(acq);
        loop_state.param = next;
    }

    inline std::unordered_map<size_t, loop_state_t>
    update_loop_parameters(std::unordered_map<size_t, loop_state_t>&& loop_states)
    {
        for(auto& l : loop_states)
        {
            auto loop_id = l.first;
            auto& loop_state = l.second;

            if(loop_state.warming_up)
            {
                update_param_warmup(loop_state);

                if(_is_debug)
                {
                    std::cout << "-- warming up loop " << loop_id 
                              << " current observations: " << loop_state.obs_x.size() 
                              << std::endl;
                }
            }
            else
            {
                update_param_non_warmup(loop_state);

                if(_is_debug)
                {
                    std::cout << "-- updating GP of loop " << loop_id
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

        if(getenv("DEBUG"))
        {
            _is_debug = true;
        }
        if(getenv("LOOPSTAT"))
        {
            _show_loop_stat = true;
        }

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

    void bo_next_called_start(size_t num_tasks)
    {
        if(__builtin_expect (_show_loop_stat == false, 1))
            return;
        
        auto tid = std::this_thread::get_id();
        
    }

    void bo_next_called_stop(size_t num_tasks)
    {
        if(__builtin_expect (_show_loop_stat == false, 1))
            return;

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
            loop_state.warming_up = true;
            loop_state.iteration = 0;
            loop_state.param = param;
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

        if(__builtin_expect (_is_debug == false, 1))
        {
            std::cout << "-- loop " << region_id
                      << " requested schedule parameter " << param
                      << std::endl;
        }
        return param;
    }
    
    void bo_schedule_begin(unsigned long long region_id,
                           unsigned long long N)
    {
        if(__builtin_expect (_is_debug == false, 1))
        {
            std::cout << "-- loop " << region_id << " starting execution."
                      << std::endl;
        }

        _loop_states[region_id].loop_start();
        _loop_states[region_id].num_tasks = N;
    }

    void bo_schedule_end(unsigned long long region_id)
    {
        auto& loop_state = _loop_states[region_id];
        auto duration = loop_state.loop_stop<bosched::millisecond>();

        if(_is_bo_schedule)
        {
            if( loop_state.warming_up && loop_state.obs_x.size() < 20 )
            {
                loop_state.obs_x.push_back(loop_state.param);
                loop_state.obs_y.push_back(duration.count());
            }
            else if(!loop_state.warming_up)
            {
                loop_state.obs_y.push_back(duration.count());
            }
        }

        if(__builtin_expect (_show_loop_stat == false, 1))
        {
            
        }

        if(__builtin_expect (_is_debug == false, 1))
        {
            std::cout << "-- loop " << region_id << " ending execution with runtime "
                      << loop_state.obs_y.back() << "ms" << std::endl;
        }
    }
}


