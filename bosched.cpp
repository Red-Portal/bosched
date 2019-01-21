
#include <iostream>

#include "LPBO/LPBO.hpp"
#include "loop_state.hpp"
#include "metrics.hpp"
#include "state_io.hpp"
#include "tls.hpp"
#include "utils.hpp"

#include <atomic>
#include <blaze/Blaze.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <thread>
#include <unordered_map>

extern char const* __progname;

double const _epsilon = 1e-7;

bool _show_loop_stat = false;
bool _is_debug       = false;
bool _is_bo_schedule = false;
bool _is_new_file    = false;
std::mt19937 _rng __attribute__((init_priority(101)));
std::string _progname __attribute__((init_priority(101)));
std::uniform_real_distribution<double> dist(_epsilon, 1.0);
std::unordered_map<size_t, bosched::loop_state_t> _loop_states __attribute__((init_priority(101)));
long _procs;
nlohmann::json _stats;

namespace bosched
{
    inline double
    warmup_next_param()
    {
        double next = dist(_rng);
        return next;
    }

    inline void
    update_param_warmup(loop_state_t& loop_state)
    {
        if(loop_state.obs_x.size() > 20)
        {
            size_t particle_num = 10;
            double survival_rate = 0.8;

            try
            {
                loop_state.gp.emplace(loop_state.obs_x,
                                      loop_state.obs_y,
                                      particle_num,
                                      survival_rate);
            }
            catch(std::runtime_error const& err)
            {
                std::cout << "-- covariance matrix singularity detected.. skipping"
                          << std::endl;
                return;
            }

            loop_state.iteration = 1;
            loop_state.warming_up = false;

            auto [next, mean, var, acq] =
                lpbo::bayesian_optimization(*loop_state.gp,
                                            _epsilon,
                                            loop_state.iteration,
                                            1000);
            loop_state.param = next;
            loop_state.pred_mean.push_back(mean);
            loop_state.pred_var.push_back(var);
            loop_state.pred_acq.push_back(acq);
        }
    }

    inline void
    update_param_non_warmup(loop_state_t& loop_state)
    {
        if(loop_state.obs_y.size() < 1)
            return;

        auto sum = std::accumulate(loop_state.obs_y.begin(),
                                   loop_state.obs_y.end(), 0.0);
        auto y_avg = sum / loop_state.obs_y.size();
        try
        { loop_state.gp->update(loop_state.param, y_avg); }
        catch(std::runtime_error const& err)
        {
            std::cout << "-- covariance matrix singularity detected.. skipping"
                      << std::endl;
            return;
        }

        auto [next, mean, var, acq] =
            lpbo::bayesian_optimization(*loop_state.gp,
                                        _epsilon,
                                        loop_state.iteration,
                                        1000);
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

    inline void
    eval_loop_parameters(std::unordered_map<size_t, loop_state_t>& loop_states)
    {
        for(auto& l : loop_states)
        {
            auto loop_id = l.first;
            auto& loop_state = l.second;

            if(loop_state.warming_up)
                continue;

            auto [param, mean, var] = lpbo::find_best_mean(*loop_state.gp, _epsilon, 2000);
            loop_state.param = param;

            if(_is_debug)
            {
                std::cout << "-- evaluating mean GP of loop " << loop_id
                          << " best param: " << loop_state.param
                          << " mean: " << mean
                          << " var: " << var
                          << std::endl;
            }
        }
    }
}


extern "C"
{
    void __attribute__ ((constructor(65535)))
    bo_load_data()
    {
        std::ios_base::Init();

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

        if(getenv("EVAL"))
        {
            bosched::eval_loop_parameters(_loop_states);
        }
    }

    void __attribute__ ((destructor))
    bo_save_data()
    {
        if(getenv("EVAL"))
            return;

        auto updated_states = _is_bo_schedule ?
            update_loop_parameters(std::move(_loop_states)) 
            : std::move(_loop_states);

        auto next = bosched::write_loops(updated_states);

        using namespace std::literals::string_literals;
        if(_show_loop_stat)
        {
            auto date = bosched::format_current_time();
            auto stat_file_name = ".stat."s + _progname + "."s + date;
            auto stat_stream = std::ofstream(stat_file_name + ".json"s);
            stat_stream << _stats.dump(2); 
            stat_stream.close();
        }

        auto file_name = ".bostate."s + _progname;
        auto stream = std::ofstream(file_name + ".json"s);
        stream << next.dump(2); 
        stream.close();
    }

    void bo_record_iteration_start()
    {
        if(__builtin_expect (_show_loop_stat == false, 1))
            return;
        bosched::iteration_start_record();
    }

    void bo_record_iteration_stop()
    {
        if(__builtin_expect (_show_loop_stat == false, 1))
            return;
        bosched::iteration_stop_record();
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

        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id
                      << " requested schedule parameter " << param
                      << std::endl;
        }
        return param;
    }
    
    void bo_schedule_begin(unsigned long long region_id,
                           unsigned long long N,
                           long procs)
    {
        if(_is_bo_schedule || _show_loop_stat)
        {
            _loop_states[region_id].loop_start();
            _loop_states[region_id].num_tasks = N;
            _procs = procs;
        }

        if(__builtin_expect (_show_loop_stat, false))
        {
            bosched::init_tls();
        }
        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id << " starting execution."
                      << " iterations: " << N
                      << std::endl;
        }
    }

    void bo_schedule_end(unsigned long long region_id)
    {
        if(_is_bo_schedule || _show_loop_stat)
        {
            using time_scale_t = bosched::microsecond;

            auto& loop_state = _loop_states[region_id];

            auto duration = loop_state.loop_stop<time_scale_t>();

            if(_is_bo_schedule)
            {
                if( loop_state.warming_up && loop_state.obs_x.size() < 30 )
                {
                    loop_state.obs_x.push_back(loop_state.param);
                    loop_state.obs_y.push_back(duration.count());
                }
                else if(!loop_state.warming_up)
                {
                    loop_state.obs_y.push_back(duration.count());
                }
            }

            if(__builtin_expect (_show_loop_stat, false))
            {
                auto work_time = std::chrono::duration_cast<time_scale_t>(
                    bosched::fetch_total_runtime());

                auto total_overhead = (duration.count() - (work_time.count() / _procs));
                auto efficiency = 1 / (1 + (total_overhead / work_time.count()));

                auto log = nlohmann::json();
                log["num_tasks"]  = loop_state.num_tasks;
                log["work_time"]  = work_time.count();
                log["loop_time"]  = duration.count();
                log["efficiency"] = efficiency;
                log["task_mean"]  = work_time.count() / loop_state.num_tasks;
                if(_is_debug)
                {
                    std::cout << "-- loop " << region_id << " stats \n"
                              << log.dump(2) << '\n' << std::endl;
                }
                _stats.push_back(std::move(log));
            }
        }

        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id << " ending execution" << std::endl;
        }
    }
}


