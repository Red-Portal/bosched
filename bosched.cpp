
#include "loop_state.hpp"
#include "metrics.hpp"
#include "state_io.hpp"
#include "tls.hpp"
#include "utils.hpp"
#include "performance.hpp"
#include "profile.hpp"
#include "param.hpp"

#include <highfive/H5File.hpp>
#include <iostream>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <thread>
#include <unordered_map>
//#include <mpi.h>

extern char const* __progname;

bool _show_loop_stat = false;
bool _is_debug       = false;
bool _is_bo_schedule = false;
bool _is_new_file    = false;
bool _profile_loop   = false;
bool _is_eval        = false;
bool _fallback       = false;
//bool _random_extrplt = false;
std::mt19937   _rng __attribute__((init_priority(101)));
nlohmann::json _stats       __attribute__((init_priority(101)));
std::unordered_map<size_t, bosched::loop_state_t>    _loop_states __attribute__((init_priority(101)));
std::unordered_map<size_t, prof::profiles>           _profiles    __attribute__((init_priority(101)));
std::unordered_map<size_t, bosched::workload_params> _params      __attribute__((init_priority(101)));
long _procs;

namespace bosched
{
    inline double
    warmup_next_param(double seed = 0.0)
    {
        static std::uniform_real_distribution<double> dist(0.0, 1.0);
	static std::mt19937 rng(seed);
        double next = dist(rng);
        return next;
    }

    std::pair<int, int> process_binding()
    {
        using namespace std::literals::string_literals;
#ifdef SYS_getcpu
	int cpu, node;
	if(int status = syscall(SYS_getcpu, &cpu, &node, NULL);
	    status == -1)
	{
	    throw std::runtime_error("syscall getcpu failed with status "s
				     + std::to_string(status));
	}
	return {cpu, node};
#else
	throw std::error_code("syscall getcpu unvailable");
#endif
    }
}

void prefetch_page(size_t prealloc_len)
{
    auto buf = std::vector<char>(prealloc_len);
    for(size_t i = 0; i < prealloc_len; i += 1024 * 4)
	buf[i] = 'a';
    asm volatile("" : : "r,m"(buf) : "memory");
}

extern "C"
{
    void __attribute__ ((constructor(65535)))
    bo_load_data()
    {
        std::ios_base::Init();

        using namespace std::literals::string_literals;
        auto progname = std::string(__progname);

        if(getenv("DEBUG"))
            _is_debug = true;

        if(getenv("FALLBACK"))
            _fallback = true;

	size_t pages = getpagesize();
	prefetch_page(pages * 1024 * 512);

        if(getenv("PROFILE"))
        {
            _profile_loop = true;
        }
	else
	{
	    auto file_name =
		[](std::string const& name){
		    if(getenv("NUMA"))
		    {
			auto [cpuid, nodeid] = bosched::process_binding();
			(void)cpuid;
			auto id = std::to_string( nodeid );
			return ".bostate."s + id + "."s + name;
		    }
		    else
		    { return ".bostate."s + name; }
		}(progname);
	    std::ifstream stream(file_name + ".json"s);

	    if(stream)
	    {
		_is_new_file = false;
		auto data = nlohmann::json();
		stream >> data;
		_loop_states = bosched::read_loops(data);
		_params      = bosched::load_workload_params(data);
	    }
	    else
	    {
		_is_new_file = true;
	    }
	    stream.close();

	    for(auto& [key, val] : _loop_states )
	    {
		(void)key;
		if(val.warming_up)
		    val.param = bosched::warmup_next_param(val.param);
	    }
	}

        if(getenv("LOOPSTAT"))
        {
            _show_loop_stat = true;
            auto stat_file_name = ".stat."s + progname;
            auto stat_stream = std::ifstream(stat_file_name + ".json"s);
            if(stat_stream)
                stat_stream >> _stats; 
            stat_stream.close();
	}

	if(_is_debug)
	{
	    auto cmd = "cat /proc/"s + std::to_string(getpid()) + "/status"s;
	    auto ret = system(cmd.c_str());
	    (void)ret;
	}

	if(getenv("EVAL"))
	{
	    _is_eval = true;
        }
    }

    void __attribute__ ((destructor))
    bo_save_data()
    {
        using namespace std::literals::string_literals;

        auto progname = std::string(__progname);

        if(_show_loop_stat)
        {
            auto stat_file_name = ".stat."s + progname;
            auto stat_stream = std::ofstream(stat_file_name + ".json"s);
            stat_stream << _stats.dump(2); 
            stat_stream.close();
        }

        if(_is_eval)
            return;

	// int mpi_init = 0;
	// MPI_Initialized(&mpi_init);
	// if(static_cast<bool>(mpi_init)) 
	// {
	//     MPI_Comm_rank(comm)
	//     if()
	// 	return;
	// }

        if(_profile_loop)
        {
            auto prof_file_name = ".workload."s + progname + ".h5"s;
	    prof::save_profiles(prof_file_name, std::move(_profiles));
        }
	else if(_loop_states.size() > 0)
	{
	    auto next      = bosched::write_loops(std::move(_loop_states));
	    auto file_name =
		[](std::string const& name){
		    if(getenv("NUMA"))
		    {
			auto [cpuid, nodeid] = bosched::process_binding();
			(void)cpuid;
			auto id = std::to_string( nodeid );
			return ".bostate."s + id + "."s + name;
		    }
		    else
		    { return ".bostate."s + name; }
		}(progname);

	    auto stream = std::ofstream(file_name + ".json"s);
	    stream << next.dump(2); 
	    stream.close();
	}
    }

    void bo_workload_profile_start(long iteration)
    {
        if(_profile_loop)
            prof::iteration_profile_start(iteration);
    }

    void bo_workload_profile_stop()
    {
        if(_profile_loop)
            prof::iteration_profile_stop();
    }

    void bo_record_iteration_start()
    {
        if(__builtin_expect (_show_loop_stat == false, 1))
            return;
        stat::iteration_start_record();
    }

    void bo_record_iteration_stop()
    {
        if(__builtin_expect (_show_loop_stat == false, 1))
            return;
        stat::iteration_stop_record();
    }

    void bo_binlpt_load_loop(unsigned long long region_id,
                             unsigned** task_map)
    {
        auto& profile = _params[region_id].binlpt;
        *task_map = profile.data();
        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id << '\n'
		      << " workload size = " << profile.size() << '\n'
                      << " requested workload binlpt profile"
                      << std::endl;
        }
    }

    void bo_hss_load_loop(unsigned long long region_id,
                          unsigned** task_map)
    {
        auto& profile = _params[region_id].hss;
        *task_map = profile.data();
        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id << '\n'
		      << " workload size = " << profile.size() << '\n'
                      << " requested workload hss profile"
                      << std::endl;
        }
    }

    double bo_fss_parameter(unsigned long long region_id)
    {
        double param = _params[region_id].fss;
        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id
                      << " requested fss schedule parameter " << param
                      << std::endl;
        }
        return param;
    }

    // double bo_fac_parameter(unsigned long long region_id)
    // {
    //     double param = _params[region_id].fac;
    //     if(__builtin_expect (_is_debug, false))
    //     {
    //         std::cout << "-- loop " << region_id
    //                   << " requested fss schedule parameter " << param
    //                   << std::endl;
    //     }
    //     return param;
    // }

    double bo_css_parameter(unsigned long long region_id)
    {
        double param = _params[region_id].css;
        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id
                      << " requested css schedule parameter " << param
                      << std::endl;
        }
        return param;
    }

    double bo_tss_parameter(unsigned long long region_id)
    {
        double param = _params[region_id].tss.value();
        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id
                      << " requested tss schedule parameter " << param
                      << std::endl;
        }
        return param;
    }

    double bo_tape_parameter(unsigned long long region_id)
    {
        double param = _params[region_id].tape.value();
        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id
                      << " requested schedule parameter " << param
                      << std::endl;
        }
        return param;
    }

    double
    bo_schedule_parameter(unsigned long long region_id,
                          int is_bo_schedule)
    {
        auto& loop_state = _loop_states[region_id];
        _is_bo_schedule = static_cast<bool>(is_bo_schedule);

        if(_is_new_file)
            loop_state.warming_up = true;
        
	if(is_bo_schedule)
	{
	    if(_is_eval)
	    {
		loop_state.param = loop_state.eval_param;
	    }
	    // else if(loop_state.warming_up && _random_extrplt)
	    // {
	    // 	loop_state.param = bosched::warmup_next_param();
	    // }
	}
	double param = loop_state.param;

        if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id
                      << " requested boched schedule parameter " << param
                      << std::endl;
        }
	return param;
    }

    int bo_fallback_static (unsigned long long region_id, int is_bo)
    {
        auto _is_bo = static_cast<bool>(is_bo);
	if(_fallback && _is_eval && _is_bo)
	{
	    auto& loop_state = _loop_states[region_id];
	    return (int)(loop_state.param < 1e-5);
	}
    }
    
    void bo_schedule_begin(unsigned long long region_id,
                           unsigned long long N,
                           long procs)
    {
	auto& loop = _loop_states[region_id];
        if(_is_bo_schedule || _show_loop_stat)
        {
            loop.loop_start();
            loop.num_tasks = N;
            _procs = procs;
        }

        if(__builtin_expect(_profile_loop, false))
        {
	    prof::profiling_init(N);
        }

        if(__builtin_expect (_show_loop_stat, false))
        {
            stat::init_tls();
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
		double exec_time = duration.count();
		loop_state.obs_x.push_back(loop_state.param);
		loop_state.obs_y.push_back(exec_time * _procs / loop_state.num_tasks);
	    }

	    if(__builtin_expect (_show_loop_stat, false))
            {
                auto work_time = std::chrono::duration_cast<time_scale_t>(
                    stat::total_work()).count();

                auto parallel_time = duration.count();

                auto work_per_processor = stat::work_per_processor();
                auto performance        = work_time / parallel_time;
                auto cost               = parallel_time * _procs;
                auto effectiveness      = performance / cost;
                auto cov                = bosched::coeff_of_variation(work_per_processor);
                auto slowdown           = bosched::slowdown(work_per_processor);

                auto& log = _stats[std::to_string(region_id)];
                log["num_tasks"    ].push_back(loop_state.num_tasks);
                log["work_time"    ].push_back(work_time);
                log["parallel_time"].push_back(parallel_time);
                log["effectiveness"].push_back(effectiveness);
                log["performance"  ].push_back(performance);
                log["task_mean"    ].push_back(work_time / loop_state.num_tasks);
                log["slowdown"     ].push_back(slowdown);
                log["cov"          ].push_back(cov);
                log["cost"         ].push_back(cost);
                if(_is_debug)
                {
                    std::cout << "-- loop " << region_id << " stats \n"
                              << log.dump(2) << '\n' << std::endl;
                }
            }
        }

        if(__builtin_expect(_profile_loop, false))
        {
	    _profiles[region_id].push(prof::load_profile());
	}

	if(__builtin_expect (_is_debug, false))
        {
            std::cout << "-- loop " << region_id << " ending execution" << std::endl;
        }
    }
}


