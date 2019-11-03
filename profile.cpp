
#include "metrics.hpp"
#include "profile.hpp"
#include <chrono>
#include <omp.h>
#include <string>

namespace prof
{
    using iter_id = size_t;
    std::vector<std::pair<iter_id, bosched::clock::time_point>> _timestamp;
    std::vector<float>  _loads;

    void save_profiles(std::string const& filename,
		       std::unordered_map<size_t, prof::profiles>&& profiles)
    {
	auto file = HighFive::File(filename, HighFive::File::OpenOrCreate);
	auto keys = file.listObjectNames(); 
	for(auto const& [key, value] : profiles)
	{
	    auto name            = std::to_string(key);
	    auto const& new_prof = value;
	    if(file.exist(name))
	    {
		auto dataset     = file.getDataSet(name);
		auto prev_shape  = dataset.getSpace().getDimensions();
		auto next_shape = {prev_shape[0] + new_prof.entries(),
				   new_prof.length()};
		dataset.resize(next_shape);
		dataset.select({prev_shape[0], 0},
			       {new_prof.entries(), new_prof.length()})
		    .write(new_prof.data());
	    }
	    else
	    {
		auto space       = HighFive::DataSpace(
		    {new_prof.entries()            , new_prof.length()},
		    {HighFive::DataSpace::UNLIMITED, new_prof.length()});  

		auto props       = HighFive::DataSetCreateProps();
		props.add(HighFive::Chunking({1, new_prof.length()}));

		auto dataset = file.createDataSet<float>(name, space, props);
		dataset.write(new_prof.data());
	    }
	}
    }

    void profiling_init(size_t num_tasks)
    {
        _timestamp = std::vector<std::pair<iter_id, bosched::clock::time_point>>(
            omp_get_max_threads());
        _loads = std::vector<float>(num_tasks);
    }

    void iteration_profile_start(long iter)
    {
        size_t tid = omp_get_thread_num();
        _timestamp[tid] = std::make_pair(static_cast<iter_id>(iter),
                                         bosched::clock::now());
    }

    void iteration_profile_stop()
    {
        auto stop_stamp = bosched::clock::now();
        size_t tid      = omp_get_thread_num();
        auto [iter, start_stamp] = _timestamp[tid];
        _loads[iter] = bosched::microsecond(stop_stamp - start_stamp).count();
    }

    std::vector<float>
    load_profile()
    {
        return _loads;
    }
}
