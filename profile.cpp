
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

    template<typename Float>
    inline boost::multi_array<Float, 2>
    append_multiarray(boost::multi_array<Float, 2>&& a,
		      boost::multi_array<Float, 2> const& b)
    {
	auto a_shape = a.shape();
	auto b_shape = b.shape();
	assert(a_shape[1] == b_shape[1]);
	std::cout << a_shape[0] << " " << a_shape[1] << " " << b_shape[0] << " " << b_shape[1] << std::endl;
	using array_type = typename std::remove_reference<decltype(a)>::type;
	auto extents     = typename array_type::extent_gen();
	a.resize(extents[a_shape[0] + b_shape[0]][a_shape[1]]);
	std::copy(b.begin(), b.end(), a.end() - b.size());
	return a;
    }

    void save_profiles(std::string const& filename,
		       std::unordered_map<size_t, prof::profiles>&& profiles)
    {
	auto file = HighFive::File(filename, HighFive::File::OpenOrCreate);
	auto keys = file.listObjectNames(); 
	for(auto const& [key, value] : profiles)
	{
	    auto name            = std::to_string(key);
	    auto const& new_prof = value.data();
	    if(file.exist(name))
	    {
		auto dataset       = file.getDataSet(name);
		auto dataset_shape = dataset.getSpace().getDimensions();
		auto prof_shape    = new_prof.shape();
		auto prev_shape    = dataset_shape;
		dataset_shape[0]  += prof_shape[0];
		assert(dataset_shape[1] == prof_shape[1]);
		dataset.resize(dataset_shape);
		dataset.select({dataset_shape[0] - 1, dataset_shape[1] - 1})
		    .write(new_prof.data());
	    }
	    else
	    {
		auto prof_shape_raw  = new_prof.shape();
		auto prof_shape      = std::vector<size_t>(2);
		prof_shape[0]        = prof_shape_raw[0];
		prof_shape[1]        = prof_shape_raw[1];

		auto chunk_shape = std::vector<hsize_t>(2);
		chunk_shape[0]   = 1;
		chunk_shape[1]   = prof_shape[1];

		auto props       = HighFive::DataSetCreateProps();
		auto shape_limit = prof_shape;
		shape_limit[0]   = HighFive::DataSpace::UNLIMITED;
		auto space       = HighFive::DataSpace(
		    prof_shape, shape_limit);  
		props.add(HighFive::Chunking(chunk_shape));

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
