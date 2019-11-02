
#ifndef _PROFILE_HPP_
#define _PROFILE_HPP_

#include <boost/multi_array.hpp>
#include <cassert>
#include <cstdlib>
#include <highfive/H5File.hpp>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace prof
{
    class profiles
    {
	boost::multi_array<float, 2>  _data;
	
    public:
	inline
	profiles() : _data() {}

	inline
	profiles(boost::multi_array<float, 2>&& loaded)
	    : _data(std::move(loaded))
	{}

	inline void
	push(std::vector<float> const& data) 
	{
	    //assert(data.size() == _data.shape()[0]);
	    size_t num_entry = _data.shape()[1];
	    auto extents = decltype(_data)::extent_gen();
	    // row major order
	    _data.resize(extents[num_entry + 1][data.size()]);
	    std::copy(data.begin(), data.end(),
		      _data[num_entry].end() - data.size());
	}

	inline boost::multi_array<float, 2>
	data() const
	{
	    return _data;
	}
    };

    void save_profiles(std::string const& filename,
		       std::unordered_map<size_t, prof::profiles>&& profiles);

    void profiling_init(size_t num_tasks);

    void iteration_profile_start(long iter);

    void iteration_profile_stop();

    std::vector<float> load_profile();
}

#endif
