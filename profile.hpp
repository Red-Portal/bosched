
#ifndef _PROFILE_HPP_
#define _PROFILE_HPP_

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
	std::vector<std::vector<float>>  _data;
	
    public:
	inline
	profiles() : _data() {}

	inline void
	push(std::vector<float>&& data) 
	{
	    if(_data.size() > 0)
	    {
		printf("%d %d\n", _data.back().size(), data.size());
		assert(_data.back().size() == data.size());
	    }
	    _data.push_back(std::move(data));
	}

	inline std::vector<std::vector<float>>
	data() const
	{
	    return _data;
	}

	inline size_t
	entries() const
	{
	    return _data.size();
	}

	inline size_t
	length() const
	{
	    return _data.back().size();
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
