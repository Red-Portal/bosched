
#ifndef _STATE_IO_HPP_
#define _STATE_IO_HPP_

#include "loop_state.hpp"
#include "utils.hpp"

#include <cstddef>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace bosched
{
    inline std::unordered_map<size_t, loop_state_t>
    read_state(nlohmann::json const& loop_data)
    {
        std::unordered_map<size_t, loop_state_t> loop_states;
        for(auto& l : loop_data)
        {
            loop_state_t state;
            auto loop_id     = l["id"];

	    if(loop_id == 0 || l["N"] == 0)
		continue;

            state.warming_up = l["warmup"];

	    if(!state.warming_up)
	    {
		state.param = l["param"];
		if(getenv("HIST"))
		    state.eval_param = l["eval_param1"];
		else
		    state.eval_param = l["eval_param2"];
	    }
	    else
	    {
		state.param = l["obs_x"].back()[0];
	    }

            if(getenv("DEBUG"))
            {
		size_t entry_count = 0;
		if(l.count("hist_x") > 0)
		    entry_count = l["hist_x"].size();

                std::cout << "-- deserialized loop " << loop_id << '\n'
                          << " warmup              = " << state.warming_up << '\n'
			  << " observation entries = " << entry_count << std::endl;
            }

	    state.loop_json  = l;
            loop_states[loop_id] = std::move(state);
        }
        return loop_states;
    }

    inline nlohmann::json
    write_state(std::unordered_map<size_t, loop_state_t>&& loop_states)
    {
        nlohmann::json serialized_states;
        for(auto& l : loop_states)
        {
            auto loop_id = l.first;
            auto& loop_state = l.second;

	    if(loop_state.num_tasks == 0)
		continue;

            auto serialized_state      = loop_state.loop_json;
            serialized_state["id"]     = loop_id;
            serialized_state["warmup"] = loop_state.warming_up;
	    serialized_state["N"]      = loop_state.num_tasks;
	    serialized_state["obs_x"].emplace_back(std::move(loop_state.obs_x));
	    serialized_state["obs_y"].emplace_back(std::move(loop_state.obs_y));

	    if(getenv("DEBUG"))
            {
                std::cout << "-- serialized loop " << serialized_state["id"] << '\n'
                          << " observations = " << serialized_state["obs_x"].size() << std::endl;
            }
            serialized_states.emplace_back(std::move(serialized_state));
        }
        return serialized_states;
    }

    inline std::unordered_map<size_t, loop_state_t>
    read_loops(nlohmann::json const& data)
    {
        using namespace std::literals::string_literals;
        std::unordered_map<size_t, loop_state_t> loop_states;

        auto date      = data["date"];
        auto num_loops = data["num_loops"];

        std::cout << "--------- bayesian optimization ---------" << '\n'
                  << " modified date   | " << date << '\n'
                  << " number of loops | " << num_loops << '\n'
                  << std::endl; 

        if(!getenv("DEBUG") )
        {
            std::cout << " set environment variable DEBUG for detailed info"
                      << std::endl;
        }

        loop_states = read_state(data["loops"]);
        return loop_states;
    }

    inline nlohmann::json
    write_loops(std::unordered_map<size_t, loop_state_t>&& loop_states)
    {
        nlohmann::json data;
        data["date"]      = format_current_time();
        data["num_loops"] = loop_states.size();
        data["loops"]     = write_state(std::move(loop_states));
        return data;
    }
}

#endif
