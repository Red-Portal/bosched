
#ifndef _STATE_IO_HPP_
#define _STATE_IO_HPP_

#include "loop_state.hpp"
#include "utils.hpp"

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace bosched
{
    inline std::unordered_map<size_t, loop_state_t>
    read_state(nlohmann::json const& loop_data)
    {
        std::unordered_map<size_t, loop_state_t> loop_states;
        for(auto const& l : loop_data)
        {
            loop_state_t state;
            auto loop_id     = l["id"];
            state.param      = l["param"];
            state.warming_up = l["warmup"];

	    if(!state.warming_up)
	    {
		auto& gmm = l["gmm"];
		state.eval_param = gmm["eval_param"];
		auto& gmm_weight_json = gmm["gmm_weight"];
		state.gmm_weight = std::vector<double>(gmm_weight_json.begin(),
						       gmm_weight_json.end());
		auto& gmm_mean_json = gmm["gmm_mean"];
		state.gmm_mean = std::vector<double>(gmm_mean_json.begin(),
						     gmm_mean_json.end());
		auto& gmm_stddev_json = gmm["gmm_stddev"];
		state.gmm_stddev = std::vector<double>(gmm_stddev_json.begin(),
						       gmm_stddev_json.end());
	    }

	    auto& obs_x = l["obs_x"];
	    auto& obs_y = l["obs_y"];
	    state.obs_x  = std::vector<double>(obs_x.cbegin(), obs_x.cend());
	    state.obs_y  = std::vector<double>(obs_y.cbegin(), obs_y.cend());

	    if(l.count("hist_x") > 0)
	    {
		state.hist_x = std::move(l["hist_x"]); //;
		state.hist_y = std::move(l["hist_y"]); //#std::vector<double>(hist_y.cbegin(), hist_y.cend());
	    }

            if(getenv("DEBUG"))
            {
                std::cout << "-- deserialized loop " << loop_id
                          << " warmup              = " << state.warming_up
                          << " parameter           = " << state.param
			  << " observation entries = " << state.hist_x.size()
                          << std::endl;
            }

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

            nlohmann::json serialized_state;
            serialized_state["id"]     = loop_id;
            serialized_state["param"]  = loop_state.param;
            serialized_state["warmup"] = loop_state.warming_up;

	    serialized_state["obs_x"]  = nlohmann::json(std::move(loop_state.obs_x));
	    serialized_state["obs_y"]  = nlohmann::json(std::move(loop_state.obs_y));
	    serialized_state["hist_x"] = std::move(loop_state.hist_x);
	    serialized_state["hist_y"] = std::move(loop_state.hist_y);

	    if(!loop_state.warming_up)
	    {
		auto gmm = nlohmann::json();
		gmm["gmm_weight"] = nlohmann::json(std::move(loop_state.gmm_weight));
		gmm["gmm_mean"]   = nlohmann::json(std::move(loop_state.gmm_mean));
		gmm["gmm_stddev"] = nlohmann::json(std::move(loop_state.gmm_stddev));
		gmm["eval_param"] = nlohmann::json(std::move(loop_state.eval_param));
		serialized_state["gmm"] = std::move(gmm);
	    }

	    if(getenv("DEBUG"))
            {
                std::cout << "-- serialized loop " << serialized_state["id"]
                          << " parameter    = " << serialized_state["param"]
                          << " observations = " << serialized_state["obs_x"].size()
                          << std::endl;
            }

            serialized_states.emplace_back(serialized_state);
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
