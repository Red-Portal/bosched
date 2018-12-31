
#ifndef _STATE_IO_HPP_
#define _STATE_IO_HPP_

#include "loop_state.hpp"
#include "utils.hpp"

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
            state.id = l["id"];
            state.param = l["param"];
            state.warming_up = l["warmup"];
            state.iteration = l["iteration"];

            if(state.warming_up)
            {
                auto obs_x = l.at("obs_x");
                auto obs_y = l.at("obs_y");
                state.obs_x = std::vector<double>(obs_x.cbegin(), obs_x.cend());
                state.obs_y = std::vector<double>(obs_y.cbegin(), obs_y.cend());
            }
            else
            {
                state.gp.emplace(l["gp"]);
            }

            loop_states[state.id] = std::move(state);

            if(getenv("DEBUG"))
            {
                std::cout << " [ deserialized ]" << '\n'
                          << " loop id: " << state.id
                          << " parameter: " << state.param
                          << " number of observations: " << state.obs_x.size()
                          << std::endl;
            }
        }
        return loop_states;
    }

    inline nlohmann::json
    write_state(std::unordered_map<size_t, loop_state_t>&& loop_states)
    {
        nlohmann::json serialized_states;
        for(auto const& l : loop_states)
        {
            auto const& loop_state = l.second;

            nlohmann::json serialized_state;
            serialized_state["id"] = loop_state.id;
            serialized_state["param"] = loop_state.param;
            serialized_state["warmup"] = loop_state.warming_up;
            serialized_state["iteration"] = loop_state.iteration;

            if(loop_state.warming_up)
            {
                serialized_state["obs_x"] = nlohmann::json(std::move(loop_state.obs_x));
                serialized_state["obs_y"] = nlohmann::json(std::move(loop_state.obs_y));
            }
            else
            {
                serialized_state["gp"] = loop_state.gp->serialize();
            }

            serialized_states.emplace_back(serialized_state);

            if(getenv("DEBUG"))
            {
                std::cout << " [ serialized ]" << '\n'
                          << " loop id: " << loop_state.id
                          << " parameter: " << loop_state.param
                          << " number of observations: " << loop_state.obs_x.size()
                          << std::endl;
            }
        }
        return serialized_states;
    }

    inline std::unordered_map<size_t, loop_state_t>
    read_loops(nlohmann::json&& data)
    {
        using namespace std::literals::string_literals;
        std::unordered_map<size_t, loop_state_t> loop_states;

        auto date      = data["data"];
        auto num_loops = data["num_loops"];

        std::cout << "--------- bayesian optimization ---------" << '\n'
                  << " modified date   | " << date << '\n'
                  << " number of loops | " << num_loops << '\n'
                  << '\n'
                  << " set environment variable DEBUG for detailed info"
                  << std::endl;

        loop_states = read_state(data["loops"]);
        return loop_states;
    }

    inline nlohmann::json
    write_loops(std::unordered_map<size_t, loop_state_t>&& loop_states)
    {
        std::cout << "size: " << loop_states.size() << std::endl;
        nlohmann::json data;
        data["date"] = format_current_time();
        data["num_loops"] = loop_states.size();
        data["loops"] = write_state(std::move(loop_states));
        return data;
    }
}

#endif
