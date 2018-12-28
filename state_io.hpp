
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
    warmup_state(nlohmann::json const& loop_data)
    {
        std::unordered_map<size_t, loop_state_t> loop_states;
        for(auto const& l : loop_data)
        {
            loop_state_t state;
            state.id = l["id"];
            state.param = l["param"];

            auto obs_x = l["obs_x"];
            auto obs_y = l["obs_y"];

            state.obs_x = std::vector<double>(obs_x.cbegin(), obs_x.cend());
            state.obs_y = std::vector<double>(obs_y.cbegin(), obs_y.cend());
            loop_states[state.id] = std::move(state);
        }
        return loop_states;
    }

    inline std::unordered_map<size_t, loop_state_t>
    running_state(nlohmann::json const& loop_data)
    {
        std::unordered_map<size_t, loop_state_t> loop_states;
        for(auto const& l : loop_data)
        {
            loop_state_t state;
            state.id = l["id"];
            state.param = l["param"];

            auto obs_x = l["obs_x"];
            auto obs_y = l["obs_y"];

            state.obs_x = std::vector<double>(obs_x.cbegin(), obs_x.cend());
            state.obs_y = std::vector<double>(obs_y.cbegin(), obs_y.cend());
            state.gp.emplace(l["gp"]);

            if(getenv("DEBUG"))
            {
                std::cout << " [ deserialized ]" << '\n'
                          << " loop id: " << state.id
                          << " parameter: " << state.param
                          << " number of observations: " << state.obs_x.size()
                          << std::endl;
            }
            loop_states[state.id] = std::move(state);
        }
        return loop_states;
    }

    using iteration_t = size_t;
    using loop_table_t = std::unordered_map<size_t, loop_state_t>;
    using parsed_state_t = std::tuple<iteration_t, loop_table_t>;

    inline parsed_state_t
    read_loops(nlohmann::json&& data,
               std::string const& file_name)
    {
        using namespace std::literals::string_literals;
        std::unordered_map<size_t, loop_state_t> loop_states;

        auto iteration = data["iterations"];
        auto num_loops = data["num_loop"];

        std::cout << "--------- bayesian optimization ---------" << '\n'
                  << " modified date   | " << data["date"] << '\n'
                  << " iterations      | " << iteration << '\n'
                  << " number of loops | " << num_loops << '\n'
                  << '\n'
                  << " set environment variable DEBUG for detailed info"
                  << std::endl;

        if(iteration < 10)
            loop_states = warmup_state(data["loops"]);
        else
            loop_states = running_state(data["loops"]);
        return {iteration, loop_states};
    }

    inline nlohmann::json
    write_loops(nlohmann::json&& prev)
    {
        prev["date"] = format_current_time();
        ++prev["iterations"];

        for(auto& l : prev["loops"])
        {
            size_t loop_id = l.key();
            auto& loop_state = _global_state.loop_states[loop_id];
            l.value()["param"] = loop_state.param;
            l.value()["obs_x"] = std::move(loop_state.obs_x);
            l.value()["obs_y"] = std::move(loop_state.obs_y);

            l.value()["pred_mean"].push_back(loop_state.pred_mean);
            l.value()["pred_var"].push_back(loop_state.red_var);
        }
        return prev;
    }
}

#endif
