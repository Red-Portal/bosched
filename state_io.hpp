
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
            auto loop_id = l["id"];

            loop_state_t state;
            state.param = l["param"];
            state.warming_up = l["warmup"];
            state.iteration = l["iteration"];

            if(state.warming_up)
            {
                auto obs_x = l["obs_x"];
                auto obs_y = l["obs_y"];
                state.obs_x = std::vector<double>(obs_x.cbegin(), obs_x.cend());
                state.obs_y = std::vector<double>(obs_y.cbegin(), obs_y.cend());
            }
            else
            {
                auto bytes = l["gp"];
                auto byte_vec = std::vector<uint8_t>(bytes.begin(), bytes.end());
                auto str = std::string(byte_vec.begin(), byte_vec.end());
                state.gp.emplace(str);

                auto mean = l["mean"];
                auto var  = l["var"];
                auto acq  = l["acq"];
                state.pred_mean = std::vector<double>(mean.cbegin(), mean.cend());
                state.pred_var  = std::vector<double>(var.cbegin(), var.cend());
                state.pred_acq  = std::vector<double>(acq.cbegin(), acq.cend());
            }

            if(getenv("DEBUG"))
            {
                std::cout << "-- deserialized loop " << loop_id
                          << " warmup: "             << state.warming_up
                          << " parameter: "          << state.param
                          << " observations: "       << state.obs_x.size()
                          << std::endl;
            }

            loop_states[loop_id] = std::move(state);
        }
        return loop_states;
    }

    inline nlohmann::json
    write_state(std::unordered_map<size_t, loop_state_t> const& loop_states)
    {
        nlohmann::json serialized_states;
        for(auto const& l : loop_states)
        {
            auto loop_id = l.first;
            auto const& loop_state = l.second;

            nlohmann::json serialized_state;
            serialized_state["id"] = loop_id;
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
                auto str = loop_state.gp->serialize();
                auto bytes = std::vector<uint8_t>(str.begin(), str.end());
                serialized_state["gp"] = bytes;

                serialized_state["mean"] = loop_state.pred_mean;
                serialized_state["var"] = loop_state.pred_var;
                serialized_state["acq"] = loop_state.pred_acq;
            }

            if(getenv("DEBUG"))
            {
                std::cout << "-- serialized loop " << serialized_state["id"]
                          << " parameter: "        << serialized_state["param"]
                          << " observations: "     << serialized_state["obs_x"].size()
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
    write_loops(std::unordered_map<size_t, loop_state_t> const& loop_states)
    {
        nlohmann::json data;
        data["date"] = format_current_time();
        data["num_loops"] = loop_states.size();
        data["loops"] = write_state(std::move(loop_states));
        return data;
    }
}

#endif
