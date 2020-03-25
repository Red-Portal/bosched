
#ifndef _PARAMS_HPP_
#define _PARAMS_HPP_

#include <vector>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace bosched
{
    struct workload_params
    {
        double css;
        double fss;
	//double fac;
        std::optional<double> tss;
        std::optional<double> tape;
        std::vector<unsigned> binlpt;
        std::vector<unsigned> hss;
    };

    inline std::unordered_map<size_t, bosched::workload_params>
    load_workload_params(nlohmann::json& loops_json)
    {
        auto result = std::unordered_map<size_t, bosched::workload_params>();
        result.reserve(loops_json["num_loops"]);
        for(auto& loop : loops_json["loops"])
        {
            auto param_bundle  = workload_params();
	    if(loop.count("params") == 0)
	    {
		result[loop["id"]] = std::move(param_bundle);
		continue;
	    }

	    auto params       = loop["params"];
            double css_param  = params["css"];
            double fss_param  = params["fss"];
            //double fac_param  = value["fac"];

            if(params.count("tape") > 0)
            {
                double tape = params["tape"];
                param_bundle.tape.emplace(tape);
            }

            if(params.count("tss") > 0)
            {
                double tss = params["tss"];
                param_bundle.tss.emplace(tss);
            }

            auto& binlpt_json = params["binlpt"];
            auto& hss_json    = params["hss"];

            param_bundle.css    = css_param;
            param_bundle.fss    = fss_param;
            //param_bundle.fac    = fac_param;
            param_bundle.binlpt = std::vector<unsigned>(binlpt_json.size());
            param_bundle.hss    = std::vector<unsigned>(hss_json.size());

            std::transform(binlpt_json.begin(),
                           binlpt_json.end(),
                           param_bundle.binlpt.begin(),
                           [](auto elem){
                               return static_cast<unsigned>(elem);
                           });

            std::transform(hss_json.begin(),
                           hss_json.end(),
                           param_bundle.hss.begin(),
                           [](auto elem){
                               return static_cast<unsigned>(elem);
                           });

            result[loop["id"]] = std::move(param_bundle);
        }
        return result;
    }
}

#endif
