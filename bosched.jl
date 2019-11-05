
import ArgParse
import Base.Filesystem
import JSON
using HDF5
using Plots
using Statistics
using TerminalMenus

include("simulation/binlpt_utils.jl")
include("IBBO/IBBO.jl")

const fs  = Base.Filesystem

function bo_fss_transform(x)
    return 2^(11*x - 7)
end

function bo_fac_transform(x)
    return 2^(5*x)
end

# function bo_tss_transform(x)
#     return 2^(11*x - 7)
# end

function bo_taper_transform(x)
    return 2^(13*x - 7)
end

function bo_css_transform(x)
    return 2^(10*x - 5)
end

function cmd_args(args, show)
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--mode"
        arg_type = String
        help     = "Mode of operation. mode ∈ {\"classic\", \"bosched\", \"both\", \"visualize\"}."
        required = true
        "--path"
        arg_type = String
        help     = "Path of workload data."
        default  = "."
        "--h"
        arg_type = Float64
        help     = "Scheduling overhead parameter (ms)."
        default  = 0.1
        "--P"
        arg_type = Int64
        help     = "Number of cores."
        default  = 32 
        "--subsize"
        arg_type = Int64
        help     = "Bayesian optimization subsample size"
        default  = 32 
        "name"
        arg_type = String
        help     = "Name of workload executable."
        default  = ""
    end

    args = ArgParse.parse_args(args, s)
    if(show)
        for (key, value) in args
            if(value != nothing)
                println(" $key => $value")
            end
        end
    end
    return args
end

function compute_workload_profile(data)
    med = median(data, dims=2)
    return med[:,1]
end

function stretch(arr, lo, hi)
    normzed = (arr .- minimum(arr)) / maximum(arr)
    return normzed * (hi - lo) .+ lo
end

function classic_mode(workload_profile, loop_states, h, P)
    loops = names(workload_profile)
    for loop in loops
        arr = read(workload_profile, loop)
        μ   = mean(arr)
        σ   = stdm(arr, μ)

        d    = Dict()
        prof = compute_workload_profile(arr)
        #println(prof)
        d["css"]    = h / σ
        d["fss"]    = μ / σ
        d["binlpt"] = binlpt_balance(prof, P, 2 * P)
        d["hss"]    = stretch(prof, 0, 255)

        idx = findfirst(x->x["id"] == parse(Int64, loop), loop_states)
        loop_states[idx]["params"] = d

        println("----- $loop classic mode -----")
        println(" μ   = $μ")
        println(" σ   = $σ")
        println(" css = $(d["css"])")
        println(" fss = $(d["fss"])\n")
    end
    return loop_states
end

function bo_subsample(obs_x, obs_y, subsample)
    res_x = []
    res_y = []
    for i = 1:length(obs_x)
        x = obs_x[i]
        y = obs_y[i]
        push!(res_x, x[1:length(x)])
        push!(res_y, y[1:length(y)])
    end
    return vcat(res_x...), vcat(res_y...)
end

function bosched_mode(loop_states, subsize)
    for loop in loop_states
        if(!haskey(loop, "hist_x"))
            loop["hist_x"] = []
            loop["hist_y"] = []
        end
        append!(loop["hist_x"], loop["obs_x"]) 
        append!(loop["hist_y"], loop["obs_y"]) 
        x, y = bo_subsample(loop["hist_x"], loop["hist_y"], subsize)
        x = convert(Array{Float64}, x)
        y = convert(Array{Float64}, y)
        @assert length(x) == length(y)

        println("----- $(loop["id"]) bosched mode -----")
        w, μ, σ, H, best_θ, best_y = IBBO(x, y, true, false)
        loop["eval_param"] = best_θ
        loop["gmm_weight"] = w
        loop["gmm_mean"]   = μ
        loop["gmm_stddev"] = σ
        loop["warmup"]     = false

        println(" num components  = $(length(w))")
        println(" mixture entropy = $H")
        println(" observations    = $(length(x))")
    end
    return loop_states
end

function visualize_gp(loop_states, subsize)
    options = ["continue", "export", "exit"]
    menu    = RadioMenu(options, pagesize=3)
    for loop in loop_states
        x, y = bo_subsample(loop["hist_x"], loop["hist_y"], subsize)
        x = convert(Array{Float64}, x)
        y = convert(Array{Float64}, y)

        @assert length(x) == length(y)
        w, μ, σ, H, αx, αy, gpx, gpμ, gpσ², samples, best_θ, best_y =
            IBBO_log(x, y, true, true)
        p1 = Plots.plot(gpx, gpμ, grid=false, ribbon=sqrt.(gpσ²)*1.96,
                        fillalpha=.5, label="GP (95%)", xlims=(0.0,1.0))
        Plots.scatter!(p1, x, whitening(y), label="data points", xlims=(0.0,1.0))
        p2 = Plots.plot(αx, αy, label="acquisition", xlims=(0.0,1.0))
        display(plot(p1, p2, layout=(2,1)))

        choice = request("choose action:", menu)
        if(options[choice] == "export")
            
        elseif(options[choice] == "exit")
            exit()
        end
    end
end

function main(args)
    args  = cmd_args(args, true)
    path  = args["path"]
    files = readdir(path)

    bostate_fname = filter(x->occursin(".bostate", x), files) |>
        f-> filter(x->occursin(args["name"], x), f) |>
        f-> fs.joinpath.(path, f)
    bostate_fname = bostate_fname[1]

    bostate = open(bostate_fname) do file
        str   = read(file, String)
        loops = JSON.parse(str)
        loops
    end

    if(!(args["mode"] == "classic"
         || args["mode"] == "bosched"
         || args["mode"] == "both"
         || args["mode"] == "visualize"))
        error("Mode of operation must be, mode ∈ {\"classic\", \"bosched\", \"both\"}.")
    end
    if(args["mode"] == "classic" || args["mode"] == "both")
        workload_fname = filter(x->occursin(".workload", x), files) |>
            f-> filter(x->occursin(args["name"], x), f) |>
            f-> fs.joinpath.(path, f)
        workload_fname = workload_fname[1]

        bostate = h5open(workload_fname) do file
            bostate["loops"] = classic_mode(file, bostate["loops"],
                                            args["h"], args["P"])
            bostate
        end
    end
    if(args["mode"] == "bosched" || args["mode"] == "both")
        bostate["loops"]  = bosched_mode(bostate["loops"], args["subsize"])
    end
    if(args["mode"] == "visualize")
        visualize_gp(bostate["loops"], args["subsize"])
    end

    open(bostate_fname, "w") do file
        loops = JSON.print(file, bostate, 4)
    end
end

main(ARGS)
