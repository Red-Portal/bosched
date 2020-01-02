
include("simulation/binlpt_utils.jl")
include("LABO/LABO.jl")

import ArgParse
import Base.Filesystem
import JSON
using .LABO
using HDF5
using Plots
using Statistics
using TerminalMenus
using UnicodePlots
using Base.Iterators

const fs  = Base.Filesystem

function bo_fss_transform(x)
    return 2^(15*x - 8)
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
    return 2^(19*x - 13)
end

function cmd_args(args, show)
    s = ArgParse.ArgParseSettings("Bayesian Optimization Loop Scheduling Offline Routine")
    ArgParse.@add_arg_table s begin
        "--mode"
        arg_type     = String
        help         = "Mode of operation. mode ∈ ."
        metavar      = "{classic|bosched|both|visualize}"
        range_tester = (mode-> mode ∈ ["classic", "bosched", "both", "visualize"])
        required     = true
        "--path"
        arg_type = String
        help     = "Path of workload data."
        metavar  = "<dir>"
        default  = "."
        "--h"
        arg_type = Float64
        help     = "Scheduling overhead parameter (us)."
        metavar  = "<millisecs>"
        default  = 0.1
        "--P"
        arg_type = Int64
        help     = "Number of cores."
        metavar  = "<cores>"
        default  = 32 
        "--subsize"
        arg_type = Int64
        help     = "Bayesian optimization subsample size"
        metavar  = "<size>"
        default  = 256
        "--time_samples"
        arg_type = Int64
        help     = "Locality axis sample numbers"
        metavar  = "<size>"
        default  = 4
        "name"
        arg_type = String
        help     = "Name of workload executable."
        metavar  = "<file>"
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
    for loop in loop_states
        arr = read(workload_profile, string(loop["id"]))
        μ   = mean(arr)
        σ   = stdm(arr, μ)

        d    = Dict()
        prof = compute_workload_profile(arr)
        #println(prof)
        d["css"]    = h / σ
        d["fss"]    = μ / σ
        d["binlpt"] = binlpt_balance(prof, P, 2 * P)
        d["hss"]    = stretch(prof, 0, 255)
        loop["params"] = d

        println("----- $(loop["id"]) classic mode -----")
        println(" μ   = $μ")
        println(" σ   = $σ")
        println(" css = $(d["css"])")
        println(" fss = $(d["fss"])\n")
    end
    return loop_states
end

function update_dataset(loop_state)
    if(!haskey(loop_state, "hist_x") || loop_state["hist_x"] == nothing)
        loop_state["hist_x"] = []
        loop_state["hist_y"] = []
    end
    if(!isempty(loop_state["obs_x"]))
        push!(loop_state["hist_x"], loop_state["obs_x"]...) 
        push!(loop_state["hist_y"], loop_state["obs_y"]...) 
        loop_state["obs_x"] = []
        loop_state["obs_y"] = []
    end
    return loop_state
end

function bosched_mode(loop_states, time_samples, subsize, P)
    for loop in loop_states
        println("----- $(loop["id"]) bosched mode -----")
        loop  = update_dataset(loop)
        x, y  = loop["hist_x"], loop["hist_y"]

        x  = hcat(x...)
        y  = hcat(y...)
        x  = convert(Array{Float64}, x)
        y  = convert(Array{Float64}, y)
        y /= (loop["N"] / P)
        @assert size(x) == size(y)

        θ_next, θ_mean, acq_opt, mean_opt = LABO.labo(
            x, -y, length(x[1]), time_samples, subsize, verbose=true)

        hist_x   = collect(flatten(loop["hist_x"]))
        hist_y   = collect(flatten(loop["hist_y"]))
        hist_min = findmin(hist_y)
        θ_hist   = hist_min[1]

        loop["eval_param1"] = θ_hist
        loop["eval_param2"] = θ_mean
        loop["param"]       = θ_next
        loop["warmup"]      = false

        println(" best param until now = $(θ_hist)")
        println(" predicted optimum    = $(θ_mean), optimum mean = $(mean_opt)")
        println(" next query point     = $(θ_next), acquisition  = $(acq_opt)")
        println(" number of data point = $(size(x, 2))")
    end
    return loop_states
end

function visualize_gp(loop_states, time_samples, subsize, P)
    #uimodes  = ["GUI", "CUI"]
    #uimenu   = RadioMenu(uimodes, pagesize=2)
    #uichoice = request("choose interface:", uimenu)

    options = ["continue", "export", "exit"]
    menu    = RadioMenu(options, pagesize=3)
    for loop in loop_states
        x, y  = loop["hist_x"], loop["hist_y"]
        max_t = 

        x  = hcat(x...)
        y  = hcat(y...)
        x  = convert(Array{Float64}, x)
        y  = convert(Array{Float64}, y)
        y /= (loop["N"] / P)
        @assert size(x) == size(y)

        d = Dict()
        θ_next, θ_mean, acq_opt, mean_opt = LABO.labo(
            x, -y, length(x[1]), time_samples, subsize, verbose=true, logdict=d)

        αx   = d[:acq_x]
        αy   = d[:acq_y]
        gpx  = d[:gp_x]
        gpσ  = d[:gp_std]
        gpμ  = d[:gp_mean]
        gpμ *= -1

        p1   = Plots.plot(gpx, gpμ, grid=false, ribbon=gpσ*1.96,
                          fillalpha=.5, label="GP (95%)", xlims=(0.0,1.0))
        flat_x = collect(flatten(x))
        flat_y = collect(flatten(y))
        Plots.scatter!(p1, flat_x, LABO.whiten(flat_y),
                       label="data points", xlims=(0.0,1.0))
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
    println("-- found $bostate_fname")

    bostate = open(bostate_fname) do file
        str   = read(file, String)
        loops = JSON.parse(str)
        loops
    end
    println("-- $bostate_fname contains $(bostate["num_loops"]) loops")

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
        println("-- found $workload_fname")

        bostate = h5open(workload_fname) do file
            bostate["loops"] = classic_mode(file,
                                            bostate["loops"],
                                            args["h"],
                                            args["P"])
            bostate
        end
    end
    if(args["mode"] == "bosched" || args["mode"] == "both")
        bostate["loops"]  = bosched_mode(bostate["loops"],
                                         args["time_samples"],
                                         args["subsize"],
                                         args["P"])
    end
    if(args["mode"] == "visualize")
        visualize_gp(bostate["loops"],
                     args["time_samples"],
                     args["subsize"],
                     args["P"])
    end

    open(bostate_fname, "w") do file
        loops = JSON.print(file, bostate, 4)
    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main(ARGS)
    return 0
end

# if get(ENV, "COMPILE_STATIC", "false") == "false"
#     julia_main(ARGS)
# end
