
import ArgParse
import Base.Filesystem
import JSON
using CSV
using GaussianProcesses
using Statistics

const fs  = Base.Filesystem

include("particle_gp.jl")
include("acquisition.jl")

function IBBO(x, y)
    μ  = mean(y)
    σ  = stdm(y, μ) 
    m  = MeanConst(μ)
    k  = SE(0.0, σ)
    set_priors!(k, [Normal(), Normal()])

    println(" fitting initial gp")
    k  = fix(k, :lσ)
    ϵ  = -1.0
    gp = GP(x[:], y[:], m, k, ϵ)
    println(" fitting initial gp - done")

    println(" optimizing hyperparameters")
    gp = nuts(gp)
    println(" optimizing hyperparameters - done")

    println(" sampling y*")
    η         = maximum(y)
    P         = gp.weights
    num_ystar = 100
    num_gridx = 100
    for i = 1:P
        sample_ystar(i, η, num_ystar, num_gridx, gp)
    end
    println(" sampling y* - done")

    println(" sampling acquisition function ")
    num_samples   = 1000
    sample_points = collect(0.0:1/num_samples:1.0)'[:,:]
    acq_samples   = acquisition(sample_points)
    println(" sampling acquisition function - done")

    println(" fitting gaussian mixture model")
    
    println(" fitting gaussian mixture model - done")
    return ws, μs, σs, ent
end

function cmd_args(args, show)
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--render"
        arg_type = String
        default  = "."
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

function main(args)
    parsed_args = cmd_args(args, true)
    filename    = fs.joinpath(parsed_args["path"], args[1])

    loops = open(filename) do file
        str   = read(file, String)
        loops = JSON.parse(str)
    end

    for (key, value) in loops
        println("-- processing loop ", key)
        obs_x = value["obs_x"]
        obs_y = value["obs_y"]

        println(" average runtime : ", mean(obs_y))
        println(" number of sample: ", length(obs_y))

        if(num_obs < 5)
            println("number of sample is less than 5, skipping...")
            continue
        end

        ws, μs, σs, ent = IBBO(obs_x, obs_y)

        value["ws"] = ws
        value["μs"] = μs
        value["σs"] = σs
            
        if(parsed_args["render"] != nothing)
            
        end
    end
end

main(ARGS)
