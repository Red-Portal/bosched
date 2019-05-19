
import JSON
import ArgParse
import Base.Filesystem
using Statistics

const fs  = Base.Filesystem

function cmd_args(args, show)
    s = ArgParse.ArgParseSettings()
    ArgParse.@add_arg_table s begin
        "--css"
        arg_type = Float64
        "--fss"
        arg_type = Float64
        "--tape"
        arg_type = Float64
        "--tss"
        arg_type = Float64
        "--uniform"
        arg_type = String
        "--bias1"
        arg_type = String
        "--bias2"
        arg_type = String
        "--bias3"
        arg_type = String
        "--N"
        arg_type = Int
        "--path"
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

function css_transform_range(param::Float64)
    return exp(15 * param - 8);
end

function fss_transform_range(param::Float64)
    return 2.0^(27 * param - 20);
end

function tape_transform_range(param::Float64)
    return exp(15 * param - 10);
end

function bias1()
    ntasks = 2^15;
    tasks  = zeros(Float64, ntasks)
    for i in 0:ntasks-1
        tasks[i+1] = convert(Float64, i)* (10.0 / (32 * 1024)) + 10;
    end
    return tasks
end

function bias2()
    ntasks = 2^15;
    tasks  = zeros(Float64, ntasks)
    for i in 0:ntasks-1
        tasks[i+1] = convert(Float64, i) * (-10.0 / (32 * 1024)) + 20;
    end
    return tasks
end

function bias3()
    ntasks = 2^15;
    tasks  = zeros(Float64, ntasks)
    key    = UInt16(1024)
    for i in 0:ntasks-1
        if key & UInt16(i) > 0
            tasks[i+1] = 10.0;
        else
            tasks[i+1] = 20.0;
        end
    end
    return tasks
end

function main(args)
    parsed_args = cmd_args(args, true)
    filename    = fs.joinpath(parsed_args["path"], ".params.json")

    loops = open(filename) do file
        str   = read(file, String)
        loops = JSON.parse(str)
    end

    for (key, value) in loops
        println("-- processing loop ", key)
        if(parsed_args["css"] != nothing)
            param = parsed_args["css"]
            value["css"] = css_transform_range(param);
        end

        if(parsed_args["fss"] != nothing)
            param = parsed_args["fss"]
            value["fss"] = fss_transform_range(param);
        end

        if(parsed_args["tss"] != nothing)
            value["tss"] = parsed_args["tss"];
        end

        if(parsed_args["tape"] != nothing)
            param = parsed_args["tape"]
            value["tape"] = tape_transform_range(param)
        end
    end

    if(parsed_args["bias1"] != nothing)
        loop     = parsed_args["bias1"]
        workload = bias1()[1:parsed_args["N"]]
        μ        = mean(workload)
        σ        = stdm(workload, μ, corrected = size(workload)[1] > 1)
        println("μ: ", μ, ", σ: ", σ)
        loops[loop]["fss"] = σ / μ;
        loops[loop]["css"] = 0.074401855 / σ;
    end

    if(parsed_args["bias2"] != nothing)
        loop     = parsed_args["bias2"]
        workload = bias2()[1:parsed_args["N"]]
        μ        = mean(workload)
        σ        = stdm(workload, μ, corrected = size(workload)[1] > 1)
        println("μ: ", μ, ", σ: ", σ)
        loops[loop]["fss"] = σ / μ;
        loops[loop]["css"] = 0.074401855 / σ;
    end

    if(parsed_args["bias3"] != nothing)
        loop     = parsed_args["bias3"]
        workload = bias3()[1:parsed_args["N"]]
        μ        = mean(workload)
        σ        = stdm(workload, μ, corrected = size(workload)[1] > 1)
        println("μ: ", μ, ", σ: ", σ)
        loops[loop]["fss"] = σ / μ;
        loops[loop]["css"] = 0.074401855 / σ;
    end

    if(parsed_args["uniform"] != nothing)
        loop     = parsed_args["uniform"]
        workload = bias3()[1:parsed_args["N"]]
        μ        = 10
        σ        = 1.0
        println("μ: ", μ, ", σ: ", σ)
        loops[loop]["fss"] = σ / μ;
        loops[loop]["css"] = 0.074401855 / σ;
    end

    str = JSON.json(loops)

    open(filename, "w") do file
        write(file, str)
    end
end

main(ARGS)
