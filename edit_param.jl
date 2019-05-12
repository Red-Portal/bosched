
import JSON
import ArgParse
import Base.Filesystem

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

    str = JSON.json(loops)

    open(filename, "w") do file
        write(file, str)
    end
end

main(ARGS)
