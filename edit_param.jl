
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
            value["css"] = parsed_args["css"];
        end

        if(parsed_args["fss"] != nothing)
            value["fss"] = parsed_args["fss"];
        end

        if(parsed_args["tss"] != nothing)
            value["tss"] = parsed_args["tss"];
        end

        if(parsed_args["tape"] != nothing)
            value["tape"] = parsed_args["tape"];
        end
    end

    str = JSON.json(loops)

    open(filename, "w") do file
        write(file, str)
    end
end

main(ARGS)
