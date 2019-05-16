
import Plots
import JSON
import CSV

using DataFrames

function init_gp()
    x = Array{Float64, 1}()
    y = Array{Float64, 1}()
    model = ccall((:test_init, "./libLPBO.so"), Ptr{Cvoid}, 
                  (Ptr{Float64}, Ptr{Float64}, Int64),
                  x, y, 0)
    return model
end

function deserialize_gp(model, serialized)
    n = size(serialized)[1]
    bytes = convert(Array{UInt8, 1}, serialized)
    println("  deserializing...")
    ccall((:test_deserialize, "./libLPBO.so"), Cvoid,
          (Ptr{Cvoid}, Ptr{UInt8}, Int64), model, bytes, n)
    println("  deserializing - done")
    return model
end

function render_gp(model, resolution)
    gp_means = Array{Float64, 1}(undef, resolution)
    gp_vars  = Array{Float64, 1}(undef, resolution)
    ccall((:test_render_gp, "./libLPBO.so"), Cvoid,
          (Ptr{Cvoid}, Int64, Ptr{Float64}, Ptr{Float64}),
          model, resolution, gp_means, gp_vars)
    
    return gp_means, gp_vars
end

function get_history(model, iteration)
    x = Array{Float64, 1}(undef, iteration + 30)
    y = Array{Float64, 1}(undef, iteration + 30)
    n = Ref{Int64}(0)
    ccall((:test_get_history, "./libLPBO.so"), Cvoid,
          (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ref{Int64}),
          model, x, y, n)

    #println(x[1:n[]])
    #println(y[1:n[]])

    return x[1:n[]], y[1:n[]]
    #return Array{Float64, 1}(), Array{Float64, 1}()
end

function print(p,
               resolution,
               gp_mean::Array{Float64, 1},
               gp_var::Array{Float64, 1},
               history_x::Array{Float64, 1},
               history_y::Array{Float64, 1})

    x_range = range(0.0, length=resolution, step=1/resolution)
    Plots.plot!(p, x_range, gp_mean, ribbon=sqrt.(gp_var) * 1.96,
                fillalpha=.3, label="gaussian process", lw=3, reuse=false)
    Plots.plot!(p, history_x, history_y, seriestype=:scatter, color="blue", label="data points")
    #Plots.plot!(p, [x_selected], [y_selected], seriestype=:scatter, color="red", label="selected point")
    Plots.display(p)
    input = readline(stdin)
    return input == "y"
end

function export_data(loop_data,
                     resolution::Int64,
                     gp_mean::Array{Float64, 1},
                     gp_var::Array{Float64, 1},
                     history_x::Array{Float64, 1},
                     history_y::Array{Float64, 1})

    println("-- exporting loop data")
    df = DataFrame(data_x=history_x, data_y=history_y)
    CSV.write("data_points.csv", df)

    df = DataFrame(gp_x=range(0.0, length=resolution, step=1/resolution),
                   gp_means=gp_mean,
                   gp_var=1.96*sqrt.(gp_var))
    CSV.write("gp.csv", df)

    means = loop_data["mean"]
    vars = loop_data["var"]
    acq = loop_data["acq"]
    len = size(means)[1]
    df = DataFrame(iter=1:len,
                   means=means,
                   vars=vars,
                   acq=acq,
                   x=history_x[end-len+1:end],
                   y=history_y[end-len+1:end])
    CSV.write("logs.csv", df)
end

function main(args)
    raw_json = open(args[1]) do file
        read(file, String)
    end
    data = JSON.parse(raw_json)

    println("state file date: ", data["date"])
    println("num loops: ", data["num_loops"])

    model = init_gp()
    p = Plots.plot()

    for i in data["loops"]
        println("-- processing ", i["id"])
        if(i["warmup"] == false)
            resolution = 100
            deserialize_gp(model, i["gp"])
            means, vars = render_gp(model, resolution)
            x, y = get_history(model, i["iteration"])
            export_flag = print(p, resolution, means, vars, x, y)
            if(export_flag)
                export_data(i, resolution, means, vars, x, y)
            end
        end
    end
end

main(ARGS)
