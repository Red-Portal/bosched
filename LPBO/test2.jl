
using Plots

function main()
    p = plot()

    eps = 2;
    function objective(x)
        return 10 * sin(4.0 * π * x) +  eps * randn();
    end

    function objective_noiseless(x)
        return 10 * sin(4.0 * π * x);
    end

    samples = 10
    x = rand(Float64, samples)
    y = objective.(x)
    var_history = Array{Float64, 1}();
    mean_history = Array{Float64, 1}();
    merit_history = Array{Float64, 1}();

    plot!(p,  0.0:0.001:1.0, objective_noiseless.(0.0:0.001:1.0))
    plot!(p, x, y, seriestype=:scatter)
    display(p)
    readline(stdin)

    #ccall((:build, "./libLPBO.so"), Void, (Ptr{Float64}, Ptr{Float64}, Int32), x, y, size(x)[1])

    resolution = 100
    y_tilda = Array{Float64}(undef, resolution)
    gp_means = Array{Float64}(undef, resolution)
    gp_vars = Array{Float64}(undef, resolution)

    model = ccall((:test_init, "./libLPBO.so"), Ptr{Cvoid}, 
                  (Ptr{Float64}, Ptr{Float64}, Int64),
                  x, y, samples)

    x_selected = 0.0
    y_selected = 0.0

    for i = 1:30
        mean  = Ref{Float64}(0.0);
        merit = Ref{Float64}(0.0);
        var   = Ref{Float64}(0.0);

        x_selected = ccall((:test_next_point, "./libLPBO.so"), Float64,
                           (Ptr{Cvoid}, Float64, Float64, Int64, Int64,
                            Ref{Float64}, Ref{Float64}, Ref{Float64}),
                           model, x_selected, y_selected, i, 200, merit, mean, var)
        y_selected = objective(x_selected)

        ccall((:test_render_acquisition, "./libLPBO.so"), Cvoid,
              (Ptr{Cvoid}, Int64, Int64, Ptr{Float64}),
              model, i, resolution, y_tilda)

        ccall((:test_render_gp, "./libLPBO.so"), Cvoid,
              (Ptr{Cvoid}, Int64, Ptr{Float64}, Ptr{Float64}),
              model, resolution, gp_means, gp_vars)
        
        x_range = range(0.0, length=resolution, step=1/resolution)
        p = plot(x_range, y_tilda, label="acquisition", lw=3)
        plot!(p, x_range, objective_noiseless.(x_range), label="objective", lw=3)
        plot!(p, x, y, seriestype=:scatter, color="blue", label="data points")
        plot!(p, [x_selected], [y_selected], seriestype=:scatter, color="red", label="selected point")
        plot!(p, x_range, gp_means, ribbon=sqrt.(gp_vars) * 0.96,
              fillalpha=.5, label="gaussian process", lw=3)
        display(p)
        readline(stdin)

        push!(x, x_selected)
        push!(y, y_selected)
        push!(mean_history, mean[])
        push!(merit_history, merit[])
        push!(var_history, var[])
        samples += 1

        println("iteration: ", i, " mean: ", mean[], " var: ", var[],
                " x: ", x_selected, " y: ", y_selected)
    end

    str = Vector{UInt8}(undef, 1024 * 1024);
    size = Ref{Int64}(0);
    println("serializing...")
    ccall((:test_serialize, "./libLPBO.so"), Cvoid, (Ptr{Cvoid}, Ref{UInt8}, Ref{Int64}), model, str, size)
    println("deserializing...")
    ccall((:test_deserialize, "./libLPBO.so"), Cvoid, (Ptr{Cvoid}, Ptr{UInt8}, Int64), model, str, size[])
    println("...done!")

    for i = 31:100
        mean  = Ref{Float64}(0.0);
        merit = Ref{Float64}(0.0);
        var   = Ref{Float64}(0.0);

        x_selected = ccall((:test_next_point, "./libLPBO.so"), Float64,
                           (Ptr{Cvoid}, Float64, Float64, Int64, Int64,
                            Ref{Float64}, Ref{Float64}, Ref{Float64}),
                           model, x_selected, y_selected, i, 200, merit, mean, var)
        y_selected = objective(x_selected)

        ccall((:test_render_acquisition, "./libLPBO.so"), Float64,
              (Ptr{Cvoid}, Int64, Int64, Ptr{Float64}),
              model, i, resolution, y_tilda)

        ccall((:test_render_gp, "./libLPBO.so"), Cvoid,
              (Ptr{Cvoid}, Int64, Ptr{Float64}, Ptr{Float64}),
              model, resolution, gp_means, gp_vars)

        x_range = range(0.0, length=resolution, step=1/resolution)
        p = plot(x_range, y_tilda, label="acquisition", lw=3)
        plot!(p, x_range, objective_noiseless.(x_range), label="objective", lw=3)
        plot!(p, x, y, seriestype=:scatter, color="blue", label="data points")
        plot!(p, [x_selected], [y_selected], seriestype=:scatter, color="red", label="selected point")
        plot!(p, x_range, gp_means, ribbon=gp_vars, fillalpha=.5, label="gaussian process", lw=3)
        display(p)
        readline(stdin)

        push!(x, x_selected)
        push!(y, y_selected)
        push!(mean_history, mean[])
        push!(merit_history, merit[])
        push!(var_history, var[])
        samples += 1
        println("iteration: ", i, " mean: ", mean[], " var: ", var[],
                " x: ", x_selected, " y: ", y_selected)
    end

    plot!(1:30, merit_history)
end 

main()
