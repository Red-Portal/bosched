
using PyCall
@pyimport matplotlib.pyplot as plt

eps = 2;
function objective(x)
    return 10 * sin(4 * π * x) +  eps * randn();
end

function objective_noiseless(x)
    return 10 * sin(4 * π * x);
end

samples = 10
x = rand(Float64, samples)
y = objective.(x)
var_history = Array{Float64, 1}();
mean_history = Array{Float64, 1}();
merit_history = Array{Float64, 1}();

plt.plot(0.0:0.001:1.0, objective_noiseless(0.0:0.001:1.0))
plt.scatter(x, y)
plt.show()

#ccall((:build, "./libLPBO.so"), Void, (Ptr{Float64}, Ptr{Float64}, Int32), x, y, size(x)[1])

resolution = 100
x_tilda = Array{Float64}(0.0:(1/resolution):1.0)
y_tilda = Array{Float64}(resolution + 1)

model = ccall((:test_init, "./libLPBO.so"), Ptr{Void}, 
              (Ptr{Float64}, Ptr{Float64}, Int64),
              x, y, samples)

x_selected = 0
y_selected = 0

for i = 1:30
    mean  = Ref{Float64}(0.0);
    merit = Ref{Float64}(0.0);
    var   = Ref{Float64}(0.0);

    x_selected = ccall((:test_next_point, "./libLPBO.so"), Float64,
                       (Ptr{Void}, Float64, Float64, Int64, Int64,
                        Ref{Float64}, Ref{Float64}, Ref{Float64}),
                       model, x_selected, y_selected, i, 200, merit, mean, var)
    y_selected = objective(x_selected)

    ccall((:test_render_acquisition, "./libLPBO.so"), Float64,
          (Ptr{Void}, Ptr{Float64}, Ptr{Float64},
           Int64, Int64, Ptr{Float64}, Ptr{Float64}, Int64),
          model, x, y, samples, i, x_tilda, y_tilda, resolution)

    plt.plot(x_tilda, y_tilda)
    plt.plot(0.0:0.001:1.0, objective_noiseless(0.0:0.001:1.0))
    plt.scatter(x, y, color="blue")
    plt.scatter(x_selected, y_selected, color="red")
    plt.show()

    push!(x, x_selected)
    push!(y, y_selected)
    push!(mean_history, mean[])
    push!(merit_history, merit[])
    push!(var_history, var[])
    samples += 1

    println("iteration: ", i, " mean: ", mean[], " var: ", var[],
            " x: ", x_selected, " y: ", y_selected)
end

str = Vector{UInt8}(1024 * 1024);
size = Ref{Int64}(0);
println("serializing...")
ccall((:test_serialize, "./libLPBO.so"), Void, (Ptr{Void}, Ref{UInt8}, Ref{Int64}), model, str, size)
println("deserializing...")
ccall((:test_deserialize, "./libLPBO.so"), Void, (Ptr{Void}, Ptr{UInt8}, Int64), model, str, size[])
println("...done!")

for i = 31:100
    mean  = Ref{Float64}(0.0);
    merit = Ref{Float64}(0.0);
    var   = Ref{Float64}(0.0);

    x_selected = ccall((:test_next_point, "./libLPBO.so"), Float64,
                       (Ptr{Void}, Float64, Float64, Int64, Int64,
                        Ref{Float64}, Ref{Float64}, Ref{Float64}),
                       model, x_selected, y_selected, i, 200, merit, mean, var)
    y_selected = objective(x_selected)

    ccall((:test_render_acquisition, "./libLPBO.so"), Float64,
          (Ptr{Void}, Ptr{Float64}, Ptr{Float64},
           Int64, Int64, Ptr{Float64}, Ptr{Float64}, Int64),
          model, x, y, samples, i, x_tilda, y_tilda, resolution)

    plt.plot(x_tilda, y_tilda)
    plt.plot(0.0:0.001:1.0, objective_noiseless(0.0:0.001:1.0))
    plt.scatter(x, y, color="blue")
    plt.scatter(x_selected, y_selected, color="red")
    plt.show()

    push!(x, x_selected)
    push!(y, y_selected)
    push!(mean_history, mean[])
    push!(merit_history, merit[])
    push!(var_history, var[])
    samples += 1
    println("iteration: ", i, " mean: ", mean[], " var: ", var[],
            " x: ", x_selected, " y: ", y_selected)
end

plt.plot(1:30, merit_history)
#plt.plot(1:30, var_history)
plt.show()
