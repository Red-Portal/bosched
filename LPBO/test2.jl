
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
y_hat = Array{Float64, 1}();

plt.plot(0.0:0.001:1.0, objective_noiseless(0.0:0.001:1.0))
plt.scatter(x, y)
plt.show()

#ccall((:build, "./libLPBO.so"), Void, (Ptr{Float64}, Ptr{Float64}, Int32), x, y, size(x)[1])

resolution = 100
x_tilda = Array{Float64}(0.0:(1/resolution):1.0)
y_tilda = Array{Float64}(resolution + 1)

for i = 1:30
    x_selected = ccall((:next_point, "./libLPBO.so"), Float64,
                       (Ptr{Float64}, Ptr{Float64}, Int64, Int64, Int64),
                       x, y, samples, i, 500)
    y_selected = objective(x_selected)
    println("iteration: ", i, " x: ", x_selected, " y: ", y_selected)


    ccall((:render_acquisition, "./libLPBO.so"), Float64,
          (Ptr{Float64}, Ptr{Float64}, Int64, Int64, Ptr{Float64}, Ptr{Float64}, Int64),
          x, y, samples, i, x_tilda, y_tilda, resolution)

    plt.plot(x_tilda, y_tilda)
    plt.plot(0.0:0.001:1.0, objective_noiseless(0.0:0.001:1.0))
    plt.scatter(x, y, color="blue")
    plt.scatter(x_selected, y_selected, color="red")
    plt.show()

    push!(x, x_selected)
    push!(y, y_selected)
    push!(y_hat, y_selected)
    samples += 1
end

plt.plot(1:30, y_hat)
plt.show()
