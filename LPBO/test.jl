
using Plots

eps = 0.3;
function objective(x)
    return 10 * (x - 0.4)^2 - 10 +  eps * randn();
end

x = rand(Float64, 20)
y = objective.(x)


ccall((:build, "./libLPBO.so"), Void, (Ptr{Float64}, Ptr{Float64}, Int32), x, y, size(x)[1])

means = zeros(1000)
vars  = zeros(1000)
for i = 1:1000
    m = [means[i]]
    v = [vars[i]]
    ccall((:predict, "./libLPBO.so"), Void, (Float64, Ptr{Float64}, Ptr{Float64}),
          0.001 * (i-1), m, v)
    means[i] = m[1]
    vars[i]  = v[1]
end

μ = means
σ = sqrt.(vars)

x *= 1000

plot([μ, μ], fillrange=[μ+σ, μ-σ], fillalpha=0.3, c=:blue)
scatter!(x, y)
