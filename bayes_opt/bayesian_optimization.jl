
using DiffResults
using Distributions
using Distances
using ForwardDiff
using LinearAlgebra
using Optim
using JuMP, Ipopt
using Sobol
using Base.Threads

abstract type Acquisition end

mutable struct BOContext{Acq <: Acquisition, GP <: autobo.GPBase}
    gp::GP
    optimum_x::Vector{Float64}
    optimum_y::Float64
    iteration::Int64
    acquisition::Vector{Float64}

    function BOContext(gp::autobo.GPBase,
                       acq::Type{<:Acquisition},
                       x::Matrix{Float64},
                       y::Vector{Float64},
                       iteration::Int64)
        idx = argmax(y)
        return new{acq, typeof(gp)}(gp, x[:,idx], y[idx], iteration, Float64[])
    end

    function BOContext(gp::autobo.GPBase,
                       acq::Type{<:Acquisition},
                       optimum_x::Vector{Float64},
                       optimum_y::Float64,
                       iteration::Int64)
        idx = argmax(optimum_y)
        return new{acq, typeof(gp)}(
            gp, optimum_x[idx:idx], #'Matrix'ification
            optimum_y[idx], iteration, Float64[])
    end
end

function finite_diff(f, x, ε = 1e-4)
"""
 For autodiff debugging purposes. Check the symbolic gradient with numerical 
 gradients!
"""
    ∇ = zeros(Float64, length(x))
    for i in 1:length(x)
        p    = zeros(Float64, length(x)) 
        p[i] = 1
        ∇[i] = (f(x + ε*p) - f(x)) / ε
    end
    return ∇
end

"""
Automatic Differentiation (AD) snippet (just in case for the future)

 diff_result = ForwardDiff.hessian!(
     diff_result, acq_x -> acquisition(acqtype, acq_x, boctx), x)
 fx = DiffResults.value(diff_result)
 ∇f = DiffResults.gradient(diff_result)
 Hf = DiffResults.hessian(diff_result)
"""

abstract type InnerOpt end

abstract type NelderMead <: InnerOpt end

include("optimizers/newton.jl")
include("optimizers/admm.jl")
include("optimizers/pproxpda.jl")
include("optimizers/next.jl")

function next_point!(opttype::Type{<:InnerOpt},
                     restarts::Int64,
                     boctx::BOContext,
                     verbose::Int64 = 1)
    max_iter = 1000
    next_x, acq = optimize_acquisition(
        opttype, restarts, max_iter, boctx, verbose)
    boctx.iteration += 1
    push!(boctx.acquisition, acq)
    return next_x
end
