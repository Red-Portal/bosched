
module LABO

using AdvancedHMC
using Base.Threads
using Distributions
using GaussianProcesses
using JSON
using LineSearches
using LinearAlgebra
using MCMCDiagnostics
using NLopt
using Optim
using Plots
using Random
using Roots
using SpecialFunctions
using Statistics
using StatsBase

include("particle_gp.jl")
include("acquisition.jl")
include("expmean.jl")
include("mcmc.jl")
include("plot_gp.jl")

function optimize_acquisition(gp, verbose::Bool=true)
    ϵ        = 1e-10
    dim      = 1

    f(x, g) = acquisition(x[1], gp)

    opt = NLopt.Opt(:GN_DIRECT, dim)

    opt.lower_bounds  = [0.0]
    opt.upper_bounds  = [1.0]
    opt.ftol_abs      = 1e-20
    opt.xtol_abs      = 1e-20
    opt.maxeval       = 1024
    opt.max_objective = f
    opt_y, opt_x, ret = NLopt.optimize(opt, rand(dim))

    if(verbose)
        println("----------------------------------")
        println("Acquisition optimizer = ", opt_x)
        println("Acquisition optimum   = ", opt_y)
        println("Result                = ", ret)
    end
    return opt_x[1], opt_y
end

function optimize_mean(gp, verbose::Bool=true)
    ϵ        = 1e-10
    dim      = 1

    f(x, g) = GaussianProcesses.predict_y(gp, x[:,:])[1][1]

    opt = NLopt.Opt(:GN_DIRECT, dim)
    opt.lower_bounds  = [0.0]
    opt.upper_bounds  = [1.0]
    opt.ftol_abs      = 1e-20
    opt.xtol_abs      = 1e-20
    opt.maxeval       = 1024
    opt.max_objective = f
    opt_y, opt_x, ret = NLopt.optimize(opt, rand(dim))
    if(verbose)
        println("----------------------------------")
        println("Mean optimizer = ", opt_x)
        println("Mean optimum   = ", opt_y)
        println("Result         = ", ret)
    end
    return opt_x[1], opt_y
end

# Overwrite Gaussian Processes's Cholesky routine to not 'retry'
function GaussianProcesses.make_posdef!(m::AbstractMatrix, chol_factors::AbstractMatrix)
    n = size(m, 1)
    size(m, 2) == n || throw(ArgumentError("Covariance matrix must be square"))
    copyto!(chol_factors, m)
    chol = cholesky!(Symmetric(chol_factors, :U))
    return m, chol
end

function make_posdef!(m::AbstractMatrix)
    chol_buffer = similar(m)
    return make_posdef!(m, chol_buffer)
end

function multistart_optimize!(gp::GaussianProcesses.GPBase, restarts::Int64)
    params = Vector{Float64}[]
    like   = Float64[]
    dist   = Uniform(-5, 5)
    nparam = GaussianProcesses.num_params(gp)
    for i = 1:restarts
        θ_init = rand(dist, nparam)
        GaussianProcesses.set_params!(gp, θ_init)
        GaussianProcesses.update_target_and_dtarget!(gp)
        optimize!(gp, method=Optim.LBFGS(linesearch=LineSearches.BackTracking()))
        if(!isnan(gp.ll))
            push!(params, GaussianProcesses.get_params(gp))
            push!(like, gp.ll)
        end
    end
    idx = argmax(like)
    GaussianProcesses.set_params!(gp, params[idx])
    gp
end

function whiten(y)
    μ  = mean(y)
    σ  = stdm(y, μ) 
    y .-= μ
    y ./= σ
    return y
end

function build_gp(data_x, data_y, time_idx, verbose::Bool=true)
    tmax = time_idx[end]
    m    = MeanExp(1.0, 1.0, 1.0, [Normal(0.0, 2.0),
                                   Normal(0.0, 2.0),
                                   Normal(0.0, 2.0)])
    k    = SEArd([1.0, 1.0], 1.0, [Normal(-1.0, max(log(tmax), 2.0)),
                                   Normal(-1.0, 2.0),
                                   Normal(0.0, 2.0)])
    ϵ    = -1.0
    gp   = GP(data_x, data_y, m, k, ϵ)
    set_priors!(gp.logNoise, [Normal(-1.0, 2.0)])
    gp   = ess(gp, num_samples=500, num_adapts=200,
               thinning=5, verbose=verbose)
    # gp = mala(gp, num_samples=1024, num_adapts=1024,
    #           thinning=8, verbose=verbose)
end

function filter_outliers(gp::AbstractParticleGP,
                         data_x,
                         data_y,
                         time_idx,
                         verbose::Bool=true)
    α = log(0.01)
    μ, σ² = predict_y(gp, data_x)
    dists = Normal.(μ, σ²)
    probs = logpdf.(dists, data_y)
    idx   = probs .< α
    if(!all(idx))
        data_x = data_x[:, .!idx]
        data_y = data_y[.!idx]
        gp = build_gp(data_x, data_y, time_idx, verbose)
    end
    if(verbose)
        println("- filtered $(count(idx)) outliers")
    end
    gp
end

function data_preprocess(data_x, data_y, time_idx;
                         verbose::Bool=true,
                         legacy::Bool=false,
                         samples::Int64=4)
    if(verbose)
        println("- data preprocess")
    end

    if(legacy)
        data_x = vcat(data_x...)
        data_y = vcat(data_y...)

        data_x = hcat(data_x...)
        data_y = hcat(data_y...)

        data_x = convert(Array{Float64}, data_x)
        data_y = convert(Array{Float64}, data_y)
    end

    data_x = data_x[time_idx, :]
    data_y = data_y[time_idx, :]
    data_x = vcat(data_x...)'
    data_y = vcat(data_y...)

    @assert size(data_x, 2) % length(time_idx) == 0
    num_data = floor(Int64, size(data_x, 2) / length(time_idx))
    iter_axe = vcat(repeat(time_idx, num_data))
    data_x   = hcat(iter_axe, data_x[1,:])

    data_y = whiten(data_y)
    data_x = Array(data_x')

    if(verbose)
        println("- data preprocess - done")
    end
    return data_x, data_y
end

function fit_surrogate(data_x, data_y, time_idx, subsample, verbose=true)
    iters = length(time_idx)
    idx   = zeros(subsample)

    ndata = size(data_x,2)
    if(ndata > subsample)
        StatsBase.knuths_sample!(1:ndata, idx)
        idx = convert(Array{Int64}, idx)

        data_x = data_x[:,idx]
        data_y = data_y[idx]
        println("- subsampling $(subsample) samples from $(ndata) samples")
    end

    if(verbose)
        println("- fit gp")
    end
    gp = build_gp(data_x, data_y, time_idx, verbose)
    if(verbose)
        println("- fit gp - done")
    end

    if(verbose)
        println("- filter outliers")
    end
    gp = filter_outliers(gp, data_x, data_y, time_idx, verbose)
    if(verbose)
        println("- filter outliers - done")
    end
    gp
end

function debug_gp()
    bostate = open(".bostate.nn.json") do file
        str   = read(file, String)
        loops = JSON.parse(str)
        loops
    end
    loop   = bostate["loops"][1]
    raw_x = loop["hist_x"]
    raw_y = loop["hist_y"]

    data_x, data_y = data_preprocess(raw_x, raw_y)
    gp = fit_gp(data_x, data_y, 256)
    return gp
end

function debug_bo()
    bostate = open(".bostate.nn.json") do file
        str   = read(file, String)
        loops = JSON.parse(str)
        loops
    end
    loop  = bostate["loops"][1]
    raw_x = loop["hist_x"]
    raw_y = loop["hist_y"]

    d = Dict()
    a, b = labo(raw_x, -raw_y, 16, 512, logdict=d, legacy=true)
    return a, b, d
end

function labo(data_x, data_y, time_max, time_samples, subsample;
              verbose::Bool=true, legacy::Bool=false, logdict=nothing)
    η = maximum(data_y)

    time_idx = begin
        if(time_max <= time_samples)
            time_idx = collect(1:time_max)
        else
            time_idx = range(1, stop=time_max, length=time_samples)
            time_idx = round(time_idx, RoundNearest)
        end
    end

    data_x, data_y = data_preprocess(data_x, data_y, time_idx,
                                     verbose=true, legacy=legacy)
    gp = fit_surrogate(data_x, data_y, time_idx, subsample, verbose)
    gp = TimeMarginalizedGP(gp, time_idx)

    if(verbose)
        println("- sampling y*")
    end
    P         = length(gp.non_marg_gp.weights)
    num_ystar = 16
    num_gridx = 512
    gp.non_marg_gp.η = sample_ystar(η, num_ystar, num_gridx, gp)
    if(verbose)
        println("- sampling y* - done")
    end

    if(verbose)
        println("- optimize acquisition")
    end
    θ_next, acq_opt = optimize_acquisition(gp, verbose)
    if(verbose)
        println("- optimize acquisition - done")
    end

    if(verbose)
        println("- optimize mean")
    end
    θ_mean, mean_opt = optimize_mean(gp, verbose)
    if(verbose)
        println("- optimize mean - done")
    end

    if(logdict != nothing)
        logα(x) = log(acquisition(x, gp))
        αx = 0.0:0.01:1.0
        αy = exp.(logα.(αx))
        logdict[:data_x]     = data_x
        logdict[:data_y]     = data_y
        logdict[:acq_x]      = αx
        logdict[:acq_y]      = αy
        logdict[:param_next] = θ_next
        logdict[:acq_next]   = acq_opt
        logdict[:mean_opt]   = mean_opt
        logdict[:gp]         = gp

        μs = Float64[]
        σs = Float64[]
        θ  = collect(0.0:0.01:1.0)

        x     = collect(0.0:0.01:1.0)
        μ, σ² = GaussianProcesses.predict_y(gp, x[:,:])
        logdict[:gp_x]    = x
        logdict[:gp_mean] = μ
        logdict[:gp_std]  = sqrt.(σ²)
    end
    return θ_next, θ_mean, acq_opt, mean_opt
end
end
