
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
using ProgressMeter
using Random
using Roots
using SpecialFunctions
using Statistics
using StatsBase

include("particle_gp.jl")
include("acquisition.jl")
include("expmean.jl")
include("expkernel.jl")
include("mcmc.jl")
include("plot_gp.jl")

function optimize_acquisition(gp, verbose::Bool=true)
    ϵ        = 1e-10
    dim      = 1

    f(x, g) = acquisition(x[1], gp)

    opt = NLopt.Opt(:GN_DIRECT, dim)

    opt.lower_bounds  = [0.0]
    opt.upper_bounds  = [1.0]
    #opt.ftol_abs      = 1e-7
    opt.xtol_abs      = 1e-3
    opt.maxeval       = 2^10
    opt.max_objective = f

    val, time, _, _, _ = @timed begin
        opt_y, opt_x, ret = NLopt.optimize(opt, rand(dim))
    end
    opt_y, opt_x = val

    if(verbose)
        @info("Acquisition optimization result",
              acquisition_optimizer = opt_x,
              acquisition_optimum = opt_y,
              status = ret,
              elapsed = time)
    end
    return opt_x[1], opt_y
end

function optimize_mean(gp, verbose::Bool=true)
    ϵ        = 1e-10
    dim      = 1

    f(x, g) = GaussianProcesses.predict_y(gp, x[:,:])[1][1]

    opt = NLopt.Opt(:GN_DIRECT, dim)
    opt.lower_bounds   = [0.0]
    opt.upper_bounds   = [1.0]
    opt.xtol_abs       = 1e-3
    opt.maxeval        = 2^10
    opt.max_objective  = f

    val, time, _, _, _ = @timed begin
        opt_y, opt_x, ret = NLopt.optimize(opt, rand(dim))
    end
    opt_y, opt_x = val

    if(verbose)
        @info("Mean optimization result",
              mean_optimizer=opt_x,
              mean_optimum = opt_y,
              status = ret,
              elapsed = time)
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
        GaussianProcesses.optimize!(gp, method=Optim.LBFGS(linesearch=LineSearches.BackTracking()))
        if(!isnan(gp.target))
            push!(params, GaussianProcesses.get_params(gp))
            push!(like, gp.target)
        end
    end
    idx = argmax(like)
    ParticleGP(gp, hcat([params[idx]]...))
end

function whiten(y)
    μ  = mean(y)
    σ  = stdm(y, μ) 
    y .-= μ
    y ./= σ
    return y
end

function build_gp(data_x, data_y, time_idx, verbose::Bool=true)
    k = begin
        if(time_idx[end] == 1)
            k_x = Mat52Iso(exp(-2.0), exp(1.0), [Normal(-2.0, 2.0), Normal(0.0, 2.0)])
            k   = Masked(k_x, [2])
        else
            # Sum Kernel
            # k = Mat52Ard([exp(1.0), exp(-2.0)], exp(0.0),
            #              [Normal(4.0, 16.0), Normal(-2.0, 2.0), Normal(0.0, 2.0)])

            k_x = Mat52Iso(exp(-2.0), exp(0.0),
                           [Normal(-2.0, 2.0), Normal(0.0, 2.0)])
            k_x = Masked(k_x, [2])

            k_t = Exp(exp(-2.0), exp(0.0), exp(0.0),
                      [Normal(-2.0, 2.0),Normal(0.0, 2.0), Normal(0.0, 2.0)])
            k_t = fix(k_t, :lσ)
            k_t = Masked(k_t, [1])

            # α   = Const(log.(4.0) / 2, [Normal(0.0, 2.0)])
            # α   = fix(α, :lσ)
            # k   = (α*k_x + k_t)

            # k_xt = Mat52Ard([exp(log(time_idx[end]/2)), exp(0.0)], exp(0.0),
            #                [Normal(log(time_idx[end]/2), log(time_idx[end]/2)),
            #                 Normal(-2.0, 2.0),
            #                 Normal(0.0, 2.0)])
            k = k_x + k_t# + k_xt
        end
    end
    #println(size(data_x))
    #println(size(data_y))
    m    = MeanZero() #MeanConst(0.0)
    ϵ    = -2.0
    gp   = GP(data_x, data_y, m, k, ϵ)
    set_priors!(gp.logNoise, [Normal(-2.0, 2.0)])

    #gp = ess(gp, num_samples=2^10, num_adapts=2^10, thinning=2^3, verbose=verbose)
    #gp = nuts(gp, num_samples=512, num_adapts=512,
    #          thinning=4, verbose=verbose)
    #K  = mean([g.cK.mat for g in gp.gp])
    #display(Plots.heatmap(K))

    #GaussianProcesses.optimize!(gp)
    #println(gp)
    gp   = slice(gp, num_samples=1024, num_adapts=1024,
                 thinning=8, width=4.0, verbose=verbose)
    return gp
end

function data_preprocess(data_x, data_y, time_idx;
                         verbose::Bool=true,
                         legacy::Bool=false,
                         samples::Int64=4)
    if(verbose)
        @info "Preprocessing data"
    end

    extrapolating = size(data_x, 1) < time_idx[end]

    if(extrapolating)
        time_idx = time_idx[1:end-1]
        data_x = data_x[time_idx, :]
        data_y = data_y[time_idx, :]
    end

    data_x = vcat(data_x...)'
    data_y = vcat(data_y...)

    num_data = floor(Int64, size(data_x, 2) / length(time_idx))
    iter_axe = vcat(repeat(time_idx, num_data))
    data_x   = hcat(iter_axe, data_x[1,:])

    data_y = whiten(data_y)
    data_x = Array(data_x')
    return data_x, data_y
end

function fit_surrogate(data_x, data_y, time_idx, subsample, verbose=true)
    idx   = zeros(subsample)
    ndata = size(data_x,2)
    if(ndata > subsample)
        StatsBase.knuths_sample!(1:ndata, idx)
        idx = convert(Array{Int64}, idx)

        data_x = data_x[:,idx]
        data_y = data_y[idx]
        @info "subsampling $(subsample) samples from $(ndata) samples"
    end

    if(verbose)
        @info "Fitting gaussian process"
    end
    gp = build_gp(data_x, data_y, time_idx, verbose)
end

function debug_gp(;uniform_quant=false, time_samples=4)
    bostate = open(".bostate.sc_omp.json") do file
        str   = read(file, String)
        loops = JSON.parse(str)
        loops
    end
    loop   = bostate["loops"][1]
    raw_x = loop["hist_x"]
    raw_y = loop["hist_y"]

    raw_x = hcat(raw_x...)
    raw_y = hcat(raw_y...)

    data_x = convert(Array{Float64}, raw_x)
    data_y = convert(Array{Float64}, raw_y)

    time_max = size(raw_x, 1)
    time_idx, time_w = time_quantization(time_max, time_samples, uniform_quant)
    data_x, data_y = data_preprocess(data_x, data_y, time_idx, verbose=true)
    gp = fit_surrogate(data_x, data_y, time_idx, 128, true)
    gp = TimeMarginalizedGP(gp, time_idx, time_w)
    return gp
end

function debug_bo(time_points::Int64=4)
    bostate = open(".bostate.sc_omp.json") do file
        str   = read(file, String)
        loops = JSON.parse(str)
        loops
    end
    loop  = bostate["loops"][1]
    raw_x = loop["hist_x"]
    raw_y = loop["hist_y"]

    raw_x = hcat(raw_x...)
    raw_y = hcat(raw_y...)
    raw_x = raw_x[:, 1:2:end]
    raw_y = raw_y[:, 1:2:end]

    data_x = convert(Array{Float64}, raw_x)
    data_y = convert(Array{Float64}, raw_y)

    data_x = data_x[1:32,:]
    data_y = data_y[1:32,:]
    time_max = size(data_x, 1)
    d = Dict()
    a, b = labo(data_x, -data_y, time_max, time_points, 512; logdict=d)
    return a, b, d
end

function time_quantization(extrapol_idx::Int64,
                           time_max::Int64,
                           time_samples::Int64,
                           uniform_quant::Bool)
    time_idx = []
    time_w   = []
    if(time_max <= time_samples)
        time_idx = collect(1:time_max)
        N = length(time_idx)
        return time_idx, ones(N) / N
    end

    if(uniform_quant)
        time_idx = range(1, stop=time_max, length=time_samples)
    else
        time_idx = range(1, stop=log(time_max), length=time_samples)
        time_idx = ℯ.^time_idx
        time_idx = round.(Int64, time_idx, RoundNearest)
        time_idx[1] = 1.0
    end
    time_idx = vcat(time_idx, extrapol_idx)
    N        = length(time_idx)

    time_w    = zeros(N)
    time_w[1] = 1
    for i = 2:N
        time_w[i] = time_idx[i] - time_idx[i-1]
        @assert time_w[i] > 0.0
    end
    time_idx, time_w
end

function labo(data_x, data_y, extra, time_samples, subsample;
              verbose::Bool=true,
              legacy::Bool=false,
              uniform_quant::Bool=true,
              logdict=nothing)
    η = maximum(data_y)
    time_idx, time_w = time_quantization(extra, size(data_y, 1),
                                         time_samples, uniform_quant)
    data_x, data_y = data_preprocess(data_x, data_y, time_idx,
                                     verbose=true, legacy=legacy)

    gp = fit_surrogate(data_x, data_y, time_idx, subsample, verbose)
    gp = TimeMarginalizedGP(gp, time_idx, time_w)

    if(verbose)
        @info "Sampling y*"
    end
    P         = length(gp.non_marg_gp.weights)
    num_ystar = 32
    num_gridx = 1024
    ηs        = zeros(num_ystar, P)
    for i in P
        ηs[:,i] = sample_ystar(η, num_ystar, num_gridx, gp.non_marg_gp.gp[i])
    end
    gp.non_marg_gp.η = ηs

    if(verbose)
        @info "Optimizing acquisition"
    end
    θ_next, acq_opt = optimize_acquisition(gp, verbose)

    if(verbose)
        @info "Optimizing mean"
    end
    θ_mean, mean_opt = optimize_mean(gp, verbose)

    if(logdict != nothing)
        logα(x) = log(acquisition(x, gp))
        αx = collect(0.0:0.01:1.0)
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
        logdict[:tmgp_x]    = x
        logdict[:tmgp_mean] = μ
        logdict[:tmgp_std]  = sqrt.(σ²)

        x     = range(0.0, stop=1.0, length=50)
        t     = range(1.0, stop=extra, length=50)
        xgrid = repeat(x', 50, 1)
        tgrid = repeat(t, 1, 50)
        tx    = vcat(vec(tgrid)', vec(xgrid)')
        
        μ, Σ  = predict_y(gp.non_marg_gp, tx)

        μ = reshape(μ, 50, 50)
        Σ = reshape(Σ, 50, 50)
        μ = Array(μ')
        Σ = Array(Σ')

        logdict[:gp_x]    = collect(x)
        logdict[:gp_t]    = collect(t)
        logdict[:gp_mean] = μ
        logdict[:gp_std]  = Σ
    end
    return θ_next, θ_mean, acq_opt, mean_opt
end
end
