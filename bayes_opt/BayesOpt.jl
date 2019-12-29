
using AdvancedHMC
using Distributions
using GaussianProcesses
using JSON
using LineSearches
using LinearAlgebra
using MCMCDiagnostics
using Optim
using Random
using Roots
using SpecialFunctions
using StatsBase
using Statistics
using Plots
using Base.Threads

include("particle_gp.jl")
include("acquisition.jl")
include("expkernel.jl")
include("expmean.jl")
include("mcmc.jl")
include("plot_gp.jl")

function optimize_acquisition(gp, verbose)
    ϵ        = 1e-10
    dim      = 1

    f(x, g) = acquisition(x[1], gp)

    opt = NLopt.Opt(:GN_DIRECT, dim)

    opt.lower_bounds  = [0]
    opt.upper_bounds  = [1]
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

function optimize_mean(gp, verbose)
    ϵ        = 1e-10
    dim      = 1

    f(x, g) = GaussianProcesses.predict_y(gp, x[:,:])[1][1]

    opt = NLopt.Opt(:GN_DIRECT, dim)
    opt.lower_bounds  = [0]
    opt.upper_bounds  = [1]
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

function build_gp(data_x, data_y, verbose::Bool=true)
    tmax = maximum(@view data_x[:,1])
    σ1   = log(2)
    m    = MeanExp(1.0, 1.0, 1.0, [Normal(0.0, σ1),
                                   Normal(0.0, σ1),
                                   Normal(0.0, σ1)])
    k  = SEArd([1.0, 1.0], 1.0, [Normal(0.0, max(log(tmax), σ1)),
                                 Normal(0.0, σ1),
                                 Normal(0.0, σ1)])
    ϵ  = -1.0
    gp = GP(data_x, data_y, m, k, ϵ)
    set_priors!(gp.logNoise, [Normal(0.0, σ1)])

    gp = ess(gp, num_samples=1000, num_adapts=200, thinning=5, verbose=verbose)
end

function filter_outliers(gp::AbstractParticleGP,
                         data_x,
                         data_y,
                         verbose::Bool=true)
    α = log(0.01)
    μ, σ² = predict_y(gp, data_x)
    dists = Normal.(μ, σ²)
    probs = logpdf.(dists, data_y)
    idx   = probs .< α
    if(!all(idx))
        data_x = data_x[:, .!idx]
        data_y = data_y[.!idx]
        gp = build_gp(data_x, data_y, verbose)
    end
    if(verbose)
        println("- filtered $(count(idx)) outliers")
    end
    gp
end

function data_preprocess(data_x, data_y, verbose::Bool=true)
    if(verbose)
        println("- data preprocess")
    end
    data_x = vcat(data_x...)
    data_y = vcat(data_y...)
    trans_x = [] 

    for i in 1:length(data_x)
        batch  = convert(Vector{Float64}, data_x[i])
        zipped = hcat(collect(1:length(batch)), batch)
        push!(trans_x, zipped)
    end
    data_x = vcat(trans_x...)
    data_y = vcat(data_y...)

    data_x = convert(Array{Float64, 2}, data_x)
    data_y = convert(Array{Float64, 1}, data_y)
    data_y = whiten(data_y)
    data_x = Array(data_x')
    if(verbose)
        println("- data preprocess - done")
    end
    return data_x, data_y
end

function fit_gp(data_x, data_y, subsample, verbose=true)
    iters = length(data_x[1])
    idx   = zeros(subsample)

    StatsBase.knuths_sample!(1:size(data_x,2), idx)
    idx = convert(Array{Int64}, idx)

    data_x = data_x[:,idx]
    data_y = data_y[idx]

    if(verbose)
        println("- fit gp")
    end
    gp = build_gp(data_x, data_y, verbose)
    if(verbose)
        println("- fit gp - done")
    end

    if(verbose)
        println("- filter outliers")
    end
    gp = filter_outliers(gp, data_x, data_y, verbose)
    if(verbose)
        println("- filter outliers - done")
    end
    gp
    #return gp, data_x, data_y
    # μs = Float64[]
    # σs = Float64[]
    # θs = collect(0.0:0.01:1.0)
    # for θ = θs
    #     x = hcat(1.0:1.0:iters, ones(iters)*θ)
    #     μ, σ² = predict_y(gp, x')
    #     μ  = # μ[end]
    #     mean(μ)
    #     σ² = # σ²[end]
    #     mean(σ²)
    #     push!(μs, μ)
    #     push!(σs, sqrt(σ²))
    # end
    # return θs, μs, σs
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
    loop   = bostate["loops"][1]
    raw_x = loop["hist_x"]
    raw_y = loop["hist_y"]

    a, b, c = LABO(raw_x, raw_y, 16, 256)
    return a, b, c
end

function LABO(data_x, data_y, max_t, subsample, verbose=true)
    data_x, data_y = data_preprocess(data_x, data_y, verbose)
    gp             = fit_gp(data_x, data_y, subsample, verbose)
    println(size(gp.gp[1].x))

    if(verbose)
        println("- sampling y*")
    end
    η         = maximum(data_y)
    P         = length(gp.weights)
    num_ystar = 32
    num_gridx = 256
    ystar = Matrix{Float64}(undef, P, num_ystar)
    Threads.@threads for i = 1:P
        ystar[i,:] = marg_sample_ystar(i, η, num_ystar, num_gridx, gp, max_t)
    end
    gp.particles = vcat(gp.particles, ystar')
    if(verbose)
        println("- sampling y* - done")
    end

    logα(x) = log(marg_acquisition(x, gp, max_t))
    x = 0.0:0.01:1.0
    y = exp.(logα.(x))
    #optimize_acquisition(gp, verbose)
    return gp, x, y
end
