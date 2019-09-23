using Suppressor
using ForwardDiff
using GaussianProcesses
using Statistics
using Distributions
using ScikitLearn
using StatsFuns
@sk_import mixture: BayesianGaussianMixture

import GaussianProcesses.predict_y

const fs  = Base.Filesystem

include("bayesian_treatment.jl")
include("particle_gp.jl")
include("acquisition.jl")
include("slice_sampler.jl")

function GaussianProcesses.predict_y(gp::PrecomputedParticleGP, x)
    bundle = GaussianProcesses.predict_y.(gp.gp, Ref(x))
    μ      = [elem[1] for elem in bundle]
    σ²     = [elem[2] for elem in bundle]

    μ  = hcat(μ...) * gp.weights
    σ² = hcat(σ²...) * gp.weights
    return μ, σ²
end

function entropy_upperbound(w, μ, σ)
"""
  Entropy upper bound estimation based on Kullback-Leibler divergence

  Kolchinsky, Artemy, and Brendan Tracey. "Estimating mixture entropy 
  with pairwise distances." Entropy 19.7 (2017): 361.
"""
    @inline function H(σ²)
        return 1/2 * log(2*π*ℯ*σ²)
    end
    n    = length(w)
    logw = log.(w)
    first_term  = sum([w[i] * H(σ[i]^2) for i = 1:n])
    second_term = 0.0
    if(n == 1)
        return first_term
    end
    for i = 1:n
        σi²   = σ[i]^2
        logKL = zeros(Float64, n)
        for j = 1:n
            KL_t1    = log(σ[j] / σ[i])
            KL_t2    = (σi² + (μ[i] - μ[j])^2) / (2.0*σ[j]^2)
            KL       = KL_t1 + KL_t2 - 1/2
            logKL[j] = logw[j] - KL
        end
        second_term += w[i] * logsumexp(logKL)
    end
    return first_term - second_term
end

function IBBO(x, y, verbose::Bool)
    μ  = mean(y)
    σ  = stdm(y, μ) 
    x .-= μ
    x ./= σ

    m  = MeanConst(0.0)
    k  = SE(0.0, 1.0)
    set_priors!(k, [Normal(1.0,2.0), Normal(1.0,2.0)])
    k  = fix(k, :lσ)

    if(verbose)
        println("- fitting initial gp")
    end
    #k  = fix(k, :lσ)
    ϵ  = -1.0
    gp = GP(x[:,:]', y, m, k, ϵ)
    set_priors!(gp.logNoise, [Normal(1.0,2.0)])
    if(verbose)
        println("- fitting initial gp - done")
        println("- optimizing hyperparameters")
    end
    gp = slice(gp, 200, 10, thinning=1)
    if(verbose)
        println("- optimizing hyperparameters - done")
        println("- sampling y*")
    end

    η         = maximum(y)
    P         = length(gp.weights)
    num_ystar = 100
    num_gridx = 1000
    ystar = Matrix{Float64}(undef, P, 100)
    for i = 1:P
        ystar[i,:] = sample_ystar(i, η, num_ystar, num_gridx, gp)
    end
    gp.particles = vcat(gp.particles, ystar')
    if(verbose)
        println("- sampling y* - done")
        println("- sampling acquisition function ")
    end

    α(x) = acquisition(x, gp)
    samples = batch_slice_sampler(α, 16, 300, 100, 1000)
    if(verbose)
        println("- sampling acquisition function - done")
        println("- fitting gaussian mixture model")
    end

    gmm_config = BayesianGaussianMixture(n_components=10,
                                         max_iter=1000,
                                         weight_concentration_prior=1.0)
    model = fit!(gmm_config, samples[:,:])
    w = model.weights_[:,1,1]
    μ = model.means_[:,1,1]
    σ = sqrt.(model.covariances_[:,1,1])

    # Truncate mixtures at weight probability .1%
    trunc = w .> 0.001
    w  = w[trunc]
    w /= sum(w)
    μ  = μ[trunc]
    σ  = σ[trunc]
    if(verbose)
        println("- fitting gaussian mixture model - done")
        println("- computing entropy bound")
    end

    H = entropy_upperbound(w, μ, σ)
    if(verbose)
        println("- computing entropy bound - done")
        println(" num mixtures = $(length(w)), entropy bound = $(H)")
    end
    return w, μ, σ, H
end

function IBBO_log(x, y, verbose::Bool)
    μ  = mean(y)
    σ  = stdm(y, μ) 
    y .-= μ
    y ./= -σ # Sign flipping for converting to maximiation

    m  = MeanConst(0.0)
    k  = SE(0.0, 1.0)
    set_priors!(k, [Normal(1.0,2.0), Normal(1.0,2.0)])
    k  = fix(k, :lσ)

    if(verbose)
        println("- fitting initial gp")
    end
    k  = fix(k, :lσ)
    ϵ  = -1.0
    gp = GP(x[:,:]', y, m, k, ϵ)
    set_priors!(gp.logNoise, [Normal(1.0,2.0)])
    if(verbose)
        println("- fitting initial gp - done")
        println("- optimizing hyperparameters")
    end
    gp = slice(gp, 200, 10, thinning=1)
    gp_gridx = collect(0.0:0.01:1.0)
    μs, σs   = predict_y(gp, gp_gridx')
    gp_gridμ = μs
    gp_gridσ = σs

    if(verbose)
        println("- optimizing hyperparameters - done")
        println("- sampling y*")
    end

    η         = maximum(y)
    P         = length(gp.weights)
    num_ystar = 100
    num_gridx = 1000
    ystar = Matrix{Float64}(undef, P, 100)
    for i = 1:P
        ystar[i,:] = sample_ystar(i, η, num_ystar, num_gridx, gp)
    end
    gp.particles = vcat(gp.particles, ystar')
    if(verbose)
        println("- sampling y* - done")
        println("- sampling acquisition function ")
    end

    α(x) = acquisition(x, gp)
    α_gridx = collect(0.0:0.01:1.0)
    α_gridy = α.(α_gridx)
    samples = batch_slice_sampler(α, 16, 300, 100, 1000)
    if(verbose)
        println("- sampling acquisition function - done")
        println("- fitting gaussian mixture model")
    end

    gmm_config = BayesianGaussianMixture(n_components=10,
                                         max_iter=1000,
                                         weight_concentration_prior=1.0)
    model = fit!(gmm_config, samples[:,:])
    w = model.weights_[:,1,1]
    μ = model.means_[:,1,1]
    σ = sqrt.(model.covariances_[:,1,1])

    # Truncate mixtures at weight probability .1%
    trunc = w .> 0.001
    w  = w[trunc]
    w /= sum(w)
    μ  = μ[trunc]
    σ  = σ[trunc]
    if(verbose)
        println("- fitting gaussian mixture model - done")
        println("- computing entropy bound")
    end

    H = entropy_upperbound(w, μ, σ)
    if(verbose)
        println("- computing entropy bound - done")
        println(" num mixtures = $(length(w)), entropy bound = $(H)")
    end
    return w, μ, σ, H, α_gridx, α_gridy, gp_gridx, gp_gridμ, gp_gridσ, samples
end
