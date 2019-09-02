
using ForwardDiff
using GaussianProcesses
using Statistics
using Distributions
using ScikitLearn
@sk_import mixture: BayesianGaussianMixture

const fs  = Base.Filesystem

include("bayesian_treatment.jl")
include("particle_gp.jl")
include("acquisition.jl")
include("slice_sampler.jl")

function entropy_upperbound(w, μ, σ)
"""
  Entropy upper bound estimation based on Kullback-Leibler divergence

  Kolchinsky, Artemy, and Brendan Tracey. "Estimating mixture entropy 
  with pairwise distances." Entropy 19.7 (2017): 361.
"""
    @inline function H(σ²)
        return 1/2 * log(2*π*ℯ*σ²)
    end
    n = length(w)
    first_term  = sum([w[i] * H(σ[i]^2) for i = 1:n])
    second_term = 0.0
    if(n == 1)
        return first_term
    end
    for i = 1:n
        σi²   = σ[i]^2
        j_sum = 0.0 
        for j = 1:n
            if(i == j)
                continue
            end
            KL_t1  = log(σ[j] / σ[i])
            KL_t2  = (σi² + (μ[i] - μ[j])^2) / (2.0*σ[j]^2)
            KL     = KL_t1 + KL_t2 - 1/2
            println("$i, $j, $(KL), $(KL_t1), $(KL_t2)")
            j_sum += w[j] * exp(-KL)
        end
        second_term += w[i] * log(j_sum)
    end
    println("$(first_term), $(second_term)")
    return first_term - second_term
end

function IBBO(x, y, verbose::Bool)
    μ  = mean(y)
    σ  = stdm(y, μ) 
    m  = MeanConst(μ)
    k  = SE(0.0, σ)
    set_priors!(k, [Normal(), Normal()])

    if(verbose)
        println("- fitting initial gp")
    end
    k  = fix(k, :lσ)
    ϵ  = -1.0
    gp = GP(x[:,:]', y, m, k, ϵ)
    if(verbose)
        println("- fitting initial gp - done")
        println("- optimizing hyperparameters")
    end

    gp = nuts(gp)
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
    sampler = batch_slice_sampler(α, 100, 1000, 5000)
    if(verbose)
        println("- sampling acquisition function - done")
        println("- fitting gaussian mixture model")
    end

    gmm_config = BayesianGaussianMixture(n_components=10,
                                         max_iter=1000,
                                         weight_concentration_prior=1.0)
    model = fit!(gmm_config, sampler[:,:])
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

