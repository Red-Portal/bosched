
using AdvancedHMC

function sample_params(gp::GPE; noise::Bool=true, domean::Bool=true,
                       kern::Bool=true, prng=Random.GLOBAL_RNG)
    params = Float64[]
    if noise
        noise_sample = rand(prng, Product(
            GaussianProcesses.get_priors(gp.logNoise)))
        push!(params, noise_sample...)
    end

    if domean
        mean_sample = rand(prng, Product(
            GaussianProcesses.get_priors(gp.mean)))
        append!(params, mean_sample...)
    end
    if kern
        kernel_sample = rand(prng, Product(
            GaussianProcesses.get_priors(gp.kernel)))
        append!(params, kernel_sample...)
    end
    return params
end

function slice(gp::GPBase,
               num_samples=100,
               num_adapts=100;
               thinning=10,
               prng=Random.GLOBAL_RNG,
               verbose=false,
               adaptor=nothing)
    function likeprior(θ::AbstractVector)
        try
            GaussianProcesses.set_params!(gp, θ; domean=false, kern=true, noise=true)
            GaussianProcesses.update_target!(gp; domean=false, kern=true, noise=true)
            return gp.mll
        catch err
            if(!all(isfinite.(θ_prop))
               || isa(err, ArgumentError)
               || isa(err, LinearAlgebra.PosDefException))
                if(verbose)
                    println("Warning: rejecting sample due to numerical error.")
                end
                return -Inf
            else
                throw(err)
            end
        end
    end

    function ess(f::AbstractArray{Float64, 1}, likelihood)
        v     = sample_params(gp, domean=false)
        u     = rand(prng)
        logy  = likelihood(f) + log(u);
        θ     = rand(prng)*2*π;
        θ_min = θ - 2*π;
        θ_max = θ;
        f_prime = f * cos(θ) + v * sin(θ);
        while  likelihood(f_prime) <= logy
            if θ < 0
                θ_min = θ;
            else
                θ_max = θ;
            end
            θ = rand(prng) * (θ_max - θ_min) + θ_min;
            f_prime = f * cos(θ) + v * sin(θ);
        end
        return f_prime
    end
    θ_init  = GaussianProcesses.get_params(
        gp; domean=false, kern=true, noise=true)
    θ_curr  = deepcopy(θ_init)
    samples = zeros(size(θ_init, 1), num_samples)

    for i = 1:num_adapts
        θ_curr = ess(θ_curr, likeprior)
    end

    for i = 1:num_samples
        for j = 1:thinning
            θ_curr = ess(θ_curr, likeprior)
        end
        samples[:,i] = θ_curr
    end
    return PrecomputedParticleGP(gp, samples)
end

function nuts(gp, num_samples=100, num_adapts=100; verbose=false)
    function logp∂logp(θ::AbstractVector)
        GaussianProcesses.set_params!(
            gp, θ; domean=false, kern=true, noise=true)
        GaussianProcesses.update_target_and_dtarget!(
            gp; domean=false, kern=true, noise=true)
        return (gp.target, gp.dtarget)
    end

    function logp(θ::AbstractVector)
        return gp.mll
    end

    θ_init  = GaussianProcesses.get_params(
        gp; domean=false, kern=true, noise=true)
    metric  = DiagEuclideanMetric(2)
    h       = Hamiltonian(metric, logp, logp∂logp)
    prop    = NUTS(Leapfrog(find_good_eps(h, θ_init)))
    adaptor = StanHMCAdaptor(num_adapts,
                             Preconditioner(metric),
                             NesterovDualAveraging(0.8, prop.integrator.ϵ))

    # Draw samples via simulating Hamiltonian dynamics
    # - `samples` will store the samples
    # - `stats` will store statistics for each sample
    samples, stats = AdvancedHMC.sample(
        h, prop, θ_init, num_samples, adaptor,
        num_adapts; progress=verbose)
    samples = hcat(samples...)
    return PrecomputedParticleGP(gp, samples)
end

function hmc(gp; ε=0.01, iteration::Int64=1000, burnin::Int64=100,
             thinning::Int64=1, precomputed=false)
    orig_param = GaussianProcesses.get_params(
        gp; noise=true, domean=true, kern=true);
    chain = mcmc(gp; burn=burnin, nIter=iteration*thinning+burnin, ε=ε,
                 domean=false, thin=thinning)
    return PrecomputedParticleGP(gp, chain)
end
