
function effective_sample_size(X::AbstractMatrix)
    function compute_ess(ρ_scalar)
        τ_inv = 1 + 2 * ρ_scalar[1]
        K = 2
        for k = 2:2:N-2
            Δ = ρ_scalar[k] + ρ_scalar[k + 1]
            if all(Δ < 0)
                break
            else
                τ_inv += 2*Δ
            end
        end
        return min(1 / τ_inv, one(τ_inv))
    end

    N = size(X, 1)
    lags = collect(1:N-1)
    ρ    = zeros(length(lags), size(X, 2))
    StatsBase.autocor!(ρ, X, lags)
    return [compute_ess(ρ[:,i]) for i = 1:size(X,2)] .* N 
end


function nuts(gp; num_samples=100, num_adapts=100, thinning=1, verbose=false)
    params_kwargs = GaussianProcesses.get_params_kwargs(
        gp; domean=false, kern=true, noise=true, lik=true)
    function logp∂logp(θ::AbstractVector)
        try
            GaussianProcesses.set_params!(gp, θ; params_kwargs...)
            GaussianProcesses.update_target_and_dtarget!(gp; params_kwargs...)
        catch
            return (-Inf, zeros(length(θ)))
        end
        return (gp.target, gp.dtarget)
    end

    function logp(θ::AbstractVector)
        return gp.mll
    end

    θ_init  = GaussianProcesses.get_params(gp; params_kwargs...)
    metric  = DenseEuclideanMetric(length(θ_init))
    h       = Hamiltonian(metric, logp, logp∂logp)
    prop    = NUTS(Leapfrog(find_good_eps(h, θ_init)))
    adaptor = StanHMCAdaptor(num_adapts,
                             Preconditioner(metric),
                             NesterovDualAveraging(0.65, prop.integrator.ϵ))

    # Draw samples via simulating Hamiltonian dynamics
    # - `samples` will store the samples
    # - `stats` will store statistics for each sample
    samples, stats = AdvancedHMC.sample(
        h, prop, θ_init, num_samples, adaptor,
        num_adapts; progress=verbose)
    samples = hcat(samples...)
    samples = samples[:,1:thinning:end]
    ESS = [MCMCDiagnostics.effective_sample_size(samples[i,:])
           for i = 1:size(samples,1)]
    println("ESS = $(ESS)")
    return ParticleGP(gp, samples)
end

"""
    ess(gp::GPBase; kwargs...)
Sample GP hyperparameters using the elliptical slice sampling algorithm described in,
Murray, Iain, Ryan P. Adams, and David JC MacKay. "Elliptical slice sampling." 
Journal of Machine Learning Research 9 (2010): 541-548.
Requires hyperparameter priors to be Gaussian.
"""
function ess(gp; num_samples=100, num_adapts=100, thinning=1, verbose::Bool=true)
    params_kwargs = GaussianProcesses.get_params_kwargs(gp; domean=false,
                                                        kern=true,
                                                        noise=true,
                                                        lik=true)
    count = 0
    function calc_target!(θ::AbstractVector)
        count += 1
        try
            GaussianProcesses.set_params!(gp, θ; params_kwargs...)
            GaussianProcesses.update_target!(gp; params_kwargs...)
            return gp.target
        catch err
            if(!all(isfinite.(θ))
               || isa(err, ArgumentError)
               || isa(err, LinearAlgebra.PosDefException))
                return -Inf
            else
                throw(err)
            end
        end
    end

    function sample!(f::AbstractVector)
        v     = GaussianProcesses.sample_params(gp; params_kwargs...)
        u     = rand()
        logy  = calc_target!(f) + log(u);
        θ     = rand()*2*π;
        θ_min = θ - 2*π;
        θ_max = θ;
        f_prime = f * cos(θ) + v * sin(θ);
        props = 1
        while calc_target!(f_prime) <= logy
            props += 1
            if θ < 0
                θ_min = θ;
            else
                θ_max = θ;
            end
            θ = rand() * (θ_max - θ_min) + θ_min;
            f_prime = f * cos(θ) + v * sin(θ);
        end
        return f_prime, props
    end

    total_proposals = 0
    θ_cur = GaussianProcesses.get_params(gp; params_kwargs...)
    D = length(θ_cur)
    post = Array{Float64}(undef, num_samples+num_adapts, D)

    for i = 1:num_samples+num_adapts
        θ_cur, num_proposals = sample!(θ_cur)
        post[i,:] = θ_cur
        total_proposals += num_proposals
    end
    post = post[num_adapts:thinning:end,:]
    samples = post'
    if(verbose)
        println("ESS = ", effective_sample_size(post))
        println("Number of function calls: ", count)
        println("Acceptance rate: $(num_samples / count) \n")
    end
    return ParticleGP(gp, samples)
end

