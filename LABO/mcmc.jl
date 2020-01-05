
function slice(gp; num_samples=100, num_adapts=100, thinning=1, verbose=false,
               width=2.0)
    params_kwargs = GaussianProcesses.get_params_kwargs(
        gp; domean=true, kern=true, noise=true, lik=true)
    θ_cur = GaussianProcesses.get_params(gp; params_kwargs...)
    count = 0
    function targetlogp(θ::AbstractVector)
        count += 1
        try
            GaussianProcesses.set_params!(gp, θ; params_kwargs...)
            GaussianProcesses.update_target!(gp; params_kwargs...)
        catch
            return -Inf
        end
        return gp.target
    end

    function sample!(θ::AbstractArray, w::Float64, logp)
        p0 = logp(θ) + log(rand())
        n  = length(θ)

        lower = θ - w .* rand(n)
        upper = lower .+ w

        x = w .* rand(n) + lower
        while logp(x) < p0
            for i in 1:n
                value = x[i]
                if value < θ[i]
                    lower[i] = value
                else
                    upper[i] = value
                end
                x[i] = rand(Uniform(lower[i], upper[i]))
            end
        end
        x
    end

    samples = zeros(Float64, length(θ_cur), num_samples)
    for i = 1:num_adapts
        θ_cur = sample!(θ_cur, width, targetlogp)
    end

    for i = 1:num_samples
        θ_cur        = sample!(θ_cur, width, targetlogp)
        samples[:,i] = θ_cur
    end
    # ESS = [MCMCDiagnostics.effective_sample_size(samples[i,:])
    #        for i = 1:size(samples,1)]
    # println("ESS = $(ESS)")
    if(verbose)
        println("Number of function calls: ", count)
    end
    return ParticleGP(gp, samples)
end


function nuts(gp; num_samples=100, num_adapts=100, thinning=1, verbose=false)
    params_kwargs = GaussianProcesses.get_params_kwargs(
        gp; domean=true, kern=true, noise=true, lik=true)
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
    #metric  = DenseEuclideanMetric(length(θ_init))
    metric  = DiagEuclideanMetric(length(θ_init))
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
    params_kwargs = GaussianProcesses.get_params_kwargs(
        gp; domean=true, kern=true, noise=true, lik=true)
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
        println("Number of function calls: ", count)
        println("Acceptance rate: $(num_samples / count) \n")
    end
    return ParticleGP(gp, samples)
end


