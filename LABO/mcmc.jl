
function get_means(gp::GaussianProcesses.GPE; noise::Bool=true,
                   domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise && GaussianProcesses.num_params(gp.logNoise) != 0
        noise_sample = mean(Product(GaussianProcesses.get_priors(gp.logNoise)))
        append!(params, noise_sample)
    end
    if domean && GaussianProcesses.num_params(gp.mean) != 0
        mean_sample = mean(Product(GaussianProcesses.get_priors(gp.mean)))
        append!(params, mean_sample)
    end
    if kern && GaussianProcesses.num_params(gp.kernel) != 0
        kernel_sample = mean(Product(GaussianProcesses.get_priors(gp.kernel)))
        append!(params, kernel_sample)
    end
    return params
end

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

    p = Progress(num_samples + num_adapts, 1)
    samples = zeros(Float64, length(θ_cur), num_samples)
    for i = 1:num_adapts
        θ_cur = sample!(θ_cur, width, targetlogp)
        next!(p)
    end

    for i = 1:num_samples
        θ_cur        = sample!(θ_cur, width, targetlogp)
        samples[:,i] = θ_cur
        next!(p)
    end
    samples = samples[:,1:thinning:end]
    ESS = [MCMCDiagnostics.effective_sample_size(samples[i,:])
           for i = 1:size(samples,1)]
    println("ESS = $(ESS)")
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
            return (gp.target, gp.dtarget)
        catch
            return (-Inf, zeros(length(gp.dtarget)))
        end
    end

    function logp(θ::AbstractVector)
        try
            GaussianProcesses.set_params!(gp, θ; params_kwargs...)
            GaussianProcesses.update_target!(gp; params_kwargs...)
            return gp.target
        catch
            return -Inf
        end
    end

    θ_init  = GaussianProcesses.get_params(gp; params_kwargs...)
    metric  = DenseEuclideanMetric(length(θ_init))
    #metric  = DiagEuclideanMetric(length(θ_init))
    h       = Hamiltonian(metric, logp, logp∂logp)
    prop    = NUTS(Leapfrog(find_good_eps(h, θ_init)))
    adaptor = StanHMCAdaptor(Preconditioner(metric),
                             NesterovDualAveraging(0.65, prop.integrator.ϵ))

    # Draw samples via simulating Hamiltonian dynamics
    # - `samples` will store the samples
    # - `stats` will store statistics for each sample
    samples, stats = AdvancedHMC.sample(
        h, prop, θ_init, num_samples + num_adapts, adaptor,
        num_adapts; progress=verbose, drop_warmup=true)
    samples = hcat(samples...)
    samples = samples[:,1:thinning:end]
    ESS     = [MCMCDiagnostics.effective_sample_size(samples[i,:])
               for i = 1:size(samples,1)]
    display(mean(samples, dims=2)[:,1])
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
    means = get_means(gp; params_kwargs...)
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

    function sample!(f::AbstractArray{Float64, 1})
        v_biased = GaussianProcesses.sample_params(gp; params_kwargs...) 
        v        = v_biased - means
        f        = f - means

        u     = rand()
        logy  = calc_target!(f + means) + log(u);
        θ     = rand()*2*π;
        θ_min = θ - 2*π;
        θ_max = θ;
        f_prime = f * cos(θ) + v * sin(θ);
        while  calc_target!(f_prime + means) <= logy
            if θ < 0
                θ_min = θ;
            else
                θ_max = θ;
            end
            θ = rand() * (θ_max - θ_min) + θ_min;
            f_prime = f * cos(θ) + v * sin(θ);
        end
        return f_prime + get_means(gp; params_kwargs...)
    end

    total_proposals = 0
    θ_cur = GaussianProcesses.get_params(gp; params_kwargs...)
    D = length(θ_cur)
    post = Array{Float64}(undef, num_samples+num_adapts, D)

    ProgressMeter.@showprogress for i = 1:num_samples+num_adapts
        θ_cur     = sample!(θ_cur)
        post[i,:] = θ_cur
    end
    post = post[num_adapts:thinning:end,:]
    samples = post'
    if(verbose)
        ESS = [MCMCDiagnostics.effective_sample_size(samples[i,:])
               for i = 1:size(samples,1)]
        println("Number of function calls: ", count)
#        println("Acceptance rate: $(num_samples / count) \n")
        println("ESS: $(ESS)")
    end
    return ParticleGP(gp, samples)
end
