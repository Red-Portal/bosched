
using AdvancedHMC

function nuts(gp, num_samples=100, num_adapts=100;
              precomputed=false, verbose=false)
    function logp∂logp(θ::AbstractVector)
        set_params!(gp, θ; domean=false, kern=true, noise=true)
        update_target_and_dtarget!(
            gp; domean=false, kern=true, noise=true)
        return (gp.target, gp.dtarget)
    end

    function logp(θ::AbstractVector)
        return gp.mll
    end

    θ_init  = get_params(gp; domean=false, kern=true, noise=true)
    metric  = DiagEuclideanMetric(2)
    h       = Hamiltonian(metric, logp, logp∂logp)
    prop    = NUTS(Leapfrog(find_good_eps(h, θ_init)))
    adaptor = StanHMCAdaptor(num_adapts,
                             Preconditioner(metric),
                             NesterovDualAveraging(0.8, prop.integrator.ϵ))

    # Draw samples via simulating Hamiltonian dynamics
    # - `samples` will store the samples
    # - `stats` will store statistics for each sample
    samples, stats = sample(h, prop, θ_init, num_samples, adaptor,
                            num_adapts; progress=verbose)
    samples = hcat(samples...)
    if(precomputed)
        return PrecomputedParticleGP(gp, samples)
    else
        return ParticleGP(gp, chain)
    end
end

