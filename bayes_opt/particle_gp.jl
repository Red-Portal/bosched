
abstract type AbstractParticleGP <: GPBase end

mutable struct ParticleGP <: AbstractParticleGP
    gp::Array{GPBase}
    particles::Matrix{Float64}
    weights::Vector{Float64}
    dim::Int64
    num_gpparam::Int64
    kwargs::Dict{String,Bool}

    function ParticleGP(gp::GPBase,
                        particles::AbstractMatrix,
                        weights::AbstractVector;
                        noise=true, domean=false, kern=true)
        kwargs = Dict(["noise"=>noise, "domean"=>domean, "kern"=>kern])
        models = typeof(gp)[]
        for i = 1:length(weights)
            local_model = deepcopy(gp)
            GaussianProcesses.set_params!(
                local_model, particles[:,i],
                noise=noise, domean=domean, kern=kern)
            push!(models, GaussianProcesses.update_mll!(local_model))
        end
        nparam = GaussianProcesses.num_params(gp, domean=false)
        return new(models, particles, weights, gp.dim, nparam, kwargs)
    end
end

function ParticleGP(gp::GPBase,
                    particles::AbstractMatrix;
                    kwargs...)
    num_particles = size(particles)[2]
    return ParticleGP(gp, particles,
                      ones(num_particles)/num_particles;
                      kwargs...)
end

function GaussianProcesses.predict_y(gp::ParticleGP, x::AbstractArray)
    bundle = GaussianProcesses.predict_y.(gp.gp, Ref(x))
    μ      = [elem[1] for elem in bundle]
    σ²     = [elem[2] for elem in bundle]

    μ  = hcat(μ...) * gp.weights
    σ² = hcat(σ²...) * gp.weights
    return μ, σ²
end

function predict_marg_y(gp::GPBase, x::AbstractArray, max_t::Int64)
    t   = collect(1:max_t)
    μs  = zeros(length(x))
    σ²s = zeros(length(x))
    xt  = zeros(2, max_t)
    for i = 1:length(x)
        xt[1,:] .= x[i]
        xt[2,:] .= t
        μ, σ² = predict_y(gp, xt)
        μ  = sum(μ)
        σ² = sum(σ²)

        μs[i]  = μ
        σ²s[i] = σ²
    end
    return μs, σ²s
end

