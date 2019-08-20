
abstract type AbstractParticleGP <: GPBase end

mutable struct PrecomputedParticleGP <: AbstractParticleGP
    gp::Array{autobo.GPBase}
    particles::Matrix{Float64}
    weights::Vector{Float64}
    dim::Int64
    num_gpparam::Int64
    kwargs::Dict{String,Bool}

    function PrecomputedParticleGP(gp::autobo.GPBase,
                                   particles::AbstractMatrix,
                                   weights::AbstractVector;
                                   noise=true, domean=false, kern=true)
        kwargs      = Dict(["noise"=>noise, "domean"=>domean, "kern"=>kern])
        models      = typeof(gp)[]
        for i = 1:length(weights)
            local_model = deepcopy(gp)
            autobo.set_params!(local_model, particles[:,i],
                               noise=noise, domean=domean, kern=kern)
            push!(models, autobo.update_mll!(local_model))
        end
        return new(models, particles, weights, gp.dim, 2, kwargs)
    end
end

function PrecomputedParticleGP(gp::autobo.GPBase,
                               particles::AbstractMatrix;
                               kwargs...)
    num_particles = size(particles)[2]
    return PrecomputedParticleGP(gp, particles,
                                 ones(num_particles)/num_particles;
                                 kwargs...)
end

function ParticleGP(gp::autobo.GPBase,
                    particles::AbstractMatrix,
                    kwargs...)
    num_particles = size(particles)[2]
    return PrecomputedParticleGP(
        gp, particles, ones(num_particles)/num_particles; kwargs...)
end

