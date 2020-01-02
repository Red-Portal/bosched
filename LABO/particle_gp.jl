
abstract type AbstractParticleGP <: GPBase end

mutable struct ParticleGP <: AbstractParticleGP
    gp::Array{GPBase}
    η::Vector{Float64}
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
        return new(models, [], weights, gp.dim, nparam, kwargs)
    end
end

mutable struct TimeMarginalizedGP <: AbstractParticleGP
    non_marg_gp::GPBase
    time_idx::Array{Int64}
    time_w::Array{Float64}
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

function GaussianProcesses.predict_y(gp::TimeMarginalizedGP, x::AbstractArray)
    t   = gp.time_idx
    μs  = zeros(length(x))
    σ²s = zeros(length(x))
    xt  = zeros(2, length(t))
    for i = 1:length(x)
        xt[1,:] .= t
        xt[2,:] .= x[i] 
        μ, σ² = predict_y(gp.non_marg_gp, xt)
        μ     = dot(μ, gp.time_w)
        σ²    = dot(σ², gp.time_w)

        μs[i]  = μ
        σ²s[i] = σ²
    end
    return μs, σ²s
end
