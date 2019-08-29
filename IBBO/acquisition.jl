
using SpecialFunctions
using Roots
using Distributions

normal_pdf(μ, σ²) = 1/√(2π*σ²) * exp(-μ^2/(2*σ²))
normal_cdf(μ, σ²) = 1/2 * (1 + erf(μ/√(2σ²)))

function sample_ystar(idx::Int64,
                      η::Float64
                      num_samples::Int64,
                      num_grid_points::Int64,
                      gp::PrecomputedParticleGP) where {G}
    D      = gp.dim
    xgrid  = rand(D, num_grid_points)
    μ, σ²  = predict_f(gp.gp[idx], xgrid)
    σe     = exp.(gp.particles[1,idx])
    σ²     = max.(σ², σe^2 + 1e-10)
    σ      = sqrt.(σ²)
    ystar  = Vector{Float64}(undef, num_samples)

    probf(x) = prod(cdf.(Normal.(μ, sqrt.(σ²)), x))
    
    left  = optimum_y
    if(probf(left) < 0.25)
        right = maximum(μ .+ 5*σ)
        while(probf(right) < 0.75)
            right += right - left
        end
        md = find_zero(x->(probf(x) - 0.5), (left, right), atol=0.01, order=0)
        q1 = find_zero(x->(probf(x) - 0.25), (left, md), atol=0.01, order=0)
        q2 = find_zero(x->(probf(x) - 0.75), (md, right), atol=0.01, order=0)
        Gθ = (q1 - q2) / (log(log(4/3)) - log(log(4)))
        Gμ = md + Gθ*log(log(2))
        @assert Gθ > 0
        ystar .= rand(Gumbel(Gμ, Gθ), num_samples)
        ystar .= max.(ystar, left .+ 5*σe)
    else
        ystar .= left + 5*σe
    end
    return ystar
end

function acquisition(x::AbstractArray, gp::PrecomputedParticleGP)
    P  = length(gp.weights)
    N  = size(x, 2)
    w  = gp.weights
    η  = gp.particles[gp.num_gpparam+1:end,:]
    σe = exp.(gp.particles[1,:])

    α_res = Matrix{eltype(x)}(undef, N, P)
    for i = 1:P
        μ, σ² = predict_f(gp.gp[i], x)
        σ²    += σe[i]^2
        α_res[:,i] .= α.(μ, σ², t, Ref(η[:,i]))
    end
    res = α_res * w 
    if(N == 1)
        res = res[1]
    end
    return res
end

@inline function α(μ, σ², t, η)
"""
 "MES Acquisition Function"
 Wang, Zi, and Stefanie Jegelka. "Max-value entropy search for efficient 
 Bayesian optimization." Proceedings of the 34th International Conference 
 on Machine Learning-Volume 70. JMLR. org, 2017.
"""
    dist = Normal()
    z = (η .- μ) / sqrt.(σ²)
    ϕ = pdf.(dist, z)
    Φ = cdf.(dist, z) .+ 1e-20
    return mean(z .* ϕ ./ (2 * Φ) - log.(Φ))
end
