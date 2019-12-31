# Exponential mean function

"""
    MeanExp <: Mean
Polynomial mean function
```math
m(x) = ∑ᵢⱼ βᵢⱼxᵢʲ
```
with polynomial coefficients ``βᵢⱼ`` of shape ``d × D`` where ``d`` is the dimension of
observations and ``D`` is the degree of the polynomial.
"""
mutable struct MeanExp <: GaussianProcesses.Mean
    "Polynomial coefficients"
    α::Float64
    β::Float64
    μ::Float64
    "Priors for mean parameters"
    priors::Array

    """
        MeanPoly(β::Matrix{Float64})
    Create `MeanExp` with polynomial coefficients `β`.
    """
    MeanExp(α::Real, β::Real, μ::Real) = new(α, β, μ, [])
    MeanExp(α::Real, β::Real, μ::Real, priors) = new(α, β, μ, priors)
end

function GaussianProcesses.mean(mExp::MeanExp, x::AbstractVector)
    length(x) == 2 || throw(ArgumentError("Observations and mean function have inconsistent dimensions"))
    return -exp(-mExp.α*x[1] + mExp.β) + mExp.μ
end


GaussianProcesses.get_params(mExp::MeanExp) = [mExp.α, mExp.β, mExp.μ]
GaussianProcesses.get_param_names(mExp::MeanExp) = [:α, :β, :μ]
GaussianProcesses.num_params(mExp::MeanExp) = 3

function GaussianProcesses.set_params!(mExp::MeanExp, hyp::AbstractVector)
    length(hyp) == 3 || throw(ArgumentError("Polynomial mean function has $(num_param) parameters"))
    mExp.α = hyp[1]
    mExp.β = hyp[2]
    mExp.μ = hyp[3]
end


function GaussianProcesses.grad_mean(mExp::MeanExp, x::AbstractVector)
    fx = mean(mExp, x)
    dα = x[1]*fx
    dβ = -fx
    dμ = 1
    return [dα dβ dμ]
end
