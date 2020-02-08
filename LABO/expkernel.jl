
mutable struct Exp{T<:Real} <: GaussianProcesses.Kernel
    α::T
    β::T
    "Priors for kernel parameters"
    priors::Array
end

function GaussianProcesses.set_params!(kern::Exp, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Exponential decrease has three parameters, received $(length(hyp))."))
    kern.α  = exp(hyp[1])
    kern.β  = exp(hyp[2])
end

function GaussianProcesses.get_params(kern::Exp{T}) where T
    T[log(kern.α), log(kern.β)]
end
function GaussianProcesses.get_param_names(kern::Exp)  
    [:α, :β]
end
GaussianProcesses.num_params(kern::Exp) = 2

function GaussianProcesses.cov(kern::Exp, x::AbstractVector, y::AbstractVector)
    # (x, t)
    β  = kern.β
    α  = kern.α
    return (β ./ (x[1] + y[1] .+ β)).^α
end

function GaussianProcesses.cov(kern::Exp,
                               X::AbstractMatrix,
                               Y::AbstractMatrix,
                               data::GaussianProcesses.KernelData)
    β   = kern.β
    α   = kern.α
    @views K_t = (β ./ (X[1,:] .+ Y[1,:]' .+ β)).^α
    K_t
end

@inline function GaussianProcesses.dKij_dθp(kern::Exp,
                                            X1::AbstractMatrix,
                                            X2::AbstractMatrix,
                                            i::Int, j::Int, p::Int, dim::Int) 
    if p==1
        α = kern.α
        β = kern.β
        t = X1[1,i] + X2[1,j]
        return (β / (t + β))^α * log(β / (t + β))
    elseif p==2
        α = kern.α
        β = kern.β
        t = X1[1,i] + X2[1,j]
        return (β / (t + β))^(α+1) * α * (t / β.^2)
    end
end

