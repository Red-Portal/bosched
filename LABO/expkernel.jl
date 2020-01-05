
mutable struct Exp{T<:Real} <: GaussianProcesses.Kernel
    α::T
    β::T
    σ2::T
    "Priors for kernel parameters"
    priors::Array
end

function GaussianProcesses.set_params!(kern::Exp, hyp::AbstractVector)
    length(hyp) == 3 || throw(ArgumentError("Exponential decrease has three parameters, received $(length(hyp))."))
    kern.α  = exp(hyp[1])
    kern.β  = exp(hyp[2])
    kern.σ2 = exp(hyp[3]*2)
end

function GaussianProcesses.get_params(kern::Exp{T}) where T
    T[log(kern.α), log(kern.β), log(kern.σ2)/2]
end
function GaussianProcesses.get_param_names(kern::Exp)  
    [:α, :β, :σ2]
end
GaussianProcesses.num_params(kern::Exp) = 3

function GaussianProcesses.cov(kern::Exp, x::AbstractVector, y::AbstractVector)
    # (x, t)
    β  = kern.β
    α  = kern.α
    σ2 = kern.σ2
    return σ2 * (β ./ (x[1] + y[1] .+ β)).^α
end

function GaussianProcesses.cov(kern::Exp,
                               X::AbstractMatrix,
                               Y::AbstractMatrix,
                               data::GaussianProcesses.KernelData)
    β   = kern.β
    α   = kern.α
    σ2  = kern.σ2
    @views K_t = σ2*(β ./ (X[1,:] .+ Y[1,:]' .+ β)).^α
    K_t
end

@inline function GaussianProcesses.dKij_dθp(kern::Exp,
                                            X1::AbstractMatrix,
                                            X2::AbstractMatrix,
                                            i::Int, j::Int, p::Int, dim::Int) 
    if p==1
        α = kern.α
        β = kern.β
        σ2 = kern.σ2
        t = X1[1,i] + X2[1,j]
        return σ2 * (β / (t + β))^α * log(β / (t + β))
    elseif p==2
        α = kern.α
        β = kern.β
        σ2 = kern.σ2
        t = X1[1,i] + X2[1,j]
        return σ2 * (β / (t + β))^(α+1) * α * (t / β.^2)
    elseif p==3
        return 2 * √kern.σ2 * GaussianProcesses.cov(kern, X1[:,i], X2[:,j])
    end
end

