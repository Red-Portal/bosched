
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
    [:α, :β, :lσ]
end
GaussianProcesses.num_params(kern::Exp) = 3

function GaussianProcesses.cov(kern::Exp, x::AbstractVector, y::AbstractVector)
    β    = kern.β
    α    = kern.α
    σ2   = kern.σ2
    k    = σ2*(β ./ (x[1] + y[1] .+ β)).^α
    k
end

# function GaussianProcesses.cov(kern::Exp,
#                                X::AbstractMatrix,
#                                Y::AbstractMatrix,
#                                data::GaussianProcesses.KernelData)
#     β  = kern.β
#     α  = kern.α
#     σ2 = kern.σ2
#     @views T    = X[1,:] .+ Y[1,:]'
#     @views mask = Int64.(X[2:end,:] == Y[2:end,:])
#     mask = mask .* mask'
#     K_t  = σ2 * (β ./ (T .+ β)).^α
#     K_t .* mask
#     @info K_t mask
# end

@inline function GaussianProcesses.dKij_dθp(kern::Exp,
                                            X1::AbstractMatrix,
                                            X2::AbstractMatrix,
                                            i::Int, j::Int, p::Int, dim::Int) 
    if p==1
        α = kern.α
        β = kern.β
        @views t = X1[1,i] + X2[1,j]
        γ = (β / (t + β))
        return  kern.σ2 * γ^α * log(γ) * α
    elseif p==2
        α = kern.α
        β = kern.β
        @views t = X1[1,i] + X2[1,j]
        γ = (β / (t + β))
        return kern.σ2 * α * γ^α * (1 - γ)
    elseif p==3
        return @views 2*cov(kern, X1[1:1,i], X2[1:1,j])
    else
    end
end
