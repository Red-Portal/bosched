"""
    SEIso <: Isotropic{SqEuclidean}
Isotropic Squared Exponential kernel (covariance)
```math
k(x,x') = σ²\\exp(- (x - x')ᵀ(x - x')/(2ℓ²))
```
with length scale ``ℓ`` and signal standard deviation ``σ``.
"""
mutable struct Exp{T<:Real} <: Kernel
    α::T
    β::T
    "Priors for kernel parameters"
    priors::Array
end

"""
Squared Exponential kernel function
    SEIso(ll::T, lσ::T)
# Arguments:
- `ll::Real`: length scale (given on log scale)
- `lσ::Real`: signal standard deviation (given on log scale)  
"""
SEIso(ll::T, lσ::T) where T = SEIso{T}(exp(2 * ll), exp(2 * lσ), [])

function set_params!(kern::Exp, hyp::AbstractVector)
    length(hyp) == 2 || throw(ArgumentError("Squared exponential has two parameters, received $(length(hyp))."))
    se.ℓ2, se.σ2 = exp(2 * hyp[1]), exp(2 * hyp[2])
end

get_params(kern::Exp{T}) where T = T[log(se.ℓ2) / 2, log(se.σ2) / 2]
get_param_names(kern::Exp) = [:ll, :lσ]
num_params(kern::Exp) = 2

cov(kern::Exp, r::Number) = se.σ2*exp(-0.5*r/se.ℓ2)

@inline dk_dll(kern::Exp, r::Real) = r/se.ℓ2*cov(se,r)
@inline function dk_dθp(kern::Exp, r::Real, p::Int)
    if p==1
        return dk_dll(se, r)
    elseif p==2
        return dk_dlσ(se, r)
    else
        return NaN
    end
end
