
using Distributions

function batch_slice_sampler(p, batch_size, num_samples, burnin, max_proposals)
    xdom     = (0.00, 1.00)
    proposal = Uniform(xdom[1], xdom[2])
    samples  = Float64[]
    num_acc  = 0
    num_prop = 0
    s        = rand(proposal, batch_size)
    α        = p.(s)
    u        = zeros(batch_size)
    acc      = trues(batch_size)

    for t = 1:burnin
        u[acc] = rand.(Uniform.(0.0, α[acc]))
        s      = rand(proposal, batch_size)
        α      = [p(s[i]) for i = 1:batch_size]
        acc    = α .> u
    end

    while(length(samples) < num_samples && num_prop < max_proposals)
        u[acc] = rand.(Uniform.(0.0, α[acc]))
        s      = rand(proposal, batch_size)
        α      = [p(s[i]) for i = 1:batch_size]
        acc    = α .> u

        num_acc  += count(acc)
        num_prop += length(u)
        samples   = vcat(samples, s[acc])
    end
    println("acceptance: $(num_acc / num_prop), samples: $(length(samples))")
    return samples
end

# function slice_sampler(p, num_samples, burnin, max_proposals
#     x0::Float64, g::Function, w::Float64, m::Int64, gx0::Float64)
#     function sample()
#         x0 = rand(Uniform(lower, upper))
#         w  = 0.2
#         lower = 0.0
#         upper = 1.0
#         logy::Float64 = gx0 - rand(Exponential(1.0))

#         # Find the initial interval to sample from.
#         u::Float64 = rand() * w
#         L::Float64 = x0 - u
#         R::Float64 = x0 + (w-u)

#         # Expand the interval until its ends are outside the slice, or until
#         # the limit on steps is reached.  
#         J::Int64 = floor(rand() * m)
#         K::Int64 = (m-1) - J

#         while J > 0
#             if L <= lower || p(L)::Float64 <= logy
#                 break
#             end
#             L -= w
#             J -= 1
#         end

#         while K > 0
#             if R >= upper || p(R)::Float64 <= logy
#                 break
#             end
#             R += w
#             K -= 1
#         end

#         # Shrink interval to lower and upper bounds.
#         L = L < lower ? lower : L
#         R = R > upper ? upper : R
        
#         # Sample from the interval, shrinking it on each rejection.
#         x1::Float64 = 0.0  # need to initialize it in this scope first
#         gx1::Float64 = 0.0
#         while true 
#             x1 = rand() * (R-L) + L
#             gx1 = p(x1)::Float64
#             if gx1 >= logy
#                 break
#             end
#             if x1 > x0
#                 R = x1
#             else
#                 L = x1
#             end
#         end
#         return x1,gx1
#     end
# end
