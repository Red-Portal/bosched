
using Distributions

function batch_slice_sampler(p, batch_size, num_samples, max_proposals)
    xdom     = (0.0, 1.0)
    proposal = Uniform(xdom[1], xdom[2])
    samples  = Float64[]
    num_acc  = 0
    num_prop = 0
    s        = rand(proposal, batch_size)
    α        = p.(s)
    u        = zeros(batch_size)
    acc      = trues(batch_size)
    while(length(samples) < num_samples && num_prop < max_proposals)
        u[acc] = rand.(Uniform.(0.0, α[acc]))
        s      = rand(proposal, batch_size)
        α      = p(s[:,:]')
        acc    = α .> u

        num_acc  += count(acc)
        num_prop += length(u)
        samples   = vcat(samples, s[acc])
    end
    println("acceptance: $(num_acc / num_prop), samples: $(length(samples))")
    return samples
end
