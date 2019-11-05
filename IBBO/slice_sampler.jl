
import ProgressMeter
using Distributions
using Random

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
        α_acc  = α[acc]
        u[acc] = α_acc - rand.(Exponential(1.0), length(α_acc))
        s      = rand(proposal, batch_size)
        α      = [p(s[i]) for i = 1:batch_size]
        acc    = α .> u
    end

    while(length(samples) < num_samples && num_prop < max_proposals)
        α_acc  = α[acc]
        u[acc] = α_acc - rand.(Exponential(1.0), length(α_acc))
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

function slice_sampler(p, num_samples, burnin)
    function sample(x0::Float64, g::Function, w::Float64, m::Int64,
                    lower::Float64, upper::Float64, gx0::Float64)
        logy = gx0 - rand(Exponential(1.0))
        u    = rand() * w
        L    = x0 - u

        R    = x0 + (w-u)
        J    = floor(rand() * m)
        K    = (m-1) - J

        while J > 0 && L > lower && g(L) > logy
            L -= w
            J -= 1
        end

        while K > 0 && R < upper && g(R) > logy
            R += w
            K -= 1
        end

        L = L < lower ? lower : L
        R = R > upper ? upper : R

        props = 0
        x1    = 0.0  # need to initialize it in this scope first
        gx1   = 0.0
        while true 
            props += 1
            x1 = rand() * (R-L) + L
            gx1 = g(x1)::Float64
            if gx1 >= logy
                break
            end
            if x1 > x0
                R = x1
            else
                L = x1
            end
        end
        return x1, gx1, props
    end

    samples  = zeros(num_samples) 
    xdom     = (0.0, 1.0)
    proposal = Uniform(xdom[1], xdom[2])
    x_curr   = rand(proposal)
    p_x      = p(x_curr)

    prog = ProgressMeter.Progress(burnin + num_samples, "Sampling acquisition function...")
    for i = 1:burnin
        x_curr, p_x, props = sample(x_curr, p, 0.1, 16, xdom[1], xdom[2], p_x)
        ProgressMeter.next!(prog)
    end

    total_props = 0
    for i = 1:num_samples
        x_curr, p_x, props = sample(x_curr, p, 0.1, 16, xdom[1], xdom[2], p_x)
        samples[i]   = x_curr
        total_props += props
        ProgressMeter.next!(prog)
    end
    println("acceptance: $(num_samples / total_props), samples: $(length(samples))")
    return samples
end
