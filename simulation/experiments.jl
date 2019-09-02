
include("simulation.jl")

function experiment1_1(prng)
    df = DataFrame()
    axis1 = collect(-7:0.3:7)
    axis2 = collect(-15:0.5:7)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 2^y
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 0.1
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FSS, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment1_2(prng)
    df = DataFrame()
    axis1 = collect(-7:0.3:7)
    axis2 = collect(0:0.2:8)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 2^y
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 0.1
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FAC, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment1_3(prng)
    df = DataFrame()
    axis1 = collect(-7:0.3:7)
    axis2 = collect(-10:0.5:10)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 2^y
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 0.1
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_CSS, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment2_1(prng)
    df = DataFrame()
    axis1 = collect(8:0.3:17)
    axis2 = collect(-15:0.5:7)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j = 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 1.0
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 0.1
            N     = floor(Int64, 2^y)
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FSS, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment2_2(prng)
    df = DataFrame()
    axis1 = collect(8:0.3:17)
    axis2 = collect(0:0.2:8)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 1.0
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 0.1
            N     = floor(Int64, 2^y)
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FAC, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment2_3(prng)
    df = DataFrame()
    axis1 = collect(8:0.3:17)
    axis2 = collect(-10:0.5:10)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 1.0
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 0.1
            N     = floor(Int64, 2^y)
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_CSS, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment3_1(prng)
    df = DataFrame()
    axis1 = collect(-4:0.2:4)
    axis2 = collect(-15:0.5:7)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            λ     = 2^y
            dist  = Exponential(λ);
            iters = 128
            h     = 1.0
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FSS, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment3_2(prng)
    df = DataFrame()
    axis1 = collect(-4:0.2:4)
    axis2 = collect(0:0.2:8)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            λ     = 2^y
            dist  = Exponential(λ);
            iters = 128
            h     = 1.0
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FAC, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end


function experiment4_1(prng)
    df = DataFrame()
    axis1 = collect(-7:0.2:7)
    axis2 = collect(-5:0.5:15)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 1.0
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 2^y
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FSS, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment4_2(prng)
    df = DataFrame()
    axis1 = collect(-7:0.2:7)
    axis2 = collect(0:0.2:8)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 1.0
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 2^y
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_FAC, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function experiment4_3(prng)
    df = DataFrame()
    axis1 = collect(-7:0.2:7)
    axis2 = collect(-10:0.5:10)

    @showprogress "Running simulation..." for (i, y) in enumerate(axis1)
        seeds  = MersenneTwister.(rand(prng, UInt64, length(axis2)))
        outbuf = Array{DataFrame}(undef, length(axis2))

        Threads.@threads for j in 1:length(axis2)
            x     = axis2[j]
            μ     = 1.0 
            σ     = 1.0
            dist  = TruncatedNormal(μ, σ, 0.0, Inf);
            iters = 128
            h     = 2^y
            N     = 2^14
            P     = 32
            
            θ   = Dict{Symbol, Any}(:param=>Float64(2^x));
            res = run(BO_CSS, iters, seeds[j], dist, P, N, h, θ)
            outbuf[j] = res
        end
        df = vcat(df, outbuf...)
    end
    return df, 2 .^axis2, 2 .^axis1
end

function exectime_cdf1(prng, sched, θ)
    dist = TruncatedNormal(1.0, 1.0, 0.0, Inf)
    P    = 128
    N    = 2^10
    h    = 0.1

    num_samples = 2^12
    samples = zeros(num_samples)
    Threads.@threads for i = 1:num_samples
        result     = simulate(sched, prng, dist, P, N, h,
                              Dict{Symbol, Any}(:param=>θ))
        samples[i] = result[1]
    end
    return samples
end

function exectime_cdf2(prng, sched, θ)
    dist = Exponential(1.0)
    P    = 128
    N    = 2^10
    h    = 0.1

    num_samples = 2^12
    samples = zeros(num_samples)
    Threads.@threads for i = 1:num_samples
        result     = simulate(sched, prng, dist, P, N, h,
                              Dict{Symbol, Any}(:param=>θ))
        samples[i] = result[1]
    end
    return samples
end
