
using Distributions
using Statistics
using Random
using DataFrames
using Base.Threads
using ProgressMeter
using DelimitedFiles

abstract type Schedule end
abstract type FSS  <: Schedule end
abstract type CSS  <: Schedule end
abstract type TSS  <: Schedule end
abstract type QSS  <: Schedule end
abstract type TAPER <: Schedule end

abstract type FAC2 <: Schedule end

abstract type BO_FSS  <: Schedule end
abstract type BO_FAC  <: Schedule end
abstract type BO_CSS  <: Schedule end
abstract type BO_TSS  <: Schedule end
abstract type BO_TAPER <: Schedule end

function chunk!(::Type{BO_TSS}, i, R, P, N, h, dist, θ::Dict)
    δ  = θ[:param]
    Kl = 1
    Kf = √(1 + 2*N*δ)
    if(i == 1)
        θ[:K_prev] = Kf
        return floor(Int64, Kf)
    end
    K = max(θ[:K_prev] - δ, Kl)
    θ[:K_prev] = K
    return ceil(Int64, K)
end

function chunk!(::Type{TSS}, i, R, P, N, h, dist, θ::Dict)
    Kf = θ[:K_first]
    Kl = θ[:K_last]
    if(i == 1)
        θ[:K_prev] = Float64(Kf)
        return ceil(Int64, Kf)
    end

    C = 2 * N / (Kf + Kl)
    δ = (Kf - Kl) / (C - 1)
    K = max(θ[:K_prev] - δ, Kl)
    θ[:K_prev] = K
    return ceil(Int64, K)
end

function chunk!(::Type{TAPER}, i, R, P, N, h, dist, θ::Dict)
    α  = θ[:α]
    Km = θ[:K_min]
    μ  = mean(dist)
    σ  = std(dist)
    v  = α*σ/μ

    x = R / P + Km / 2
    K = max(Km, ceil(Int64, x + v^2 / 2 - v * √(2 * x + v^2/4)))
    return K
end

function chunk!(::Type{BO_TAPER}, i, R, P, N, h, dist, θ::Dict)
    Km = 1
    μ  = mean(dist)
    σ  = std(dist)
    v  = θ[:param]

    x = R / P + Km / 2
    K = max(Km, ceil(Int64, x + v^2 / 2 - v * √(2 * x + v^2/4)))
    return K
end

function chunk!(::Type{FAC2}, i, R, P, N, h, dist, θ::Dict)
    if(!haskey(θ, :index) || θ[:index] == 1)
        θ[:index] = P
        μ = mean(dist)
        σ = std(dist)
        b = (P * σ) / (2 * √R * μ)
        K = R / (2 * P)
        K = ceil(Int64, K)
        θ[:chunk] = K
        return K
    else
        θ[:index] -= 1
        return θ[:chunk]
    end
end

function chunk!(::Type{BO_FAC}, i, R, P, N, h, dist, θ::Dict)
    if(!haskey(θ, :index) || θ[:index] == 1)
        θ[:index] = P
        μ = mean(dist)
        σ = std(dist)
        K = R / P / θ[:param]
        K = ceil(Int64, K)
        θ[:chunk] = K
        return K
    else
        θ[:index] -= 1
        return θ[:chunk]
    end
end

function chunk!(::Type{FSS}, i, R, P, N, h, dist, θ::Dict)
    if(!haskey(θ, :index) || θ[:index] == 1)
        θ[:index] = P
        μ = mean(dist)
        σ = std(dist)
        b = (P * σ) / (2 * √R * μ)
        x = begin
                if(i == 1)
                    1 + b^2 + b * √(b^2 + 4)
                else
                    2 + b^2 + b * √(b^2 + 4)
                end
            end
        K = R / (x * P)
        K = ceil(Int64, K)
        θ[:chunk] = K
        return K
    else
        θ[:index] -= 1
        return θ[:chunk]
    end
end

function chunk!(::Type{BO_FSS}, i, R, P, N, h, dist, θ::Dict)
    if(!haskey(θ, :index) || θ[:index] == 1)
        θ[:index] = P
        μ = mean(dist)
        σ = std(dist)
        b = P / (2 * √R) * θ[:param]
        x = begin
                if(i == 1)
                    1 + b^2 + b * √(b^2 + 4)
                else
                    2 + b^2 + b * √(b^2 + 4)
                end
            end
        K = R / (x * P)
        K = ceil(Int64, K)
        θ[:chunk] = K
        return K
    else
        θ[:index] -= 1
        return θ[:chunk]
    end
end

function chunk!(::Type{CSS}, i, R, P, N, h, dist, θ::Dict)
    μ = mean(dist)
    σ = std(dist)
    K = ((h*√2*N) / (σ*P*√log(P)))^(2/3)
    return ceil(Int64, K)
end

function chunk!(::Type{BO_CSS}, i, R, P, N, h, dist, θ::Dict)
    μ = mean(dist)
    σ = std(dist)
    K = ((√2*N) / (P*√log(P)) * θ[:param])^(2/3)
    return ceil(Int64, K)
end

function simulate(sched::Type{<:Schedule}, prng, dist, P, N, h,
                  θ::Dict{Symbol, Any})
    i       = 1
    hist    = zeros(Float64, P)
    total_w = 0.0
    while(i < N)
        R = N - i + 1
        p = argmin(hist)
        K = chunk!(sched, i, R, P, N, h, dist, θ)
        K = min(R, K)

        work     = rand(prng, dist, K)
        hist[p] += h + sum(work)
        total_w += sum(work)

        i += K
    end
    max_idx = argmax(hist)
    min_idx = argmin(hist)

    time  = hist[max_idx]
    slow  = hist[max_idx] - hist[min_idx]
    speed = total_w / time
    effi  = speed / P
    μ     = total_w / P
    cov   = stdm(hist, μ) / μ
    return time, slow, speed, effi, cov
end

function conf(iter)
    # defaults to 95% confidence
    μ = mean(iter)
    σ = stdm(iter, μ)
    return σ * 1.96 / length(iter)
end

function run(sched::Type{<:Schedule}, iters, prng, dist, P, N, h,
             θ::Dict{Symbol, Any})
    df = DataFrame(exec = Float64[],
                   slow = Float64[],
                   speedup = Float64[],
                   efficiency = Float64[],
                   cov= Float64[])
    for i = 1:iters
        result = simulate(sched::Type{<:Schedule}, prng, dist, P, N, h, θ)
        push!(df, result)
    end
    return aggregate(df, [mean, conf])
end

function run_optimize(sched::Type{<:Schedule}, iters, prng, dist, P, N, h,
                      θ::Dict{Symbol, Any})
    df = DataFrame(exec = Float64[],
                   slow = Float64[],
                   speedup = Float64[],
                   efficiency = Float64[],
                   cov= Float64[])
    for i = 1:iters
        result = simulate(sched::Type{<:Schedule}, prng, dist, P, N, h, θ)
        push!(df, result)
    end
    return aggregate(df, [mean, conf])
end

function print_csv(df::DataFrame)
    println("exec,+-,slow,+-,speed,+-,eff,+-,cov,+-")
    for i = 1:size(df, 1)
        println(df.exec_mean[i]       , ",", df.exec_conf[i], ",",
                df.slow_mean[i]       , ",", df.slow_conf[i], ",",
                df.speedup_mean[i]    , ",", df.speedup_conf[i], ",",
                df.efficiency_mean[i] , ",", df.efficiency_conf[i], ",",
                df.cov_mean[i]        , ",", df.cov_conf[i])
    end
end

function make_2d(filename, x_labels, y_labels, arr)
    arr      = reshape(arr, (length(x_labels), length(y_labels)))
    arr      = hcat(x_labels, arr)
    y_labels = vcat(Nothing, y_labels)
    y_labels = convert(Array{Any, 2},
                       reshape(y_labels, (1, length(y_labels))))
    arr      = convert(Array{Any, 2}, arr)
    arr      = vcat(y_labels, arr)

    writedlm(filename, arr, ',')
    # open(filename, "w+") do f
    #     s = read(f, String)
    #     println(s)
    #     s = replace(s, "Nothing"=>"")
    #     write(io, s)
    # end
end

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
