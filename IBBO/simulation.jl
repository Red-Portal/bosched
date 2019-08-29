
using Distributions
using Statistics
using Random
using DataFrames

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
        b = P / (2 * √R) * θ[:param]
        K = R / (2 * P)
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
    i    = 1
    hist = zeros(Float64, P)
    while(i < N)
        R = N - i + 1
        p = argmin(hist)
        K = chunk!(sched, i, R, P, N, h, dist, θ)
        K = min(R, K)

        work     = rand(prng, dist, K)
        hist[p] += h + sum(work)

        i += K
    end
    max_idx = argmax(hist)
    min_idx = argmin(hist)

    time  = hist[max_idx]
    slow  = hist[max_idx] - hist[min_idx]
    speed = sum(hist) / time
    effi  = speed / P
    μ     = mean(hist)
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
    println(df.exec_mean[1]       , ",", df.exec_conf[1],
            df.slow_mean[1]       , ",", df.slow_conf[1],
            df.speedup_mean[1]    , ",", df.speedup_conf[1],
            df.efficiency_mean[1] , ",", df.efficiency_conf[1],
            df.cov_mean[1]        , ",", df.cov_conf[1])
end

function main()
    
end
