
using Distributions
using Statistics
using Random
using DataFrames
using Base.Threads
using ProgressMeter
using DelimitedFiles

abstract type Schedule end
abstract type FSS    <: Schedule end
abstract type AFAC   <: Schedule end
abstract type FAC2   <: Schedule end
abstract type CSS    <: Schedule end
abstract type TSS    <: Schedule end
abstract type QSS    <: Schedule end
abstract type HSS    <: Schedule end
abstract type BinLPT <: Schedule end
abstract type TAPER  <: Schedule end

abstract type BO_FSS   <: Schedule end
abstract type BO_FAC   <: Schedule end
abstract type BO_CSS   <: Schedule end
abstract type BO_TSS   <: Schedule end
abstract type BO_TAPER <: Schedule end

function chunk!(::Type{BO_TSS}, i, p, R, P, N, h, dist, θ::Dict)
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

function chunk!(::Type{TSS}, i, p, R, P, N, h, dist, θ::Dict)
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

function chunk!(::Type{TAPER}, i, p, R, P, N, h, dist, θ::Dict)
    α  = θ[:α]
    Km = θ[:K_min]
    μ  = mean(dist)
    σ  = std(dist)
    v  = α*σ/μ

    x = R / P + Km / 2
    K = max(Km, ceil(Int64, x + v^2 / 2 - v * √(2 * x + v^2/4)))
    return K
end

function chunk!(::Type{BO_TAPER}, i, p, R, P, N, h, dist, θ::Dict)
    Km = 1
    μ  = mean(dist)
    σ  = std(dist)
    v  = θ[:param]

    x = R / P + Km / 2
    K = max(Km, ceil(Int64, x + v^2 / 2 - v * √(2 * x + v^2/4)))
    return K
end

function chunk!(::Type{FAC2}, i, p, R, P, N, h, dist, θ::Dict)
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

function chunk!(::Type{BO_FAC}, i, p, R, P, N, h, dist, θ::Dict)
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

function chunk!(::Type{FSS}, i, p, R, P, N, h, dist, θ::Dict)
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

function chunk!(::Type{BO_FSS}, i, p, R, P, N, h, dist, θ::Dict)
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

function chunk!(::Type{CSS}, i, p, R, P, N, h, dist, θ::Dict)
    μ = mean(dist)
    σ = std(dist)
    K = ((h*√2*N) / (σ*P*√log(P)))^(2/3)
    return ceil(Int64, K)
end

function chunk!(::Type{BO_CSS}, i, p, R, P, N, h, dist, θ::Dict)
    μ = mean(dist)
    σ = std(dist)
    K = ((√2*N) / (P*√log(P)) * θ[:param])^(2/3)
    return ceil(Int64, K)
end

function chunk!(::Type{AFAC}, i, p, R, P, N, h, dist, θ::Dict)
    K = 1
    if(!haskey(θ, :index))
        θ[:index] = 1
    end

    if(θ[:index] == 0)
        return 1
    elseif(θ[:index] < P+1)
        K = R / (2 * P)
    else
        μs    = θ[:P_μ]
        σs    = θ[:P_σ]
        nz    = μs .!= 0
        μs_nz = μs[nz]
        σs_nz = σs[nz]

        D  = sum(σs_nz.^2 ./ μs_nz) 
        T  = 1/sum(1 ./ μs) 
        K  = (D + 2*T*R - √(D^2 + 4*D*T*R)) / (2 * μs[p])
    end
    θ[:index] += 1
    if(K <= 1)
        θ[:index] = 0
    end
    return ceil(Int64, K)
end

function chunk!(::Type{HSS}, i, p, R, P, N, h, dist, θ::Dict)
    k = 0
    chunksize = 0
    chunkweight = 0

    tasks = θ[:mean]
    chunksize = ceil(R/(1.5*P));
    for i = R:N
        w1 = 0
        w2 = 0

        k += 1
        chunkweight += tasks(i)

        w1 = chunkweight
        w2 = begin
            if((i + 1) <= N)
                chunkweight + tasks(i+1)
            else
                0
            end
        end

        if(w2 <= chunksize)
            continue
        end

        if(w1 >= chunksize)
            break
        end

        if((w2 - chunksize) > (chunksize - w1))
            break
        end
    end
    return k
end

function chunk!(::Type{BinLPT}, i, p, R, P, N, h, dist, θ::Dict)
    if()
        
    end
    tasks = θ[:mean]
    
    return ceil(Int64, K)
end

function sample_work(prng, dist::Distributions.Distribution, i, K, N)
    return rand(prng, dist, K)
end

function sample_work(prng, f, it, K, N)
    return [f(prng, it + i, N) for i = 1:K]
end

function simulate(sched::Type{<:Schedule}, prng, f, P, N, h,
                  θ::Dict{Symbol, Any})
    i       = 1
    hist    = zeros(Float64, P)
    total_w = 0.0

    if(sched == AFAC)
        θ[:P_μ] = zeros(P)
        θ[:P_σ] = zeros(P)
    end
    
    while(i < N)
        R = N - i + 1
        p = argmin(hist)
        K = chunk!(sched, i, p, R, P, N, h, f, θ)
        K = max(min(R, K), 1)

        work     = sample_work(prng, f, i, K, N)
        hist[p] += h + sum(work)
        total_w += sum(work)

        if(sched == AFAC)
            afac_μ     = mean(work)
            θ[:P_μ][p] = afac_μ
            θ[:P_σ][p] = stdm(work, afac_μ)
        end

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
                   cov = Float64[])
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

